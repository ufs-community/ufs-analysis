from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from posixpath import join as urljoin
import datetime
import pandas as pd
import xarray as xr

class DataReader(ABC):
    '''Abstract: Reads and contains datasets for various data assimilation modeling systems.'''
    def __init__(self, **kwargs):

        file_url = kwargs.get('file_url', None)

        if file_url is None:
            self.file_url = self._default_file
        else:
            self.file_url = file_url

        self._dataset_url = urljoin(self._base_url, self.file_url)

        if 'SUPPLIED' not in self.__class__.__name__:

            if file_url is None:
                print('No filename provided; deferring to default')

            print(f'Reading data from {self._dataset_url}')

        self._read_dataset()

        # This attribute is managed within self.retrieve using getters and setters
        self._set_retrieve_params(**{})

        # relabel dimensions and order coordinate axes
        self.dataset = DataReader.standardize_coords(self.dataset)

        print(f'Dataset ready.')

    @abstractmethod
    def _read_dataset(self):
        '''Read data from the given address'''
        pass

    @staticmethod
    def to_datetime(time_str: Union[str, datetime.datetime]) -> datetime.datetime:

        if isinstance(time_str, datetime.datetime):
            return time_str
        try:
            return datetime.datetime.fromisoformat(time_str)
        except ValueError:
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d")
            except:
                raise ValueError("Invalid time format. Use ISO format or YYYY-MM-DD")

    @staticmethod
    def standardize_coords(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:

        print(f'Standardizing coordinate system')

        coord_map = {'latitude': 'lat', 'y': 'lat', 'longitude': 'lon', 'x': 'lon', 'level': 'lev'}
        rename_dict = {}

        for old, new in coord_map.items():
            if old in ds.coords:
                rename_dict[old] = new

        if isinstance(ds, xr.Dataset):
            for old, new in coord_map.items():
                if old in ds.data_vars:
                    rename_dict[old] = new

        # Rename coordinates
        ds = ds.rename(rename_dict) if rename_dict else ds

        # Enforce order of coordinate axes
        # lat-lon order is required by spherepack.
        # init-lead order is being enforced for logical reasons.

        # First check if data actually need to be sorted.
        if sorted(list(ds.lon.values)) != list(ds.lon.values):
            print(f'Sorting lon ascending')
            ds = ds.sortby('lon', ascending=True)

        if sorted(list(ds.lat.values), reverse=True) != list(ds.lat.values):
            print(f'Sorting lat descending')
            ds = ds.sortby('lat', ascending=False)

        # Enforce a dimension order such that lat always comes before lon
        # We need to do this because ERA5 datasets can be inconsistent,
        # and because u-v analysis extracts the underlying the numpy arrays.
        # Easiest to ensure consistent shape here.
        # Don't mess with the other dimensions.

        dims = list(ds.dims)
        # Assume lat and lon adjacency
        # Find position of lat
        lat_ind = dims.index("lat")

        # Find position of lon
        lon_ind = dims.index("lon")

        change_order = False
        # if lon comes before lat, switch them
        if lon_ind < lat_ind:
            change_order = True
            dims[lon_ind] = 'lat'
            dims[lat_ind] = 'lon'

        # Check if init and lead exist, and follow the same logic
        if 'init' in dims and 'lead' in dims:
            init_ind = dims.index('init')
            lead_ind = dims.index('lead')

            if lead_ind < init_ind:
                change_order = True
                dims[lead_ind] = 'init'
                dims[init_ind] = 'lead'

        # Transpose the data.
        # Note that a chunking scheme must be present for large datasets.
        if change_order is True:
            ds = ds.transpose(*dims)  # Reorder

        return ds

    def info(self):
        if self.dataset is None:
            print("Dataset not available")
            return
        print(self.dataset)

    def list_variables(self):
        if self.dataset is None:
            print("Dataset not available")
            return
        print("Available variables:")
        print(list(self.dataset.variables))

    def describe(self, var_name: str = None):
        if self.dataset is None:
            print("Dataset not available")
            return

        if var_name:
            # Describe single variable
            if var_name not in self.dataset.data_vars:
                print(f"Variable '{var_name}' not found")
                return
            var = self.dataset[var_name]
            print(f"\nVariable: {var_name}")
            print(f"Dimensions: {var.dims}")
            print(f"Shape: {var.shape}")
            print("Attributes:")
            for k, v in var.attrs.items():
                print(f"  - {k}: {v}")
        else:
            # Describe all variables in a tabular format
            summary = []
            for name, var in self.dataset.data_vars.items():
                dims = ", ".join(var.dims)
                shape = " × ".join(str(s) for s in var.shape)

                # Priority attributes to display (fallback to '-')
                attr_val = (
                    var.attrs.get("long_name") or
                    "-"
                )
                units = var.attrs.get("units", var.attrs.get("units", "-"))

                summary.append([name, dims, shape, attr_val, units])

            df = pd.DataFrame(summary, columns=["Variable", "Dimensions", "Shape", "Description", "Units"])
            display(df)

    def _get_model_dims(self):
        # These are UFS models, other systems may not have definitions and that's okay.
        models = {
            "atm": {"level_dim": "lev", "depth_dim": "depthBelowLandLayer"},
            "ocn": {"depth_dim": "depth"},
            "lnd": {"depth_dim": "depthBelowLandLayer"},
            "ice": {}, 
            "wav": {}
        }

        return models.get(getattr(self, 'model', 'dummy'), {})

    #######################################################################################################
    '''
    1. Regrid operates on data contained by a DataReader object.
    2. The underlying dataset is preserved.
    '''
    def _set_retrieve_params(self, **kwargs):

        self._retrieve_params = {
            'var':       kwargs.get('var',       None),
            'lat':       kwargs.get('lat',       None),
            'lon':       kwargs.get('lon',       None),
            'time':      kwargs.get('time',      None),
            'lev':       kwargs.get('lev',       None),

            'depth':     kwargs.get('depth',     None),
            'member':    kwargs.get('member',    None),
            'lead':      kwargs.get('lead',      None),
            'ens_avg':   kwargs.get('ens_avg',   None),

            'mean':      kwargs.get('mean',      None),
            'std':       kwargs.get('std',       None),
            'save_path': kwargs.get('save_path', None)
        }

    def _get_retrieve_params(self):
        return self._retrieve_params

    #######################################################################################################
    def retrieve(self, **kwargs) -> xr.Dataset:

        # Store old retrieve params
        old_params = self._get_retrieve_params()

        try:
            # Set new params, revert to old if retrieve() fails
            self._set_retrieve_params(**kwargs)
            return self._retrieve()

        except Exception as e:
            self._set_retrieve_params(**old_params)
            raise ValueError(f'Retrieve failed: {e}')

    def _retrieve(self) -> xr.Dataset:

        '''
        self,
        var: Union[str, List[str]],
        lat: Union[float, Tuple[float, float]] = None,
        lon: Union[float, Tuple[float, float]] = None,
        time: Union[datetime.datetime, str, Tuple] = None,
        lev: Union[float, Tuple] = None,
        depth: Union[float, Tuple] = None,
        member: Union[int, Tuple] = None,
        lead: Union[int, Tuple] = None,
        ens_avg: bool = False,

        mean: Union[str, List[str]] = None,   # NEW
        std: Union[str, List[str]] = None,    # NEW
        save_path: str = None
        ) -> xr.Dataset:
        '''

        # Store all sel parameters in a dictionary, then call sel at the end.
        #sel_dict = {}
 
        params = self._get_retrieve_params()

        if self.dataset is None:
            raise RuntimeError("Dataset failed to load during initialization")

        # var always a list
        var = params['var']
        if var is None:
            raise ValueError(f'Missing retrieve parameter: var (str, List[str])\nYou must specify a variable name or list of variable names to retrieve.')

        var_list = [var] if isinstance(var, str) else var
        missing = [v for v in var_list if v not in self.dataset]
        if missing:
            raise ValueError(f"Variables not found: {missing}")

        # First subset dataset by var
        data = self.dataset[var_list]

        # Latitude selection
        lat = params['lat']
        if lat is not None and 'lat' in data.dims:
            print('Slicing by lat')
            if isinstance(lat, (tuple, list)):
                lat_slice = sorted(lat, reverse=True)
                # if not DataReader.is_ascending(data.lat):  # <-- Can assume correct order if standardize_coords is run first.
                #     lat_slice = lat_slice[::-1]
                data = data.sel(lat=slice(*lat_slice))
            else:
                data = data.sel(lat=lat, method='nearest')

        # Longitude selection
        lon = params['lon']
        if lon is not None and 'lon' in data.dims:
            print('Slicing by lon')
            if isinstance(lon, (tuple, list)):
                lon = [l % 360 for l in lon]
                data = data.sel(lon=slice(*sorted(lon)))
            else:
                adj_lon = lon % 360 if isinstance(lon, (int, float)) else lon
                data = data.sel(lon=adj_lon, method='nearest')

        # Time selection
        time = params['time']
        if time is not None:

            # Datasets could coneivably have both 'init' and 'time' dimensions
            if 'init' in data.dims:
                print('Slicing by init')
                if isinstance(time, (tuple, list)):
                    start = DataReader.to_datetime(time[0])
                    end = DataReader.to_datetime(time[1])
                    data = data.sel(init=slice(start, end))
                else:
                    time_val = DataReader.to_datetime(time)
                    data = data.sel(init=time_val, method='nearest')

            if 'time' in data.dims:
                print('Slicing by time')
                if isinstance(time, (tuple, list)):
                    start = DataReader.to_datetime(time[0])
                    end = DataReader.to_datetime(time[1])
                    data = data.sel(time=slice(start, end))
                    #sel_dict['time'] = slice(start, end)
                else:
                    time_val = DataReader.to_datetime(time)
                    data = data.sel(time=time_val, method='nearest')

        # Model-specific vertical selection
        model_dims = self._get_model_dims()

        lev = params['lev']
        # For UFS levels
        if "level_dim" in model_dims and lev is not None and model_dims["level_dim"] in data.dims:
            print(f"Slicing by model dimension {model_dims['level_dim']}")
            if isinstance(lev, (tuple, list)):
                data = data.sel({model_dims["level_dim"]: slice(*lev)})
            else:
                data = data.sel({model_dims["level_dim"]: lev}, method='nearest') # <-- Get rid of "nearest"? We must be exact?

        # For other Vertical levels
        elif lev is not None:
            vertical_dim = None

            if 'hybrid' in data.dims:  # might need to revist this
                vertical_dim = 'hybrid'
            elif 'lev' in data.dims:
                vertical_dim = 'lev'
            else:
                raise ValueError(
                    f"""You specified lev={lev}, but there is no vertical dimension (hybrid or lev) found in the dataset.
                    For variable(s) [{', '.join(list(data.keys()))}] the available dims are [{', '.join(list(data.dims))}]
                    """
                )
            print(f'Slicing by {vertical_dim}')

            if isinstance(lev, (tuple, list)):
                data = data.sel(**{vertical_dim: slice(*lev)})
            else:
                data = data.sel(**{vertical_dim: lev}, method='nearest')
                #sel_dict['lev'] = lev

        #data = data.sel(**sel_dict)  # <-- Revisit this.  We may have better performance with 1 sel operation.

        # Depth
        depth = params['depth']
        if "depth_dim" in model_dims and depth is not None and model_dims["depth_dim"] in data.dims:
            print('Slicing by depth_dim')
            if isinstance(depth, (tuple, list)):
                data = data.sel({model_dims["depth_dim"]: slice(*depth)})
            else:
                data = data.sel({model_dims["depth_dim"]: depth}, method='nearest')

        # Ensemble member and lead time
        member = params['member']
        if member is not None and 'member' in data.dims:
            print('Getting member')
            if isinstance(member, (tuple, list)):
                data = data.sel(member=slice(*member))
            else:
                data = data.sel(member=member, method='nearest')

        lead = params['lead']
        if lead is not None and 'lead' in data.dims:
            print('Slicing by lead')
            if isinstance(lead, (tuple, list)):
                data = data.sel(lead=slice(*lead))
            else:
                data = data.sel(lead=lead, method='nearest')

        # Ensemble average
        ens_avg = params['ens_avg']
        if ens_avg and 'member' in data.dims:
            print('Taking Ensemble Average')
            data = data.mean(dim='member', keep_attrs=True)

        # Mean over dimensions
        mean = params['mean']
        if mean:
            mean_dims = [mean] if isinstance(mean, str) else mean
            for dim in mean_dims:
                if dim in data.dims:
                    print('Calculating MEAN')
                    data = data.mean(dim=dim, keep_attrs=True)
                else:
                    raise ValueError(f"Mean: Dimension '{dim}' not found in dataset")

        # Std over dimensions
        std = params['std']
        if std:
            print('Calculating STD')
            std_dims = [std] if isinstance(std, str) else std
            for dim in std_dims:
                if dim in data.dims:
                    data = data.std(dim=dim, keep_attrs=True)
                else:
                    raise ValueError(f"Std: Dimension '{dim}' not found in dataset")

        # Save
        save_path = params['save_path']
        if save_path:
            if save_path.endswith(".nc"):
                data.to_netcdf(save_path)
                print(f"Data saved as NetCDF: {save_path}")
            elif save_path.endswith(".csv"):
                data.to_dataframe().to_csv(save_path)
                print(f"Data saved as CSV: {save_path}")
            else:
                raise ValueError("Unsupported format. Use .nc or .csv")

        # Reset coordinates. E.g. if one lev was selected, remove the coordinate.
        # Downstream logic depends on this action.
        data = data.reset_coords(drop=True)

        # Always return dataSET, even for 1 variable
        return data































