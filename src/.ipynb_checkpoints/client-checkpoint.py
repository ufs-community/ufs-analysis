import xarray as xr
import datetime
from typing import Union, List, Tuple
import pandas as pd
from IPython.display import display

class Client:
    def __init__(self, dataset_path: str, model: str):
        allowed_models = ["atm", "ocn", "lnd", "ice", "wav"]
        if model not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        
        self.model = model
        self.base_url = "s3://noaa-oar-sfsdev-pds/"
        self.dataset_url = self.base_url + dataset_path
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        try:
            self.dataset = xr.open_zarr(self.dataset_url, 
                                        storage_options={"anon": True},
                                        consolidated=True)
            print(f"Dataset loaded for model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    def _to_datetime(self, time_str: Union[str, datetime.datetime]) -> datetime.datetime:
        if isinstance(time_str, datetime.datetime):
            return time_str
        try:
            return datetime.datetime.fromisoformat(time_str)
        except ValueError:
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d")
            except:
                raise ValueError("Invalid time format. Use ISO format or YYYY-MM-DD")

    def _is_ascending(self, coord: xr.DataArray) -> bool:
        return coord[0] < coord[-1]

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
        return {
            "atm": {"level_dim": "lev", "depth_dim": "depthBelowLandLayer"},
            "ocn": {"depth_dim": "depth"},
            "lnd": {"depth_dim": "depthBelowLandLayer"},
            "ice": {},
            "wav": {}
        }.get(self.model, {})
    
    def retrieve(
        self,
        var: Union[str, List[str]],
        lat: Union[float, Tuple[float, float]] = None,
        lon: Union[float, Tuple[float, float]] = None,
        time: Union[datetime.datetime, str, Tuple] = None,
        level: Union[float, Tuple] = None,
        depth: Union[float, Tuple] = None,
        member: Union[int, Tuple] = None,
        lead: Union[int, Tuple] = None,
        ens_avg: bool = False,
        mean: Union[str, List[str]] = None,   # NEW
        std: Union[str, List[str]] = None,    # NEW
        save_path: str = None
    ) -> Union[xr.DataArray, xr.Dataset]:

        if self.dataset is None:
            raise RuntimeError("Dataset failed to load during initialization")

        var_list = [var] if isinstance(var, str) else var
        missing = [v for v in var_list if v not in self.dataset]
        if missing:
            raise ValueError(f"Variables not found: {missing}")
        data = self.dataset[var_list]

        # Latitude selection
        if lat is not None and 'lat' in data.dims:
            if isinstance(lat, (tuple, list)):
                lat_slice = sorted(lat)
                if not self._is_ascending(data.lat):
                    lat_slice = lat_slice[::-1]
                data = data.sel(lat=slice(*lat_slice))
            else:
                data = data.sel(lat=lat, method='nearest')

        # Longitude selection
        if lon is not None and 'lon' in data.dims:
            if isinstance(lon, (tuple, list)):
                lon = [l % 360 for l in lon]
                data = data.sel(lon=slice(*sorted(lon)))
            else:
                adj_lon = lon % 360 if isinstance(lon, (int, float)) else lon
                data = data.sel(lon=adj_lon, method='nearest')

        # Time selection
        if time is not None and 'init' in data.dims:
            if isinstance(time, (tuple, list)):
                start = self._to_datetime(time[0])
                end = self._to_datetime(time[1])
                data = data.sel(init=slice(start, end))
            else:
                time_val = self._to_datetime(time)
                data = data.sel(init=time_val, method='nearest')

        # Model-specific vertical selection
        model_dims = self._get_model_dims()

        if "level_dim" in model_dims and level is not None and model_dims["level_dim"] in data.dims:
            if isinstance(level, (tuple, list)):
                data = data.sel({model_dims["level_dim"]: slice(*level)})
            else:
                data = data.sel({model_dims["level_dim"]: level}, method='nearest')

        if "depth_dim" in model_dims and depth is not None and model_dims["depth_dim"] in data.dims:
            if isinstance(depth, (tuple, list)):
                data = data.sel({model_dims["depth_dim"]: slice(*depth)})
            else:
                data = data.sel({model_dims["depth_dim"]: depth}, method='nearest')

        # Ensemble member and lead time
        if member is not None and 'member' in data.dims:
            if isinstance(member, (tuple, list)):
                data = data.sel(member=slice(*member))
            else:
                data = data.sel(member=member, method='nearest')

        if lead is not None and 'lead' in data.dims:
            if isinstance(lead, (tuple, list)):
                data = data.sel(lead=slice(*lead))
            else:
                data = data.sel(lead=lead, method='nearest')

        # Ensemble average
        if ens_avg and 'member' in data.dims:
            data = data.mean(dim='member', keep_attrs=True)

        # Mean over dimensions
        if mean:
            mean_dims = [mean] if isinstance(mean, str) else mean
            for dim in mean_dims:
                if dim in data.dims:
                    data = data.mean(dim=dim, keep_attrs=True)
                else:
                    raise ValueError(f"Mean: Dimension '{dim}' not found in dataset")

        # Std over dimensions
        if std:
            std_dims = [std] if isinstance(std, str) else std
            for dim in std_dims:
                if dim in data.dims:
                    data = data.std(dim=dim, keep_attrs=True)
                else:
                    raise ValueError(f"Std: Dimension '{dim}' not found in dataset")

        # Save
        if save_path:
            if save_path.endswith(".nc"):
                data.to_netcdf(save_path)
                print(f"Data saved as NetCDF: {save_path}")
            elif save_path.endswith(".csv"):
                data.to_dataframe().to_csv(save_path)
                print(f"Data saved as CSV: {save_path}")
            else:
                raise ValueError("Unsupported format. Use .nc or .csv")

        return data[var] if isinstance(var, str) else data

