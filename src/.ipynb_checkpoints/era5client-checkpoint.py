import xarray as xr
import fsspec
import datetime
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
from IPython.display import display

class ERA5Client:
    def __init__(self, dataset_path: str):
        self.base_url = "gs://gcp-public-data-arco-era5/co/"
        self.dataset_url = self.base_url + dataset_path
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        try:
            self.dataset = xr.open_zarr(self.dataset_url, 
                                        storage_options={"token": "anon"},
                                        consolidated=True, decode_timedelta=False)
            print("ERA5 dataset loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ERA5 dataset: {e}")

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
                    var.attrs.get("GRIB_name") or
                    var.attrs.get("standard_name") or
                    var.attrs.get("GRIB_shortName") or
                    "-"
                )
                units = var.attrs.get("units", var.attrs.get("GRIB_units", "-"))

                summary.append([name, dims, shape, attr_val, units])

            df = pd.DataFrame(summary, columns=["Variable", "Dimensions", "Shape", "Description", "Units"])
            display(df)


    def retrieve(
        self,
        var: Union[str, List[str]],
        lat: Union[float, Tuple[float, float]] = None,
        lon: Union[float, Tuple[float, float]] = None,
        time: Union[datetime.datetime, str, Tuple] = None,
        level: Union[float, Tuple] = None,
        mean: Union[str, List[str]] = None,
        std: Union[str, List[str]] = None,
        save_path: str = None
    ) -> Union[xr.DataArray, xr.Dataset]:

        if self.dataset is None:
            raise RuntimeError("Dataset failed to load")

        var_list = [var] if isinstance(var, str) else var
        missing = [v for v in var_list if v not in self.dataset]
        if missing:
            raise ValueError(f"Variables not found: {missing}")
        data = self.dataset[var_list]

        # Handle latitude and longitude selection
        if lat is not None or lon is not None:
            # Check if latitude and longitude are variables along 'values' dimension
            if 'values' in data.dims and 'latitude' in data.variables and 'longitude' in data.variables:
                mask = xr.DataArray(np.ones_like(data['latitude'], dtype=bool), dims=['values'])

                # Latitude mask
                if lat is not None:
                    if isinstance(lat, (tuple, list)):
                        lat_min, lat_max = sorted(lat)
                        lat_mask = (data.latitude >= lat_min) & (data.latitude <= lat_max)
                    else:
                        lat_diff = abs(data.latitude - lat)
                        lat_idx = lat_diff.argmin()
                        lat_mask = data.latitude.isin(data.latitude[lat_idx])
                    mask = mask & lat_mask

                # Longitude mask
                if lon is not None:
                    if isinstance(lon, (tuple, list)):
                        lon_min, lon_max = sorted(lon)
                        lon_min = lon_min % 360
                        lon_max = lon_max % 360
                        if lon_min <= lon_max:
                            lon_mask = (data.longitude >= lon_min) & (data.longitude <= lon_max)
                        else:
                            lon_mask = (data.longitude >= lon_min) | (data.longitude <= lon_max)
                    else:
                        target_lon = lon % 360
                        lon_diff = abs((data.longitude % 360) - target_lon)
                        lon_idx = lon_diff.argmin()
                        lon_mask = data.longitude.isin(data.longitude[lon_idx])
                    mask = mask & lon_mask
                
                data = data.sel(values=mask)

            else:
                # Handle latitude and longitude as dimensions
                if lat is not None:
                    if 'latitude' in data.dims:
                        if isinstance(lat, (tuple, list)):
                            lat_slice = sorted(lat)
                            if data.latitude[0] > data.latitude[-1]:  # descending
                                lat_slice = lat_slice[::-1]
                            data = data.sel(latitude=slice(*lat_slice))
                        else:
                            data = data.sel(latitude=lat, method='nearest')
                    else:
                        raise ValueError("Cannot subset by latitude: 'latitude' not found as a dimension or variable.")

                if lon is not None:
                    if 'longitude' in data.dims:
                        if isinstance(lon, (tuple, list)):
                            lon = [l % 360 for l in lon]
                            data = data.sel(longitude=slice(*sorted(lon)))
                        else:
                            adj_lon = lon % 360 if isinstance(lon, (int, float)) else lon
                            data = data.sel(longitude=adj_lon, method='nearest')
                    else:
                        raise ValueError("Cannot subset by longitude: 'longitude' not found as a dimension or variable.")

        # Time selection
        if time is not None and 'time' in data.dims:
            if isinstance(time, (tuple, list)):
                start = self._to_datetime(time[0])
                end = self._to_datetime(time[1])
                data = data.sel(time=slice(start, end))
            else:
                time_val = self._to_datetime(time)
                data = data.sel(time=time_val, method='nearest')

        # Vertical level selection
        if level is not None:
            vertical_dim = None
            if 'hybrid' in data.dims:
                vertical_dim = 'hybrid'
            elif 'level' in data.dims:
                vertical_dim = 'level'
            else:
                raise ValueError("No vertical dimension (hybrid or level) found in the dataset.")

            if isinstance(level, (tuple, list)):
                data = data.sel(**{vertical_dim: slice(*level)})
            else:
                data = data.sel(**{vertical_dim: level}, method='nearest')

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