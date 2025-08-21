import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from typing import Union, Tuple, Optional, List

class Regrid:
    VALID_METHODS = ['bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s']

    def __init__(self,
                 model_ds: Union[xr.Dataset, xr.DataArray],
                 verification_ds: Union[xr.Dataset, xr.DataArray],
                 method: str = 'bilinear'):
        self.method = method
        # print(f"DEBUG: Initializing Regridder with method '{method}'")
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from: {self.VALID_METHODS}")
        # print("\nDEBUG: Standardizing model dataset coordinates...")
        self.model_ds = self._standardize_coords(model_ds)
        # print("DEBUG: Model dataset coordinates after standardization:")
        # print(f"       Coordinates: {list(self.model_ds.coords.keys())}")
        # print("\nDEBUG: Standardizing verification dataset coordinates...")
        self.verification_ds = self._standardize_coords(verification_ds)
#         print("DEBUG: Verification dataset coordinates after standardization:")
#         print(f"       Coordinates: {list(self.verification_ds.coords.keys())}")

#         print("\nDEBUG: Validating coordinate requirements...")

        self._validate_coords(self.model_ds, "Model")
        self._validate_coords(self.verification_ds, "Verification")
        
        # print("\nDEBUG: Creating grid definitions...")
        self.model_grid, self.verification_grid, locstream_in = self._create_grids()
        # print("DEBUG: Model grid structure:")
        # print(f"       lat: {self.model_grid.lat.shape}, lon: {self.model_grid.lon.shape}")
        # print("DEBUG: Verification grid structure:")
        # print(f"       lat: {self.verification_grid.lat.shape}, lon: {self.verification_grid.lon.shape}")
        # print(f"       locstream_in flag: {locstream_in}")

        # Build regridder with parameter logging
        # print("\nDEBUG: Building xESMF regridder with parameters:")
        # print(f"       source: {self.verification_grid.sizes}")
        # print(f"       target: {self.model_grid.sizes}")
        # print(f"       method: {method}")
        # print(f"       locstream_in: {locstream_in}")

        self.regridder = xe.Regridder(
            self.verification_grid,
            self.model_grid,
            self.method,
            locstream_in=locstream_in,
            reuse_weights=False
        )
        print("Regridder initialized successfully")

    def _standardize_coords(self, ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        coord_map = {'latitude': 'lat', 'y': 'lat', 'longitude': 'lon', 'x': 'lon'}
        rename_dict = {}
        for old, new in coord_map.items():
            if old in ds.coords:
                rename_dict[old] = new
        if isinstance(ds, xr.Dataset):
            for old, new in coord_map.items():
                if old in ds.data_vars:
                    rename_dict[old] = new
        # print("DEBUG: rename_dict: ", rename_dict)
        return ds.rename(rename_dict) if rename_dict else ds

    def _validate_coords(self, ds: Union[xr.Dataset, xr.DataArray], name: str):
        # print(f"DEBUG: Validating {name} dataset coordinates...")
        for coord in ['lat', 'lon']:
            if coord not in ds.coords:
                raise ValueError(f"{name} dataset must contain '{coord}' coordinate.")
        if name == 'Verification' and 'time' not in ds.coords:
            raise ValueError("Verification dataset must have a 'time' coordinate.")
        if name == 'Model' and not {'init', 'lead'}.issubset(ds.coords):
            raise ValueError("Model dataset must contain both 'init' and 'lead' coordinates.")
        # print(f"DEBUG: {name} dataset coordinate validation passed")

    def _create_grids(self) -> Tuple[xr.Dataset, xr.Dataset, bool]:
        model_grid = xr.Dataset({'lat': self.model_ds['lat'], 'lon': self.model_ds['lon']})
        verif_lat = self.verification_ds['lat']
        verif_lon = self.verification_ds['lon']
        if verif_lat.ndim == 1 and verif_lon.ndim == 1 and 'lat' in self.verification_ds.dims and 'lon' in self.verification_ds.dims:
            verification_grid = xr.Dataset({'lat': verif_lat, 'lon': verif_lon})
            # print("DEBUG: Verification grid is 1D structured")
            return model_grid, verification_grid, False
        else:
            verification_grid = xr.Dataset({
                'lat': (['locations'], np.ravel(verif_lat.values)),
                'lon': (['locations'], np.ravel(verif_lon.values))
            })
            # print("DEBUG: Verification grid is unstructured (locstream)")
            return model_grid, verification_grid, True

    def _get_lead_resolution(self) -> Tuple[str, np.timedelta64]:
        lead = self.model_ds['lead'].values.astype(int)
        lead_unit = self.model_ds['lead'].attrs.get("units", "hours")
        
        # print(f"\nDEBUG: Analyzing lead time units")
        # print(f"       Raw lead values: {self.model_ds['lead'].values[:5]}...")
        # print(f"       Metadata units: '{lead_unit}'")

        if lead_unit == 'months':
            # print("DEBUG: Interpreting lead as monthly intervals")
            return 'MS', np.timedelta64(30, 'D')
        elif lead_unit == 'days':
            # print("DEBUG: Interpreting lead as daily intervals")
            return 'D', np.timedelta64(lead[1] - lead[0], 'D')
        elif lead_unit == 'hours':
            # print("DEBUG: Interpreting lead as hourly intervals")
            return 'H', np.timedelta64(lead[1] - lead[0], 'h')
        else:
            raise ValueError(f"Unsupported lead unit '{lead_unit}'")

    def _resample_verification(self, verif: xr.DataArray, freq: str) -> xr.DataArray:
        # print(f"DEBUG: Resampling verification data to '{freq}' using mean aggregation.")
        # print(f"       Original time range: {verif['time'].min().item()} → {verif['time'].max().item()}")
        # print(f"       Original shape: {verif.shape}")
        return verif.resample(time=freq).mean().sortby('time')

    def _align_verif_to_model_grid(self,
                                    verif: xr.DataArray,
                                    model_init: np.ndarray,
                                    model_lead: np.ndarray,
                                    freq_unit: str,
                                    step: np.timedelta64,
                                    tolerance: Optional[np.timedelta64] = None) -> xr.DataArray:
    
        # print("DEBUG: Aligning verification time → model (init, lead) structure...")
        # print("\nDEBUG: Starting temporal alignment")
        # print(f"       Model init count: {len(model_init)}")
        # print(f"       Model lead count: {len(model_lead)}")
        # print(f"       Verification time count: {len(verif['time'])}")
        # print(freq_unit)

        if tolerance is None:
            if freq_unit == "MS":
                tolerance = np.timedelta64(16, 'D')
            elif freq_unit == "D":
                tolerance = np.timedelta64(12, 'h')
            elif freq_unit == "H":
                tolerance = np.timedelta64(1, 'h')
            else:
                tolerance = np.timedelta64(1, 'D')
            # print(f"DEBUG: Auto-selected tolerance = {tolerance}")

        verif = verif.sortby('time')
        verif_times = verif['time'].values.astype('datetime64[ns]')
        aligned = []
        matched, filled = 0, 0


        for init_idx, init in enumerate(model_init):
            lead_slices = []
            # print(f"\nDEBUG: Processing init[{init_idx}] = {init}")
            for lead_idx, lead in enumerate(model_lead):

                # Forecast time calculation
                try:
                    if freq_unit == 'MS':
                        forecast_time = np.datetime64(pd.Timestamp(init) + pd.DateOffset(months=int(lead)))
                    elif freq_unit == 'D':
                        forecast_time = np.datetime64(init) + np.timedelta64(int(lead * step.astype('timedelta64[D]').astype(int)), 'D')
                    elif freq_unit == 'H':
                        forecast_time = np.datetime64(init) + np.timedelta64(int(lead * step.astype('timedelta64[h]').astype(int)), 'h')
                    else:
                        raise ValueError(f"Unsupported frequency unit '{freq_unit}'")
                except Exception as e:
                    print(f"ERROR: Failed to compute forecast_time for lead={lead} (reason: {e})")
                    continue

                # Find closest match
                deltas = np.abs(verif_times - forecast_time)
                idx = deltas.argmin()
                delta = deltas[idx]

                if delta <= tolerance:
                    lead_slices.append(verif.isel(time=idx))
                    matched += 1
                else:
                    fill = verif.isel(time=0).copy(deep=True)
                    fill[:] = np.nan
                    lead_slices.append(fill)
                    filled += 1

            lead_stack = xr.concat(lead_slices, dim='lead')
            aligned.append(lead_stack)

        final = xr.concat(aligned, dim='init')
        final = final.assign_coords(init=('init', model_init), lead=('lead', model_lead))

        final.attrs['model_freq_unit'] = freq_unit
        final.attrs['lead_step'] = str(step)
        final.attrs['alignment_tolerance'] = str(tolerance)

        # print(f"\nDEBUG: Alignment complete — matched={matched}, filled with NaNs={filled}")
        return final


    def align(self, model_var_name: str, verif_var_name: str, auto_resample: bool = True) -> Tuple[xr.DataArray, xr.DataArray]:
        # print("\nDEBUG: Starting alignment process")
        # print(f"       Model variable: {model_var_name}")
        # print(f"       Verification variable: {verif_var_name}")


        model_var = self.model_ds if isinstance(self.model_ds, xr.DataArray) else self.model_ds[model_var_name]
        if 'init' not in model_var.dims:
            model_var = model_var.expand_dims('init')

        verif_var = self.verification_ds if isinstance(self.verification_ds, xr.DataArray) else self.verification_ds[verif_var_name]
        # print("\nDEBUG: Regridding verification data...")
        # print(f"       Original verification shape: {verif_var.shape}")
        verif_var = self.regridder(verif_var)
        # print(f"       Regridded verification shape: {verif_var.shape}")

        freq_unit, step = self._get_lead_resolution()
        if auto_resample:
            verif_var = self._resample_verification(verif_var, freq_unit)
        # print("DEBUG: Resampling done")
        # Ensure model_init and model_lead are 1D arrays
        model_init = np.atleast_1d(self.model_ds['init'].values) 
        model_lead = np.atleast_1d(self.model_ds['lead'].values.astype(int)) 

        verif_aligned = self._align_verif_to_model_grid(
            verif_var, model_init, model_lead, freq_unit, step
        )

        # print("DEBUG: Alignment complete.")
        return model_var, verif_aligned

    def check_alignment(self, model: xr.DataArray, verif: xr.DataArray):
        # print(f"DEBUG: Model shape: {model.shape}")
        print(f"       Model init range: {model.init.values.min()} → {model.init.values.max()}")
        print(f"       Model lead range: {model.lead.values.min()} → {model.lead.values.max()}")

        # print(f"DEBUG: Verification shape: {verif.shape}")
        print(f"       Verification init range: {verif.init.values.min()} → {verif.init.values.max()}")
        print(f"       Verification lead range: {verif.lead.values.min()} → {verif.lead.values.max()}")
        print(f"       Verification lead unit: {verif.attrs.get('model_freq_unit', 'unknown')}")
        print(f"       Lead step size: {verif.attrs.get('lead_step', 'unknown')}")
