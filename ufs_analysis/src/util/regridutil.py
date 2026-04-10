# ---------------------------------------------------------------------------------------------------------------------
#  Filename: regridutil.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 03 Feb. 2026
#  Purpose: Define the utility for scalar regridding via Scipy RegularGridInterpolator.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


class ScalarRegridUtility:

    def __init__(self,
                 highres_grid: xr.Dataset,
                 lowres_grid: xr.Dataset,
                 method: str = 'linear'):

        # Assign attributes to self
        self.highres_grid = highres_grid
        self.lowres_grid = lowres_grid
        self.method = method

        self.n_highres_lon = len(highres_grid.lon.values)
        self.n_highres_lat = len(highres_grid.lat.values)
        self.n_lowres_lon = len(lowres_grid.lon.values)
        self.n_lowres_lat = len(lowres_grid.lat.values)

        # Store lat/lon information
        self.old_lats = highres_grid.lat.values
        self.old_lons = highres_grid.lon.values

        self.new_lats = lowres_grid.lat.values
        self.new_lons = lowres_grid.lon.values

        # Mesh for new coordinates
        self.mesh = np.meshgrid(self.new_lats, self.new_lons, indexing='ij', sparse=False)
        xis = self.mesh[0]
        yis = self.mesh[1]

        # New coordinate pairs
        self.new_coord_pairs = np.stack((xis, yis), axis=-1).reshape(-1, 2)

    def regrid(self, ds: xr.Dataset, var: str) -> xr.Dataset:
        ''' time or init+lead structure may be present'''

        # If lev is a dimension in the data, then drop it temporarily, work on the underlying arrays, and add it back.
        stored_lev = None
        if 'lev' in ds.dims:
            stored_lev = ds.lev.values[0]  # upstream logic has already confirmed that these data are flat.
            ds = ds.squeeze(dim='lev')

        if 'time' in ds.dims:

            all_times = list(ds.time.values)
            slices = []
            for this_time in all_times:

                # This time slice of the data.
                this_ds = ds.sel(time=[this_time], drop=False)

                # Numpy array of data values to regrid.
                vals_to_regrid = this_ds[var].sel(time=this_time, drop=True).values

                # Regrid with Scipy GridInterpolator.
                # Set up Interpolator object.
                interp = RegularGridInterpolator(points=(self.old_lats, self.old_lons),
                                                 values=vals_to_regrid,
                                                 method=self.method,
                                                 bounds_error=False)  # account for poles

                # Do the interpolation.
                interp_results = interp(self.new_coord_pairs)
                # Reorganize results into 2-dimensional array
                new_results_grid = np.reshape(interp_results, (self.n_lowres_lat, self.n_lowres_lon))

                # Create xarray object.
                this_result = xr.Dataset(
                    data_vars={
                        'var_to_regrid': (("lat", "lon"), new_results_grid),
                    },
                    coords={
                        "lat": self.new_lats,
                        "lon": self.new_lons,
                    },
                    attrs={"description": "regrid"}
                )

                # Append results to a list.
                this_result = this_result.assign_coords(time=this_time)
                this_result = this_result.rename_vars({"var_to_regrid": f'{var}'})
                slices.append(this_result)

            # Concatenate list of results into a single xarray object.
            final_result = xr.concat(slices, dim='time', coords='minimal', compat='equals').sortby('time')

        elif 'init' in ds.dims:

            inits = np.atleast_1d(ds['init'].values)
            leads = np.atleast_1d(ds['lead'].values.astype(int))
            stack = []

            for this_init in inits:
                lead_slices = []
                for this_lead in leads:

                    # This time slice of the data.
                    this_ds = ds.sel(init=[this_init], lead=[this_lead], drop=False)

                    # Numpy array of data values to regrid.
                    vals_to_regrid = this_ds[var].sel(init=this_init, lead=this_lead, drop=True).values

                    # Regrid with Scipy GridInterpolator.
                    # Set up Interpolator object.
                    interp = RegularGridInterpolator(points=(self.old_lats, self.old_lons),
                                                     values=vals_to_regrid,
                                                     method=self.method,
                                                     bounds_error=False)  # account for poles

                    # Do the interpolation.
                    interp_results = interp(self.new_coord_pairs)
                    # Reorganize results into 2-dimensional array
                    new_results_grid = np.reshape(interp_results, (self.n_lowres_lat, self.n_lowres_lon))

                    this_result = xr.Dataset(
                        data_vars={
                            'var_to_regrid': (("lat", "lon"), new_results_grid),
                        },
                        coords={
                            "lat": self.new_lats,
                            "lon": self.new_lons,
                        },
                        attrs={"description": "regrid"}
                    )

                    this_result = this_result.assign_coords(init=this_init, lead=this_lead)
                    this_result = this_result.rename_vars({"var_to_regrid": f'{var}'})

                    lead_slices.append(this_result)

                lead_stack = xr.concat(lead_slices, dim='lead', coords='different', compat='equals')

                stack.append(lead_stack)

            final_result = xr.concat(stack, dim='init', coords='different', compat='equals')
            final_result = final_result.assign_coords(init=('init', inits), lead=('lead', leads))

        # Add back lev, if it existed.
        if stored_lev is not None:
            final_result = final_result.expand_dims(lev=[stored_lev])

        return final_result
