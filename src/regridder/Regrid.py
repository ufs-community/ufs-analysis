import os
import time
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Union, Tuple, Optional, List

import xesmf as xe
import spharm
import _spherepack
from ..datareader import DataReader_Super
from ..datareader import datareader as dr
from ..util import resample, timeutil

class Regrid:
    VALID_METHODS = ['bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s']

    '''
    Regrid two DataReader objects.  Assume equally spaced rectilinear grids.
    '''

    def __init__(self,
                 data_reader1,
                 data_reader2,
                 method: str = 'bilinear'):

        '''
        Attributes:
            method
            data_reader_verif
            data_reader_model
            grid_verif
            grid_model
            regridder
        '''

        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from: {self.VALID_METHODS}")

        self.method = method

        # Check data_reader input
        if not isinstance(data_reader1, DataReader_Super.DataReader):
            raise ValueError(f'data_reader1 must be type DataReader, you submitted {type(data_reader1)}')
        if not isinstance(data_reader2, DataReader_Super.DataReader):
            raise ValueError(f'data_reader2 must be type DataReader, you submitted {type(data_reader2)}')

        # Pick with dataset should be considered the 'verification' set
        self.data_reader_verif,\
        self.data_reader_model,\
        self.grid_verif,\
        self.grid_model =\
            self._pick_verification(data_reader1, data_reader2)

        # Regrid.validate_coords(self.model_ds, "Model")
        # Regrid.validate_coords(self.verification_ds, "Verification")

        # Instantiate XESMF Regridder
        print(f"Initializing Regridder with method '{self.method}'")

        self.regridder = xe.Regridder(
            self.grid_verif,
            self.grid_model,
            self.method,
            locstream_in=False, # <-- can get rid of this altogether.
            reuse_weights=False
        )

        print("Regridder initialized.")

    def _pick_verification(self, *data_readers):

        if len(data_readers) != 2:
            raise ValueError(f'The private method ._pick_verification() must be supplied exactly 2 data_reader objects, got {len(data_readers)}')

        # Get the grids for each dataset
        grids = [self._create_grid(this_data_reader) for this_data_reader in data_readers]

        # Calculate number of vertices
        n_vertices = [len(np.unique(this_grid.lat)) * len(np.unique(this_grid.lon)) for this_grid in grids]

        if len(set(n_vertices)) == 1:
            raise ValueError(f'Both Grids have the same resolution, so regridding is futile.')

        verif_index = n_vertices.index(max(n_vertices))
        model_index = n_vertices.index(min(n_vertices))

        print(f'Verification grid set to data_reader{verif_index+1}')

        # Return tuple: verif grid (higher resolution), model grid (lower resolution)
        return\
            data_readers[verif_index],\
            data_readers[model_index],\
            grids[verif_index],\
            grids[model_index]


    def _create_grid(self, data_reader) -> xr.Dataset:

        # Construct model grid
        grid = xr.Dataset({'lat': data_reader.dataset['lat'], 'lon': data_reader.dataset['lon']})
        return grid

    def _get_lead_resolution(self, data_reader) -> Tuple[str, np.timedelta64]:

        if 'lead' not in data_reader.dataset.variables:
            return (None, None)

        lead = data_reader.dataset['lead'].values.astype(int)
        lead_unit = data_reader.dataset['lead'].attrs.get("units", "hours")

        # print(f"\nDEBUG: Analyzing lead time units")
        # print(f"       Raw lead values: {self.model_ds['lead'].values[:5]}...")
        # print(f"       Metadata units: '{lead_unit}'")

        if lead_unit == 'months':
            print("Interpreting lead as monthly intervals")
            return 'MS', np.timedelta64(30, 'D')

        elif lead_unit == 'days':
            print("Interpreting lead as daily intervals")
            return 'D', np.timedelta64(lead[1] - lead[0], 'D')

        elif lead_unit == 'hours':
            print("Interpreting lead as hourly intervals")
            return 'H', np.timedelta64(lead[1] - lead[0], 'h')

        else:
            raise ValueError(f"Unsupported lead unit '{lead_unit}'")

    def _resample_verification(self, dataset, freq: str):

        print(f"Resampling verification data to '{freq}' using mean aggregation.")

        # Should we show warning if expected computation time is very very long?

        # If multiple cores are available we can speed of computation significantly.
        # If only one core is available, then this code will still work.
        # n_cores = len(os.sched_getaffinity(0))
        n_cores = 1
        print(f'Number of cores available: {n_cores}')

        # Get all times in dataset
        all_times = list(dataset.time.values)

        # batches are [(start_time, end_time),...]
        batches = resample.datetime_batcher(all_times)
        resample_args = [(dataset, slice(batch[0], batch[1]), freq) for batch in batches]

        result = []
        for resample_arg in resample_args:
                print(f'Processing {resample_arg[1]}')
                result.append(resample.resample(resample_arg[0], resample_arg[1], resample_arg[2]))

        # -- MULTIPROCESS --
        #p = mp.Pool(n_cores)
        #result = p.starmap(resample.resample, resample_args, chunksize=1)
        #p.close()
        #p.join()
        # ------------------

        print('Concatenating resample results')
        result = xr.concat(result, dim='time').sortby('time')

        #return dataset.resample(time=freq).mean().sortby('time')
        return result

    def _align_time_verif_to_model_grid(self,
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
                forecast_time = timeutil.time_offset(freq_unit, init, lead, step)

                # Find closest match
                deltas = np.abs(verif_times - forecast_time)
                idx = deltas.argmin()
                delta = deltas[idx]

                # If this timestep exists in both datasets, insert verif data
                if delta <= tolerance:
                    lead_slices.append(verif.isel(time=idx))
                    matched += 1
                # Else this timestep is in model but not verif, then insert dummy NA data
                else:
                    fill = verif.isel(time=0).copy(deep=True)
                    for var in list(fill.keys()):
                        fill[var][:] = np.nan

                    lead_slices.append(fill)
                    filled += 1

            lead_stack = xr.concat(lead_slices, dim='lead')
            aligned.append(lead_stack)

        final = xr.concat(aligned, dim='init')
        final = final.assign_coords(init=('init', model_init), lead=('lead', model_lead))

        final.attrs['model_freq_unit'] = freq_unit
        final.attrs['lead_step'] = str(step)
        final.attrs['alignment_tolerance'] = str(tolerance)

        print(f"Time dimensions aligned:  matched {matched} timesteps")
        if filled > 0:
            msg = f'Verif data not available for {filled} timesteps, filled with NaN.'
            warnings.warn(msg)

        return final

    # ALIGN GRIDS
    def align(self, var, auto_resample: bool = True, **kwargs):

        '''var can be a string or a list of two wind vector variables'''

        # if len(kwargs) == 0:
        #     msg = f"""In order to align data, you must first specify a subset to retrieve.
        #     Here is the list of available retrieve parameters:\n
        #     {', '.join(list(self.data_reader_verif.get_retrieve_params().keys()))}
        #     """
        #     raise ValueError(msg)

        print("Starting alignment process")

        # This is for checking whether vector data needs to be regridded
        U_WIND_VAR = self.data_reader_verif.U_WIND
        V_WIND_VAR = self.data_reader_verif.V_WIND

        run_sphere = False  # <-- Can be toggled TRUE with following logic:

        # -- BEGIN input parameter checks ---------------------------------------------------------------------------
        # Check if input var are wind vectors
        if isinstance(var, list):
            if len(var) == 0:
                raise ValueError(f'Must supply var as a string or list of two wind vectors. Got empty list [].')

            elif len(var) > 2:
                msg = f'If var is a list, then it can only contain a maximum of two variables, both of which must be wind vectors. Got {var}'
                raise ValueError(msg)

            elif len(var) == 2:
                # Then this should be wind vector data.
                uv_in_verif = set([U_WIND_VAR, V_WIND_VAR]) & set(var)

                if len(uv_in_verif) == 1:
                    msg = f'''You supplied one wind vector ({list(uv_in_verif)[0]}) but not the other, so spherical harmonics cannot be run.'''
                    raise ValueError(msg)

                elif len(uv_in_verif) == 0:
                    msg = f'''You supplied a list of multiple variables but they are not wind vectors.
                    If you want to supply multiple (two) variables, they must be wind vectors.'''
                    raise ValueError(msg)

                # Check that domain is global
                if kwargs.get('lat', None) is not None or kwargs.get('lon', None) is not None:
                    msg = f'''Spherical harmonics can only be run on the full global domain.
                    If you want to regrid wind vector data, do not slice by lat-lon.'''
                    raise ValueError(msg)

                run_sphere = True # Green light to run spherical harmonics.  var can be a list.

            elif len(var) == 1:
                # Check that the variable isn't a wind vector
                var = var[0]

        # In case user enters a non-string
        if not isinstance(var, str) and run_sphere is False:
            msg = f'''var must be a string representing a single variable in the dataset.'''
            raise ValueError(msg)

        # Check if input var are wind vectors
        if isinstance(var, str):
            if var in [U_WIND_VAR, V_WIND_VAR]:
                msg = f'''You supplied one wind vector {var} but not the other, so spherical harmonics cannot be run.'''
                raise ValueError(msg)
        # -- END input parameter checks -----------------------------------------------------------------------------

        kwargs['var'] = var

        # Do a retrieve on the model dataset too, but just the times
        varlist = list(self.data_reader_model.dataset.keys())
        model_ds = self.data_reader_model.retrieve(var=varlist, time=kwargs.get('time', None))

        # model_ds has leads that may extend beyond the specified time slice.
        # In that case, we must extend the time range to capture all the needed verif data.
        freq_unit, step = self._get_lead_resolution(self.data_reader_model)
        if freq_unit not in ['MS', 'D', 'H']:
            raise ValueError(f"Unsupported frequency unit '{freq_unit}'")

        first_init = sorted(model_ds.init.values)[0]
        final_init = sorted(model_ds.init.values)[-1]
        # Leads are zero-indexed, so add 1
        final_lead = max(model_ds.sel(init=final_init).lead.values) + 1

        # Don't get verif data at or beyond this datetime
        final_time = timeutil.time_offset(freq_unit, final_init, final_lead, step)

        # Update time input
        kwargs['time'] = (str(first_init), str(final_time))

        # to_align is an xarray Dataset
        to_align = self.data_reader_verif.retrieve(**kwargs)

        # For verif, we want final_time to be exclusive, not inclusive
        all_times = list(to_align.time.values)
        matches = list(all_times >= final_time)
        final_index = matches.index(True) if True in matches else None

        # Finally, time slice the verif data to match the model init + lead times
        to_align = to_align.isel(time=slice(0, final_index))

        # This logic depends on retrieve() resetting the coordinates after slicing by a single level
        if 'lev' in list(to_align.coords) or 'level_dim' in list(to_align.coords) or 'hybrid' in list(to_align.coords):
            raise KeyError(f'You must specify a single vertical level to regrid.')

        # Resample verification dataset to match UFS temporal resolution
        if auto_resample is True:
            print('auto_resample set to True')
            start_time = time.time()
            to_align = self._resample_verification(to_align, freq_unit)
            end_time = time.time()
            print(f'Resampling done in {(end_time-start_time)/60.0} minutes.')

        ## REGRID SPHERE ##
        if run_sphere is True:

            # Get shape and dims.  We use to create dummy data for xesmf
            data_shape = to_align[list(to_align.keys())[0]].shape
            data_dims = list(to_align.dims)
            dummy_data = np.full(data_shape, np.nan)

            # Get u and v input vectors
            u_input = to_align[U_WIND_VAR].values
            v_input = to_align[V_WIND_VAR].values

            # -----------------
            # REGRID VECTOR U-V
            print(f"Running spherical harmonics on {U_WIND_VAR} and {V_WIND_VAR}")
            start_time = time.time()
            u_output, v_output = self._regrid_vector_spharm(u_input, v_input)
            end_time = time.time()
            print(f"Completed sperical harmonics in {(end_time-start_time)/60.0} minutes.")
            # -----------------

            # Drop u-v fields from the dataset. The spherical harmonic results will be added back.
            to_align = to_align.drop_vars([U_WIND_VAR, V_WIND_VAR])
            to_align = to_align.assign(dummy_variable=(data_dims, dummy_data))

            # Run xesmf regridder on dummy data, then drop the dummy data
            to_align = self.regridder(to_align)
            to_align = to_align.drop_vars('dummy_variable')

            # Assign new u-v data to the new grid
            to_align = to_align.assign(verif_u=(('time', 'lat', 'lon'), u_output))
            to_align = to_align.assign(verif_v=(('time', 'lat', 'lon'), v_output))

            # Rename back to origin names
            to_align = to_align.rename({'verif_u': U_WIND_VAR, 'verif_v': V_WIND_VAR})

        ## REGRID SCALAR ##
        else:
            print(f"Running scalar regrid on {', '.join(list(to_align.keys()))}")
            start_time = time.time()
            to_align = self.regridder(to_align)
            end_time = time.time()
            print(f"Completed scalar regrid in {(end_time-start_time)/60.0} minutes.")

        # Regridding in space done.  Now align temporally.
        # Ensure model_init and model_lead are 1D arrays
        model_init = np.atleast_1d(model_ds['init'].values)
        model_lead = np.atleast_1d(model_ds['lead'].values.astype(int))

        if 'init' not in to_align.dims:
            to_align = to_align.expand_dims('init')

        # ALIGN TIME COORDINATE SYSTEMS
        to_align = self._align_time_verif_to_model_grid(
            to_align, model_init, model_lead, freq_unit, step
        )

        print("Alignment complete.")

        # Return a data_reader object
        return dr.getDataReader(datasource='supplied', dataset=to_align)

    def _regrid_vector_spharm(self, u_input, v_input):

        '''
        This code is taken and minimally adapted from PyMTDG MJO-Teleconnection-Diagnostics package.
        https://github.com/MJO-Teleconnection-Diagnostics/pyMTDG

        Input dimensions have form time:lat:lon, but this code expects lat:lon:time.
        So, first move time axis to back, and then move it back to front at the end.
        '''

        # Move time axis to back
        u_input = np.moveaxis(u_input, 0, -1)
        v_input = np.moveaxis(v_input, 0, -1)

        # Get number of Lat and Lon inputs and outputs
        nlats_in  = len(list(np.unique(self.grid_verif.lat.values)))
        nlons_in  = len(list(np.unique(self.grid_verif.lon.values))) 

        nlats_out = len(list(np.unique(self.grid_model.lat.values)))
        nlons_out = len(list(np.unique(self.grid_model.lon.values)))

        # Make pyspharm grids
        spharm_in = spharm.Spharmt(nlons_in, nlats_in, legfunc='stored', gridtype='regular')
        spharm_out = spharm.Spharmt(nlons_out, nlats_out, legfunc='stored', gridtype='regular')

        # Simplify variable names
        nlat = spharm_in.nlat
        nlon = spharm_in.nlon
        nlat_out = spharm_out.nlat
        nlon_out = spharm_out.nlon

        if nlat%2: # nlat is odd
            n2 = (nlat + 1)/2
        else:
            n2 = nlat/2
 
        nt = u_input.shape[2] # 1 means 1 timestep
        ntrunc = spharm_out.nlat - 1

        w = u_input
        v = - v_input

        lwork = (2*nt+1)*nlat*nlon
        lwork_out = (2*nt+1)*nlat_out*nlon_out

        br, bi, cr, ci, ierror = _spherepack.vhaes(v, w, spharm_in.wvhaes, lwork)
    
        bc = _spherepack.twodtooned(br, bi, ntrunc)
        cc = _spherepack.twodtooned(cr, ci, ntrunc)

        br_out, bi_out = _spherepack.onedtotwod(bc, nlat_out)
        cr_out, ci_out = _spherepack.onedtotwod(cc, nlat_out)

        v_out, w_out, ierror = _spherepack.vhses(nlon_out, br_out, bi_out, cr_out, ci_out, spharm_out.wvhses, lwork_out)

        # Move time axis to front
        w_out = np.moveaxis(w_out, -1, 0)
        v_out = np.moveaxis(v_out, -1, 0)

        return w_out, -v_out
        






#    def check_alignment(self, model: xr.DataArray, verif: xr.DataArray):
#        print(f"DEBUG: Model shape: {model.shape}")
#        print(f"       Model init range: {model.init.values.min()} → {model.init.values.max()}")
#        print(f"       Model lead range: {model.lead.values.min()} → {model.lead.values.max()}")

#        print(f"DEBUG: Verification shape: {verif.shape}")
#        print(f"       Verification init range: {verif.init.values.min()} → {verif.init.values.max()}")
#        print(f"       Verification lead range: {verif.lead.values.min()} → {verif.lead.values.max()}")
#        print(f"       Verification lead unit: {verif.attrs.get('model_freq_unit', 'unknown')}")
#        print(f"       Lead step size: {verif.attrs.get('lead_step', 'unknown')}")









