# ---------------------------------------------------------------------------------------------------------------------
#  Filename: Regrid.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Define the Regrid class, principal methods being .resample(), .regrid(), and .align().
# ---------------------------------------------------------------------------------------------------------------------

import os
import gc
import time
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Union, Tuple, Optional, List
# import xesmf as xe
import spharm
import _spherepack

from ..datareader import DataReader_Super, UFS_DataReader
from ..datareader import datareader as dr
from ..util import util, timeutil, regridutil, stats


class Regrid:
    # VALID_METHODS = ['bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s']  # xesmf
    VALID_METHODS = ['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip']  # scipy

    '''
    Regrid two DataReader objects.  Assume equally spaced rectilinear grids.
    '''

    def __init__(self,
                 data_reader1,
                 data_reader2,
                 method: str = 'linear'):

        '''
        Attributes:
        '''

        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from: {self.VALID_METHODS}")

        self.method = method  # method is used for scalar regrids only
        self.resampled = None  # set by self.resample().  This is LOADED DATA.
        self.regridded = None  # set by self.regrid()
        self.aligned = None  # set by self.align()
        self._regrid_vars = []
        self._resample_vars = []
        self.highres_grid = None  # integer 0, 1, or 2 will be assigned.
        self.to_resample = None  # integer 0, 1, or 2 will be assigned.

        # Check data_reader input
        if not isinstance(data_reader1, DataReader_Super.DataReader):
            raise ValueError(f'data_reader1 must be type DataReader, you submitted {type(data_reader1)}')
        if not isinstance(data_reader2, DataReader_Super.DataReader):
            raise ValueError(f'data_reader2 must be type DataReader, you submitted {type(data_reader2)}')

        # Assign data readers to self
        self._data_reader1 = data_reader1
        self._data_reader2 = data_reader2

        # Check if data_readers are UFS
        self._is_dr1_UFS = isinstance(self._data_reader1, UFS_DataReader.UFS_DataReader)
        self._is_dr2_UFS = isinstance(self._data_reader2, UFS_DataReader.UFS_DataReader)

        # This is a UFS-focused tool.  We need at least 1 UFS dataset to work with.
        if self._is_dr1_UFS is False and self._is_dr2_UFS is False:
            msg = f'Regrid requires at least 1 UFS dataset. '
            msg += f'Got {type(self._data_reader1)} and {type(self._data_reader2)}.'
            raise ValueError(msg)

        # Validate coordinate systems
        if self._is_dr1_UFS is True:
            self._validate_coords(self._data_reader1, "Model")
        else:
            self._validate_coords(self._data_reader1, "Verification")

        if self._is_dr2_UFS is True:
            self._validate_coords(self._data_reader2, "Model")
        else:
            self._validate_coords(self._data_reader2, "Verification")

        # Store the lat-lon grids
        self._grid1 = self._create_grid(self._data_reader1)
        self._grid2 = self._create_grid(self._data_reader2)

        # Pick with dataset should be spatially regridded onto the other.
        # Highest res is always regridded onto lower res.
        self._highres_grid = self._pick_highres_grid(self._grid1, self._grid2)  # Order matters
        # What if both resolutions are the same? (0)
        self._lowres_grid = 1 if self._highres_grid == 2 else 2

        #  Make this value public
        self.highres_grid = self._highres_grid

        # Get readable data types of the data_readers to print to console, e.g. UFS_DataReader to ERA5_DataReader.
        if self._highres_grid != 0:
            type_highres = str(type(getattr(self, f'_data_reader{self._highres_grid}')))
            type_lowres = str(type(getattr(self, f'_data_reader{self._lowres_grid}')))
        else:
            # Arbitrary
            type_highres = str(type(getattr(self, f'_data_reader1')))
            type_lowres = str(type(getattr(self, f'_data_reader2')))

        # Assign types
        self._type_highres = type_highres.split("'")[1].split('.')[-1]
        self._type_lowres = type_lowres.split("'")[1].split('.')[-1]

        # Resampling can be run if 1 dataset has init+lead structure and the other does not.
        if self._is_dr1_UFS is True and self._is_dr2_UFS is True:
            self._freq_unit, self._step = timeutil.get_lead_resolution(self._data_reader1.dataset())
            self._to_resample = 0  # no resampling to do if both datasets are UFS
            self.to_resample = 0  # Also set a public attribute for this.

        elif self._is_dr2_UFS is True:
            self._freq_unit, self._step = timeutil.get_lead_resolution(self._data_reader2.dataset())
            self._to_resample = 1
            self.to_resample = 1

        else:
            self._freq_unit, self._step = timeutil.get_lead_resolution(self._data_reader1.dataset())
            self._to_resample = 2
            self.to_resample = 2

        if self._freq_unit not in ['MS', 'D', 'H']:
            raise ValueError(f"Model has unsupported frequency unit '{self._freq_unit}'")

        # Get the grid shapes as a readable string
        if self._highres_grid != 0:
            input_grid_shape = str(getattr(self, f'_grid{self._highres_grid}').sizes)
            output_grid_shape = str(getattr(self, f'_grid{self._lowres_grid}').sizes)
        else:
            # Arbitrary
            input_grid_shape = str(getattr(self, f'_grid1').sizes)
            output_grid_shape = str(getattr(self, f'_grid2').sizes)

        # Remove some characters from stringified .sizes to make it more readable
        for ch in ["\'", "{", "(", ")", "}", "Frozen", ":"]:
            input_grid_shape = input_grid_shape.replace(ch, '')
            output_grid_shape = output_grid_shape.replace(ch, '')

        # Instantiate XESMF Regridder
        # Use this grid for highres/lowres:
        if self._highres_grid != 0:
            use_this_highres_grid = getattr(self, f'_grid{self._highres_grid}')
            use_this_lowres_grid = getattr(self, f'_grid{self._lowres_grid}')
        else:
            # Arbitrary
            use_this_highres_grid = getattr(self, f'_grid1')
            use_this_lowres_grid = getattr(self, f'_grid2')

#         self.regridder = xe.Regridder(
#             use_this_highres_grid,
#             use_this_lowres_grid,
#             self.method,
#             locstream_in=False,  # <-- can get rid of this altogether.
#             reuse_weights=False
#         )

        self.regridder = regridutil.ScalarRegridUtility(
            use_this_highres_grid,
            use_this_lowres_grid,
            method=self.method)

        print('\nRegrid Object initialized.')

        # Resample instructions
        print('\n___Resample Instructions___')
        if self._to_resample != 0:
            print(f'data_reader{self._to_resample} must be temporally resampled before spatially regridding.')
            print(f'Resample these data by running <RegridObj>.resample(var=<var>, lev=<lev>, time=<time_range>)')
            # Resample variables
            self._resample_vars = list(getattr(self, f'_data_reader{self._to_resample}').dataset().keys())
            print(f'To see all variables available for resample, run <RegridObj>.resample_vars()')
        else:
            print(f'Both datasets are UFS, so the .resample() method is disabled.')

        # Regrid instructions
        print('\n___Regrid Instructions___')
        if self._highres_grid != 0:
            print(f"Initialized the Regridder with method '{self.method}'")
            print(f"Input grid shape data_reader{self._highres_grid} ({self._type_highres}): {input_grid_shape}")
            print(f"Output grid shape data_reader{self._lowres_grid} ({self._type_lowres}): {output_grid_shape}")
            print(f"Regrid these data by running <RegridObj>.regrid(var=<var>, lev=<lev>, time=<time_range>)")
            # Regrid variables
            self._regrid_vars = list(getattr(self, f'_data_reader{self._highres_grid}').dataset().keys())
            print(f'To see all variables available for regrid, run <RegridObj>.regrid_vars()')
        else:
            print(f'Both datasets have the same spatial resolution, so the .regrid() method is disabled.')

        # Align instructions
        print('\n___Align Instructions___')
        if self._to_resample != 0:
            print(f'data_reader{self._to_resample} can have its time coordinates converted to init+lead.')
            freq_types = {'MS': 'monthly', 'D': 'daily', 'H': 'hourly'}
            print(f"Lead resolution of the UFS dataset interpreted as {freq_types[self._freq_unit]} intervals.")
            print(f'Align these data by running <RegridObj>.align() (You may need to resample and/or regrid first.)\n')
        else:
            print(f'Both datasets are UFS, so the .align() method is disabled.')

    def regrid_vars(self):
        '''Neatly print variables that can be regridded'''
        if self._highres_grid != 0:
            print(f'Variables available for Regrid (data_reader{self._highres_grid}):')
            print(util.print_fixed_width(self._regrid_vars))
        else:
            print(f'Both datasets have the same spatial resolution.')

    def resample_vars(self):
        '''Neatly print variables that can be resampled'''
        if self._to_resample != 0:
            print(f'Variables available for Resample (data_reader{self._to_resample}):')
            print(util.print_fixed_width(self._resample_vars))
        else:
            print(f'Both datasets have the same temporal resolution.')

    def _validate_coords(self, data_reader: DataReader_Super.DataReader, name: str):

        '''This is either redundant or should be defined in the DataReader class instead.'''

        ds = data_reader.dataset()
        for coord in ['lat', 'lon']:
            if coord not in ds.coords:
                raise ValueError(f"{name} dataset must contain '{coord}' coordinate.")

        if name == 'Verification' and 'time' not in ds.coords:
            raise ValueError("Verification dataset must have a 'time' coordinate.")

        if name == 'Model' and not {'init', 'lead'}.issubset(ds.coords):
            raise ValueError("Model dataset must contain both 'init' and 'lead' coordinates.")

    def _pick_highres_grid(self, grid1: xr.Dataset, grid2: xr.Dataset) -> int:

        # Calculate number of vertices
        grid1_n_vertices = len(np.unique(grid1.lat)) * len(np.unique(grid1.lon))
        grid2_n_vertices = len(np.unique(grid2.lat)) * len(np.unique(grid2.lon))

        if grid1_n_vertices == grid2_n_vertices:
            return 0  # Indicates futility of spatial regridding, though perhaps resampling can still be done.

        if grid1_n_vertices > grid2_n_vertices:
            return 1
        else:
            return 2

    def _create_grid(self, data_reader: DataReader_Super.DataReader) -> xr.Dataset:

        # Construct model grid
        grid = xr.Dataset({'lat': data_reader.dataset()['lat'], 'lon': data_reader.dataset()['lon']})
        return grid

    def _get_inits_and_leads(self, dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        # Get init and lead values, assuming they exist.
        inits = np.atleast_1d(dataset['init'].values)
        leads = np.atleast_1d(dataset['lead'].values.astype(int))
        return inits, leads

    def resample(self, var: str | list[str], lev: float = None, time: tuple[str] = None, use_mp=True):

        '''resample() can be run on datasets with 'time' coordinates'''

        if self._to_resample == 0:
            print("resampling disabled")
            return None

        var_list = [var] if isinstance(var, str) else var
        if len(var_list) > 2:
            raise ValueError('To conserve computational resources, you can only resample up to 2 variables at a time.')

        # retrieve subset as xarray Dataset
        data_reader_for_resample = getattr(self, f'_data_reader{self._to_resample}')
        ds_for_resample = data_reader_for_resample.retrieve(var=var_list, lev=lev, time=time)

        # preserve these special attributes
        WINDS = data_reader_for_resample.WINDS

        # This logic depends on retrieve() resetting the coordinates after slicing by a single level
        is_flat, _ = data_reader_for_resample.is_flat(dataset=ds_for_resample)
        if is_flat is not True:
            raise KeyError(f'You must specify a single vertical level to resample.')

        # Submit data to _resample() method.
        self.resampled = dr.getDataReader(datasource='supplied',
                                          dataset=self._resample(ds_for_resample, use_mp=use_mp),  # computation
                                          WINDS=WINDS)

        print(f"Resample results stored in <RegridObj>.resampled")
        return None

    def _resample(self, dataset: xr.Dataset, use_mp: bool) -> xr.Dataset:

        print(f"Resampling data_reader{self._to_resample} data to '{self._freq_unit}' using mean aggregation.")

        # Should we show warning if expected computation time is very very long?
        # Get all times in dataset
        all_times = list(dataset.time.values)
        print(f"Resampling from {str(all_times[0]).split(':')[0]} to {str(all_times[-1]).split(':')[0]}")

        # If multiple cores are available we can speed of computation significantly via multiprocessing.
        # However, some shells might not have dependable access to hpc resource.
        # This multiprocessing code could still theoretically work with 1 core.
        start_time = time.time()
        if use_mp is True:

            n_cores = len(os.sched_getaffinity(0))
            print(f'Number of cores available: {n_cores}')

            # batches are [(start_time, end_time),...]
            batches = timeutil.datetime_batcher(all_times)
            resample_args = [(dataset, slice(batch[0], batch[1]), self._freq_unit) for batch in batches]

            # Put this here so that subsequent multiprocesses don't inexplicably hang forever.
            mp.set_start_method("spawn", force=True)

            # -- MULTIPROCESS --
            p = mp.Pool(n_cores)
            result = p.starmap(stats.resample, resample_args, chunksize=1)

            p.close()
            p.join()
            # ------------------
            print("Finished multiprocessing.  Concatenating results.")
            result = xr.concat(result, dim='time', coords='different', compat='equals').sortby('time')
            gc.collect()
        else:
            result = stats.resample(dataset, slice(all_times[0], all_times[-1]), self._freq_unit)

        end_time = time.time()
        print(f'Resample completed in {round((end_time-start_time)/60.0, 2)} minutes.')

        return result

    def regrid(self, var: str | list[str], **kwargs):

        '''var can be a string or a list of two wind vector variables'''
        msg = f'Regridding data_reader{self._highres_grid} grid ({self._type_highres})'
        msg += f' onto data_reader{self._lowres_grid} grid ({self._type_lowres})'
        print(msg)

        # Get the dataset that needs spatial regridding
        # If resampling is needed but resample has not been calculated...
        if self._to_resample != 0 and self.resampled is None:
            msg = f'Resample your verification data first, by running'
            msg += f' <RegridObj>.resample(var=<var>, lev=<lev>, time=time_range)'
            raise ValueError(msg)

        # If the resampled data is also the data that shall be regridded...
        if self._highres_grid == self._to_resample:
            data_reader_to_regrid = self.resampled

        # Else get the highest resolution dataset
        else:
            data_reader_to_regrid = getattr(self, f'_data_reader{self._highres_grid}')

        # This is for checking whether vector data needs to be regridded
        WINDS = data_reader_to_regrid.WINDS

        run_sphere = False  # <-- Can be toggled TRUE with following logic:

        # -- BEGIN input parameter checks ---------------------------------------------------------------------------
        # Check if input var are wind vectors
        if isinstance(var, list):
            if len(var) == 0:
                raise ValueError(f'Must supply var as a string or list of two wind vectors. Got empty list [].')

            elif len(var) > 2:
                msg = f'If var is a list, then it can only contain a maximum of two variables,'
                msg += f' both of which must be wind vectors. Got {var}'
                raise ValueError(msg)

            elif len(var) == 2:
                # Then this should be wind vector data.
                n_uv_in_verif = 0
                for uvset in WINDS:
                    uv_in_verif = set([uvset['U_WIND'], uvset['V_WIND']]) & set(var)
                    n_uv_in_verif = max(n_uv_in_verif, len(uv_in_verif))

                if n_uv_in_verif == 1:
                    msg = f'You supplied one wind vector ({list(uv_in_verif)[0]})'
                    msg += f' but not the other, so spherical harmonics cannot be run.'
                    raise ValueError(msg)

                elif n_uv_in_verif == 0:
                    msg = f'You supplied a list of variables that are not registered as wind vectors in this dataset.'
                    msg += f"\nRun <RegridObj>.regrid_vars() to get a complete list of variables available for regrid."
                    raise ValueError(msg)

                # Check that domain is global
                if kwargs.get('lat', None) is not None or kwargs.get('lon', None) is not None:
                    msg = f'Spherical harmonics can only be run on the full global domain.'
                    msg += f'\nIf you want to regrid wind vector data, do not slice by lat-lon yet.'
                    raise ValueError(msg)

                run_sphere = True  # Green light to run spherical harmonics.  var can be a list.

            elif len(var) == 1:
                # Check that the variable isn't a wind vector
                var = var[0]

        # In case user enters a non-string
        if not isinstance(var, str) and run_sphere is False:
            msg = f'''var must be a string representing a single variable in the dataset.'''
            raise ValueError(msg)

        # Check if input var are wind vectors
        if isinstance(var, str):
            for uvset in WINDS:
                if var in [uvset['U_WIND'], uvset['V_WIND']]:
                    msg = f'You supplied one wind vector {var} but not the other, so spherical harmonics cannot be run.'
                    raise ValueError(msg)

        if 'member' in list(data_reader_to_regrid.dataset().dims):
            if 'member' not in kwargs and kwargs.get('ens_avg', None) is not True:
                msg = f'To regrid an ensemble model, you must specify a member or set ens_avg=True'
                raise ValueError(msg)

        # -- END input parameter checks -----------------------------------------------------------------------------

        # All parameter checks done, assign var
        kwargs['var'] = var

        # TODO: Make this more intelligent.  What if one dataset has a smaller time range?
        # Do a generic retrieve on the model dataset in order to process init+lead times
        model_number = 1 if self._is_dr1_UFS is True else 2
        model_data_reader = getattr(self, f'_data_reader{model_number}')
        varlist = list(model_data_reader.dataset().keys())
        model_ds = model_data_reader.retrieve(var=varlist, time=kwargs.get('time', None))

        if 'time' in kwargs:
            kwargs.pop('time')  # Don't need time anymore, handled by match_time_to_leads

        # Get dataset to spatially regrid (Note: It could be UFS or verif/obs depending on resolution)
        # If already resampled, lev would have been dropped after resetting coords. (resampling requires flatness)
        is_flat, vert_coords = data_reader_to_regrid.is_flat()

        # This is the first time doing a retrieve on the data-to-regrid.
        # We might encounter errors if the user hasn't run the appropriate preprocessing.
        # e.g., resampled different variables than the ones currently being requested for regrid.
        try:
            if is_flat is True:
                to_regrid_ds = data_reader_to_regrid.retrieve(**{x: kwargs[x] for x in kwargs if x not in vert_coords})
            else:
                to_regrid_ds = data_reader_to_regrid.retrieve(**kwargs)
        except Exception as e:
            msg = f"Encountered error retrieving data for regrid: {e}"
            msg += f"\nCheck that you resampled the corresponding variables first."
            msg += f"\nRun <RegridObj>.regrid_vars() to get a complete list of variables available for regrid."
            raise ValueError(msg)

        # Do this again to make sure that the final retrieved dataset is flat.
        # TODO: I do not like this code.
        is_flat, _ = data_reader_to_regrid.is_flat(dataset=to_regrid_ds)
        if is_flat is not True:
            raise KeyError(f'You must specify a single vertical level to regrid.')

        to_regrid_ds = timeutil.match_time_to_leads(verif_ds=to_regrid_ds,
                                                    ufs_ds=model_ds)

        # REGRID SPHERE ##
        if run_sphere is True:
            # Distinguish u from v. At this point we've already confirmed that the correct variables are present.
            for uvset in WINDS:
                if set(var) == set(list(uvset.values())):
                    U_WIND_VAR = uvset['U_WIND']
                    V_WIND_VAR = uvset['V_WIND']
            # Run sphere
            to_regrid_ds = self._run_sphere(to_regrid_ds, U_WIND_VAR, V_WIND_VAR)

        # REGRID SCALAR ##
        else:
            print(f"Running scalar regrid on {', '.join(list(to_regrid_ds.keys()))}")
            start_time = time.time()
            # to_regrid_ds = self.regridder(to_regrid_ds)
            to_regrid_ds = self.regridder.regrid(to_regrid_ds, var)
            end_time = time.time()
            print(f"Completed scalar regrid in {round((end_time-start_time)/60.0, 2)} minutes.")

        # Return a data_reader object
        self.regridded = dr.getDataReader(datasource='supplied',
                                          dataset=to_regrid_ds,
                                          WINDS=WINDS)

        print(f"Regrid results stored in <RegridObj>.regridded")

    def _run_sphere(self, dataset: xr.Dataset, U_WIND_VAR: str, V_WIND_VAR: str) -> xr.Dataset:

        # Our spherepack-based code expects 1 time dimension and no vertical levels of any kind.
        # Therefore, if dealing with init+leads, make a slice at each init,
        # and treat the leads as if they were times.  Merge results at the end.
        if 'init' in dataset.dims:
            model_inits = np.atleast_1d(dataset['init'].values)
            iterator_name = 'init'
            iterator_values = model_inits
        else:
            iterator_name = None
            iterator_values = [None]

        # If lev is a dimension in the data, then drop it temporarily, work on the underlying arrays, and add it back.
        stored_lev = None
        if 'lev' in dataset.dims:
            stored_lev = dataset.lev.values[0]  # upstream logic has already confirmed that these data are flat.
            dataset = dataset.squeeze(dim='lev')

        # Each time slice will be appended here, to be merged at the end.
        results = []

        print(f"Running spherical harmonics on {U_WIND_VAR} and {V_WIND_VAR}")
        # Iterate
        for this_it_value in iterator_values:

            # Are we looping over inits or not?
            if this_it_value is not None:
                temp_ds = dataset.sel(init=this_it_value)
            else:
                temp_ds = dataset

            # Get shape and dims.  We use to create dummy data for xesmf
            data_shape = temp_ds[list(temp_ds.keys())[0]].shape
            data_dims = list(temp_ds.dims)
            dummy_data = np.full(data_shape, np.nan)

            # Get u and v input vectors
            u_input = temp_ds[U_WIND_VAR].values
            v_input = temp_ds[V_WIND_VAR].values

            # -----------------
            # REGRID VECTOR U-V
            start_time = time.time()
            u_output, v_output = self._regrid_vector_spharm(u_input, v_input)
            end_time = time.time()
            # -----------------

            # Drop u-v fields from the dataset. The spherical harmonic results will be added back.
            temp_ds = temp_ds.drop_vars([U_WIND_VAR, V_WIND_VAR])
            temp_ds = temp_ds.assign(dummy_variable=(data_dims, dummy_data))

            # Run xesmf regridder on dummy data, then drop the dummy data
            # temp_ds = self.regridder(temp_ds)
            if iterator_name == 'init':
                temp_ds = temp_ds.expand_dims('init').assign_coords({'init': [this_it_value]})
                temp_ds = self.regridder.regrid(temp_ds, 'dummy_variable')
                temp_ds = temp_ds.drop_vars('dummy_variable')
                temp_ds = temp_ds.squeeze('init', drop=True)
            else:
                temp_ds = self.regridder.regrid(temp_ds, 'dummy_variable')
                temp_ds = temp_ds.drop_vars('dummy_variable')

            # Assign new u-v data to the new grid
            if iterator_name == 'init':
                temp_ds = temp_ds.assign(u_wind=(('lead', 'lat', 'lon'), u_output))
                temp_ds = temp_ds.assign(v_wind=(('lead', 'lat', 'lon'), v_output))
                temp_ds = temp_ds.expand_dims('init')
                temp_ds = temp_ds.assign_coords({'init': [this_it_value]})
            # Otherwise, we know we're working with 'time'.
            else:
                temp_ds = temp_ds.assign(u_wind=(('time', 'lat', 'lon'), u_output))
                temp_ds = temp_ds.assign(v_wind=(('time', 'lat', 'lon'), v_output))

            # Rename back to origin names
            temp_ds = temp_ds.rename({'u_wind': U_WIND_VAR, 'v_wind': V_WIND_VAR})

            # Append this time slice to the results list.
            results.append(temp_ds)

        # Merge results into 1 dataset.
        results = xr.merge(results, join='outer', compat='no_conflicts')

        # Add back lev, if it existed.
        if stored_lev is not None:
            results = results.expand_dims(lev=[stored_lev])

        print(f"Completed sperical harmonics in {round((end_time-start_time)/60.0, 2)} minutes.")

        return results

    def _regrid_vector_spharm(self, u_input: np.ndarray, v_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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

        grid_in = getattr(self, f'_grid{self._highres_grid}')
        grid_out = getattr(self, f'_grid{self._lowres_grid}')

        nlats_in = len(list(np.unique(grid_in.lat.values)))
        nlons_in = len(list(np.unique(grid_in.lon.values)))

        nlats_out = len(list(np.unique(grid_out.lat.values)))
        nlons_out = len(list(np.unique(grid_out.lon.values)))

        # Make pyspharm grids
        spharm_in = spharm.Spharmt(nlons_in, nlats_in, legfunc='stored', gridtype='regular')
        spharm_out = spharm.Spharmt(nlons_out, nlats_out, legfunc='stored', gridtype='regular')

        # Simplify variable names
        nlat = spharm_in.nlat
        nlon = spharm_in.nlon
        nlat_out = spharm_out.nlat
        nlon_out = spharm_out.nlon

        if nlat % 2:  # nlat is odd
            n2 = (nlat + 1) / 2
        else:
            n2 = nlat / 2

        nt = u_input.shape[2]  # 1 means 1 timestep
        ntrunc = spharm_out.nlat - 1

        w = u_input
        v = - v_input

        lwork = (2 * nt + 1) * nlat * nlon
        lwork_out = (2 * nt + 1) * nlat_out * nlon_out

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

    def align(self, tolerance: Optional[np.timedelta64] = None):

        ''''align() only operates on data with time dimension, and thus never UFS datasets.'''

        # print("\nDEBUG: Starting temporal alignment")
        # print(f"       Model init count: {len(model_inits)}")
        # print(f"       Model lead count: {len(model_leads)}")
        # print(f"       Verification time count: {len(verif['time'])}")
        # print(self._freq_unit)

        # Logic checks.
        # Determine if work still needs to be done before alignment.
        # If not, determine which dataset to operate on.
        # --------------------------------------------------------------------------
        # Is there a dataset to align?
        if self._to_resample == 0:
            print(f"Both datasets have init+lead structure, so there are no time coordinates to align.")
            return None

        # Has the dataset been resampled yet (if it needs to be)?
        if self.resampled is None:
            msg = f'Resample your data first.'
            raise ValueError(msg)

        # Has the dataset been regridded yet (if it needs to be)?
        if self._highres_grid != 0 and self.regridded is None:
            msg = f'Regrid your data first.'
            raise ValueError(msg)

        # Now we can grab the appropriate dataset
        # Both grids have the same spatial resolution, meaning self.regridded is irrelevant.
        if self._highres_grid == 0:

            # If resampling is also irrelevant, then grab the non-UFS dataset.
            # (Does this first logic gate mean anything?  Reconsider.)
            if self._to_resample == 0:
                to_align_datareader = self._data_reader2 if self._is_dr1_UFS is True else self._data_reader1

            # If resampling is necessary, then pick up the resampled dataset
            else:
                to_align_datareader = self.resampled

        # Else, regridded is not irrelevant
        else:
            # If resampled and regridded dataset are the same, then grab regridded
            if self._to_resample == self._highres_grid:
                to_align_datareader = self.regridded
            # Otherwise get the resampled
            else:
                to_align_datareader = self.resampled
        # --------------------------------------------------------------------------
        # End logic checks.

        # Extract the underlying xarray dataset
        to_align_ds = to_align_datareader.dataset()
        WINDS = to_align_datareader.WINDS

        if 'init' in to_align_ds.dims:
            print(f'Dataset already has init+lead dimensions; returning None.')
            return None

        print(f'Aligning time coordinate to init+lead coordinates.')

        # TODO: Make this more intelligent.
        # Do a generic retrieve on the model dataset in order to process init+lead times
        model_number = 1 if self._is_dr1_UFS is True else 2
        model_data_reader = getattr(self, f'_data_reader{model_number}')
        varlist = list(model_data_reader.dataset().keys())

        first_time = sorted(to_align_ds.time.values)[0]
        final_time = sorted(to_align_ds.time.values)[-1]

        # Store the original model inits before time slicing.
        orig_model_inits, _ = self._get_inits_and_leads(model_data_reader.dataset())

        # UFS dataset
        model_ds = model_data_reader.retrieve(var=varlist, time=(str(first_time), str(final_time)))

        # Get inits and leads from UFS dataset
        model_inits, model_leads = self._get_inits_and_leads(model_ds)

        if len(model_inits) == 0:
            msg = f"Error in .align(): Your time range ({first_time}, {final_time}) does not include any Model Inits.  "
            msg += f"This time range may be an artifact of .resample().  "
            msg += f"Please adjust your time range to include at least 1 Init, and, if necessary rerun .resample().  "
            msg += f"Here is a list of Inits found in the Model dataset:\n {orig_model_inits}"
            raise ValueError(msg)

        # Add an init dimension
        to_align_ds = to_align_ds.expand_dims('init')

        if tolerance is None:
            if self._freq_unit == "MS":
                tolerance = np.timedelta64(16, 'D')
            elif self._freq_unit == "D":
                tolerance = np.timedelta64(12, 'h')
            elif self._freq_unit == "H":
                tolerance = np.timedelta64(1, 'h')
            else:
                tolerance = np.timedelta64(1, 'D')

            # print(f"DEBUG: Auto-selected tolerance = {tolerance}")

        to_align_ds = to_align_ds.sortby('time')
        to_align_ds_times = to_align_ds['time'].values.astype('datetime64[ns]')
        aligned = []
        matched, filled = 0, 0
        print("MODEL_LEADS:", model_leads)
        for init_idx, init in enumerate(model_inits):
            lead_slices = []
            # print(f"\nDEBUG: Processing init[{init_idx}] = {init}")

            for lead_idx, lead in enumerate(model_leads):
                # Forecast time calculation
                forecast_time = timeutil.time_offset(self._freq_unit, init, lead, self._step)

                # Find closest match
                deltas = np.abs(to_align_ds_times - forecast_time)
                idx = deltas.argmin()
                delta = deltas[idx]

                # If this timestep exists in both datasets, insert to_align_ds data
                if delta <= tolerance:
                    this_lead_slice = to_align_ds.isel(time=idx)

                    # Control time coordinate
                    this_lead_slice = this_lead_slice.drop_vars('time')
                    this_lead_slice = this_lead_slice.assign_coords({'time': forecast_time})

                    lead_slices.append(this_lead_slice)
                    matched += 1
                # Else this timestep is in model but not to_align_ds, then insert dummy NA data
                else:
                    fill = to_align_ds.isel(time=0).copy(deep=True)

                    # Control time coordinate
                    fill = fill.drop_vars('time')
                    fill = fill.assign_coords({'time': forecast_time})

                    for var in list(fill.keys()):
                        fill[var][:] = np.nan

                    lead_slices.append(fill)
                    filled += 1

            lead_stack = xr.concat(lead_slices, dim='lead', coords='different', compat='equals')
            aligned.append(lead_stack)

        final = xr.concat(aligned, dim='init', coords='different', compat='equals')
        final = final.assign_coords(init=('init', model_inits), lead=('lead', model_leads))

        final.attrs['model_freq_unit'] = self._freq_unit
        final.attrs['lead_step'] = str(self._step)
        final.attrs['alignment_tolerance'] = str(tolerance)

        print(f"Time dimensions aligned:  matched {matched} timesteps")
        if filled > 0:
            msg = f'Verif data not available for {filled} timesteps, filled with NaN.'
            warnings.warn(msg)

        self.aligned = dr.getDataReader(datasource='supplied',
                                        dataset=final,
                                        WINDS=WINDS)
        print(f"Align results stored in <RegridObj>.aligned")

#    def check_alignment(self, model: xr.DataArray, verif: xr.DataArray):
#        print(f"DEBUG: Model shape: {model.shape}")
#        print(f"       Model init range: {model.init.values.min()} → {model.init.values.max()}")
#        print(f"       Model lead range: {model.lead.values.min()} → {model.lead.values.max()}")

#        print(f"DEBUG: Verification shape: {verif.shape}")
#        print(f"       Verification init range: {verif.init.values.min()} → {verif.init.values.max()}")
#        print(f"       Verification lead range: {verif.lead.values.min()} → {verif.lead.values.max()}")
#        print(f"       Verification lead unit: {verif.attrs.get('model_freq_unit', 'unknown')}")
#        print(f"       Lead step size: {verif.attrs.get('lead_step', 'unknown')}")
