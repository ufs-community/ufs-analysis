# ---------------------------------------------------------------------------------------------------------------------
#  Filename: rws.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 05 Nov. 2025
#  Purpose: Functions for calculating Rossby Wave Source and its components.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
import spharm


def calc_planetaryvorticity(da: xr.DataArray) -> np.ndarray:

    lats = da.lat.values
    omega = 7.292115e-5  # (rad/s) wiki

    # Coriolis parameter
    cp = 2. * omega * np.sin(np.deg2rad(lats))

    indices = [slice(0, None)] + [np.newaxis] * (len(da.shape) - 1)

    f = cp[tuple(indices)] * np.ones(da.shape, dtype=np.float32)

    return f


def calc_absvrt(ds: xr.Dataset, uvar: str, vvar: str) -> xr.Dataset:

    # Planetary vorticity
    pvrt = calc_planetaryvorticity(ds[uvar])

    spharm_obj = spharm.Spharmt(len(ds.lon.values), len(ds.lat.values), legfunc='stored', gridtype='regular')

    vrtspec, divspec = spharm_obj.getvrtdivspec(ugrid=ds[uvar].values,
                                                vgrid=ds[vvar].values,
                                                ntrunc=len(ds.lat.values) - 1)
    # Relative vorticity
    rvrt = spharm_obj.spectogrd(vrtspec)

    # Abosolute vorticity
    vrt = pvrt + rvrt

    # Assign result to ds
    ds = ds.assign(absvrt=(('lat', 'lon'), vrt))

    # print('Finished Vorticity calculations.')

    return ds


def calc_divergence(ds: xr.Dataset, uvar: str, vvar: str) -> xr.Dataset:

    spharm_obj = spharm.Spharmt(len(ds.lon.values), len(ds.lat.values), legfunc='stored', gridtype='regular')

    vrtspec, divspec = spharm_obj.getvrtdivspec(ugrid=ds[uvar].values,
                                                vgrid=ds[vvar].values,
                                                ntrunc=len(ds.lat.values) - 1)

    divgrid = spharm_obj.spectogrd(divspec)

    ds = ds.assign(divergence=(('lat', 'lon'), divgrid))

    # print('Finished Divergence calculations.')

    return ds


def calc_irrotationalcomponents(ds: xr.Dataset, uvar: str, vvar: str) -> xr.Dataset:

    spharm_obj = spharm.Spharmt(len(ds.lon.values), len(ds.lat.values), legfunc='stored', gridtype='regular')

    psigrid, chigrid = spharm_obj.getpsichi(ugrid=ds[uvar].values,
                                            vgrid=ds[vvar].values,
                                            ntrunc=len(ds.lat.values) - 1)

    chispec = spharm_obj.grdtospec(chigrid)
    uchi, vchi = spharm_obj.getgrad(chispec)

    ds = ds.assign(uchi=(('lat', 'lon'), uchi))
    ds = ds.assign(vchi=(('lat', 'lon'), vchi))

    # print('Finished Irrotational Components calculations.')

    return ds


def calc_gradient(ds: xr.Dataset, var: str) -> xr.Dataset:

    spharm_obj = spharm.Spharmt(len(ds.lon.values), len(ds.lat.values), legfunc='stored', gridtype='regular')

    chispec = spharm_obj.grdtospec(ds[var].values, ntrunc=len(ds.lat.values) - 1)

    etax, etay = spharm_obj.getgrad(chispec)
    ds = ds.assign(etax=(('lat', 'lon'), etax))
    ds = ds.assign(etay=(('lat', 'lon'), etay))

    # print('Finished Gradient calculations.')

    return ds


def calc_rws_components(ds: xr.Dataset, uvar: str, vvar: str) -> xr.Dataset:
    '''
    Calculate Rossby Wave Source components at time t.
    '''
    # If lev is a dimension in the data, then drop it temporarily, work on the underlying arrays, and add it back.
    stored_lev = None
    if 'lev' in ds.dims:
        stored_lev = ds.lev.values[0]  # upstream logic has already confirmed that these data are flat.
        ds = ds.squeeze(dim='lev')

    original_ds = ds.copy(deep=True)

    if 'time' in ds.dims:
        all_times = list(ds.time.values)

        slices = []
        for this_time in all_times:

            this_ds = ds.sel(time=this_time)

            # Calculate Absolute Vorticity
            this_ds = calc_absvrt(ds=this_ds, uvar=uvar, vvar=vvar)

            # Calculate Irrotational Components (Vx)
            this_ds = calc_irrotationalcomponents(ds=this_ds, uvar=uvar, vvar=vvar)

            this_ds = this_ds[['absvrt', 'uchi', 'vchi']]
            slices.append(this_ds)

        final_result = xr.concat(slices, dim='time', coords='minimal', compat='equals').sortby('time')

    # UFS handling
    else:
        inits = np.atleast_1d(ds['init'].values)
        leads = np.atleast_1d(ds['lead'].values.astype(int))

        stack = []
        for this_init in inits:
            lead_slices = []

            for this_lead in leads:

                this_ds = ds.sel(init=this_init, lead=this_lead)

                # Calculate Absolute Vorticity
                this_ds = calc_absvrt(ds=this_ds, uvar=uvar, vvar=vvar)

                # Calculate Irrotational Components (Vx)
                this_ds = calc_irrotationalcomponents(ds=this_ds, uvar=uvar, vvar=vvar)

                this_ds = this_ds[['absvrt', 'uchi', 'vchi']]

                # Append results
                lead_slices.append(this_ds)

            lead_stack = xr.concat(lead_slices, dim='lead', coords='different', compat='equals')
            stack.append(lead_stack)

        # Finished calculations
        final_result = xr.concat(stack, dim='init', coords='different', compat='equals')
        final_result = final_result.assign_coords(init=('init', inits), lead=('lead', leads))

    # Merge results into original dataset
    final_result = xr.merge([final_result, original_ds], compat='no_conflicts')

    # Add back lev, if it existed.
    if stored_lev is not None:
        final_result = final_result.expand_dims(lev=[stored_lev])

    return final_result


def calc_rws(ds: xr.Dataset,
             absvrt_stats: dict,  # Absolute Vorticity
             absvrt_anomaly: xr.Dataset,
             uchi_stats: dict,  # UCHI
             uchi_anomaly: xr.Dataset,
             vchi_stats: dict,  # VCHI
             vchi_anomaly: xr.Dataset) -> xr.Dataset:
    '''
    Calculate Rossby Wave Source at time t.
    '''

    # If lev is a dimension in the data, then drop it temporarily, work on the underlying arrays, and add it back.
    stored_lev = None
    if 'lev' in ds.dims:
        stored_lev = ds.lev.values[0]  # upstream logic has already confirmed that these data are flat.
        ds = ds.squeeze(dim='lev')

        # Additionally, squeeze the statistical data.
        absvrt_anomaly = absvrt_anomaly.squeeze(dim='lev')
        uchi_anomaly = uchi_anomaly.squeeze(dim='lev')
        vchi_anomaly = vchi_anomaly.squeeze(dim='lev')

        for key in absvrt_stats.keys():
            if absvrt_stats[key] is not None and 'lev' in absvrt_stats[key].dims:
                absvrt_stats[key] = absvrt_stats[key].squeeze(dim='lev')

            if uchi_stats[key] is not None and 'lev' in uchi_stats[key].dims:
                uchi_stats[key] = uchi_stats[key].squeeze(dim='lev')

            if vchi_stats[key] is not None and 'lev' in vchi_stats[key].dims:
                vchi_stats[key] = vchi_stats[key].squeeze(dim='lev')

    original_ds = ds.copy(deep=True)

    if 'time' in ds.dims:
        all_times = list(ds.time.values)

        slices = []
        for this_time in all_times:

            this_ds = ds.sel(time=this_time)

            # Get month number
            this_month_number = this_time.astype('datetime64[M]').astype(int) % 12 + 1

            # Absolute Vorticity
            this_vrt_clim = absvrt_stats['climatology_mean'].sel(month=this_month_number)
            this_vrt_anom = absvrt_anomaly.sel(time=this_time)['anomaly']

            # Uchi and Vchi
            this_uchi_clim = uchi_stats['climatology_mean'].sel(month=this_month_number)
            this_uchi_anom = uchi_anomaly.sel(time=this_time)['anomaly']

            this_vchi_clim = vchi_stats['climatology_mean'].sel(month=this_month_number)
            this_vchi_anom = vchi_anomaly.sel(time=this_time)['anomaly']

            # Merge Uchi and Vchi into one dataset.
            this_uchi_vchi_clim = xr.merge([this_uchi_clim, this_vchi_clim], compat='no_conflicts')

            this_uchi_vchi_anom = xr.merge([this_uchi_anom.to_dataset().rename({'anomaly': 'uchi_anomaly'}),
                                            this_vchi_anom.to_dataset().rename({'anomaly': 'vchi_anomaly'})],
                                           compat='no_conflicts')
            # Gradients
            this_gradient_vrt_clim = calc_gradient(this_vrt_clim.to_dataset(), 'absvrt')
            this_etax_clim = this_gradient_vrt_clim['etax']
            this_etay_clim = this_gradient_vrt_clim['etay']

            this_gradient_vrt_anom = calc_gradient(this_vrt_anom.to_dataset(), 'anomaly')
            this_etax_anom = this_gradient_vrt_anom['etax']
            this_etay_anom = this_gradient_vrt_anom['etay']

            # Total Divergence (del dot vchi)
            this_divergence_clim = calc_divergence(ds=this_uchi_vchi_clim, uvar='uchi', vvar='vchi')
            this_divergence_anom = calc_divergence(ds=this_uchi_vchi_anom, uvar='uchi_anomaly', vvar='vchi_anomaly')

            # Calculate RWS components
            # First term
            first_term = this_vrt_clim.values * this_divergence_anom['divergence'].values

            # Second term
            second_term = (this_uchi_vchi_anom['uchi_anomaly'].values * this_etax_clim.values)\
                + (this_uchi_vchi_anom['vchi_anomaly'].values * this_etay_clim.values)

            # Third term
            third_term = this_vrt_anom.values * this_divergence_clim['divergence'].values

            # Fourth term
            fourth_term = (this_uchi_vchi_clim['uchi'].values * this_etax_anom.values)\
                + (this_uchi_vchi_clim['vchi'].values * this_etay_anom.values)

            # Combine RWS terms
            rws_array = -1.0 * (first_term + second_term + third_term + fourth_term)

            this_ds = this_ds.assign(first_term=(('lat', 'lon'), first_term))
            this_ds = this_ds.assign(second_term=(('lat', 'lon'), second_term))
            this_ds = this_ds.assign(third_term=(('lat', 'lon'), third_term))
            this_ds = this_ds.assign(fourth_term=(('lat', 'lon'), fourth_term))

            this_ds = this_ds.assign(rws=(('lat', 'lon'), rws_array))  # RWS

            slices.append(this_ds)

        final_result = xr.concat(slices, dim='time', coords='minimal', compat='equals').sortby('time')

    # UFS handling
    else:
        inits = np.atleast_1d(ds['init'].values)
        leads = np.atleast_1d(ds['lead'].values.astype(int))

        stack = []
        for this_init in inits:
            lead_slices = []

            for this_lead in leads:

                this_ds = ds.sel(init=this_init, lead=this_lead)

                # Get month number
                this_month_number = this_init.astype('datetime64[M]').astype(int) % 12 + 1

                # Absolute Vorticity
                this_vrt_clim = absvrt_stats['climatology_mean'].sel(month=this_month_number, lead=this_lead)
                this_vrt_anom = absvrt_anomaly.sel(init=this_init, lead=this_lead)['anomaly']

                # Uchi and Vchi
                this_uchi_clim = uchi_stats['climatology_mean'].sel(month=this_month_number, lead=this_lead)
                this_uchi_anom = uchi_anomaly.sel(init=this_init, lead=this_lead)['anomaly']

                this_vchi_clim = vchi_stats['climatology_mean'].sel(month=this_month_number, lead=this_lead)
                this_vchi_anom = vchi_anomaly.sel(init=this_init, lead=this_lead)['anomaly']

                # Merge Uchi and Vchi into one dataset.
                this_uchi_vchi_clim = xr.merge([this_uchi_clim, this_vchi_clim], compat='no_conflicts')

                this_uchi_vchi_anom = xr.merge([this_uchi_anom.to_dataset().rename({'anomaly': 'uchi_anomaly'}),
                                                this_vchi_anom.to_dataset().rename({'anomaly': 'vchi_anomaly'})],
                                               compat='no_conflicts')
                # Gradients
                this_gradient_vrt_clim = calc_gradient(this_vrt_clim.to_dataset(), 'absvrt')
                this_etax_clim = this_gradient_vrt_clim['etax']
                this_etay_clim = this_gradient_vrt_clim['etay']

                this_gradient_vrt_anom = calc_gradient(this_vrt_anom.to_dataset(), 'anomaly')
                this_etax_anom = this_gradient_vrt_anom['etax']
                this_etay_anom = this_gradient_vrt_anom['etay']

                # Total Divergence (del dot vchi)
                this_divergence_clim = calc_divergence(ds=this_uchi_vchi_clim, uvar='uchi', vvar='vchi')
                this_divergence_anom = calc_divergence(ds=this_uchi_vchi_anom, uvar='uchi_anomaly', vvar='vchi_anomaly')

                # Calculate RWS components
                # First term
                first_term = this_vrt_clim.values * this_divergence_anom['divergence'].values

                # Second term
                second_term = (this_uchi_vchi_anom['uchi_anomaly'].values * this_etax_clim.values)\
                    + (this_uchi_vchi_anom['vchi_anomaly'].values * this_etay_clim.values)

                # Third term
                third_term = this_vrt_anom.values * this_divergence_clim['divergence'].values

                # Fourth term
                fourth_term = (this_uchi_vchi_clim['uchi'].values * this_etax_anom.values)\
                    + (this_uchi_vchi_clim['vchi'].values * this_etay_anom.values)

                # Combine RWS terms
                rws_array = -1.0 * (first_term + second_term + third_term + fourth_term)

                this_ds = this_ds.assign(first_term=(('lat', 'lon'), first_term))
                this_ds = this_ds.assign(second_term=(('lat', 'lon'), second_term))
                this_ds = this_ds.assign(third_term=(('lat', 'lon'), third_term))
                this_ds = this_ds.assign(fourth_term=(('lat', 'lon'), fourth_term))

                this_ds = this_ds.assign(rws=(('lat', 'lon'), rws_array))  # RWS

                # Append results
                lead_slices.append(this_ds)

            lead_stack = xr.concat(lead_slices, dim='lead', coords='different', compat='equals')
            stack.append(lead_stack)

        # Finished calculations
        final_result = xr.concat(stack, dim='init', coords='different', compat='equals')
        final_result = final_result.assign_coords(init=('init', inits), lead=('lead', leads))

    # Merge results into original dataset
    final_result = xr.merge([final_result, original_ds], compat='no_conflicts')

    # Add back lev, if it existed.
    if stored_lev is not None:
        final_result = final_result.expand_dims(lev=[stored_lev])

    return final_result
