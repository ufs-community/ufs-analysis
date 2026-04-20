# ---------------------------------------------------------------------------------------------------------------------
#  Filename: stats.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 07 Oct. 2025
#  Purpose: Utility for model and climate statistics.
# ---------------------------------------------------------------------------------------------------------------------

from typing import Union, Tuple
import copy
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import itertools
from . import timeutil, rws


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Cantarell'] + plt.rcParams['font.sans-serif']
plt.rcParams['font.size'] = 14


def resample(ds: xr.Dataset, timeslice: Tuple[str, ...], freq) -> xr.Dataset:
    return ds.sel(time=timeslice).resample(time=freq).mean().sortby('time').compute()


def calc_climatology_anomaly(ds: xr.Dataset, area_mean=False, use_member_climatology=True) -> dict:

    '''
    Compute longterm, area-based, monthly climatologies.
    If the 'time' coordinate is present, then we get 1 climatology per (month).
    If 'init'+'lead' coordinates are present, then we get 1 climatology per (init, lead)
    Results are loaded into memory.
    Returns index: xr.Dataset, clim: xr.Dataset
    '''

    results = dict.fromkeys(['monthly_mean',
                             'monthly_std',
                             'climatology_mean',
                             'climatology_std'])

    # These statistics are optional
    index_std = None
    climatology_std = None

    var = list(ds.keys())
    if len(var) > 1:
        raise ValueError('Only calculate climatology for 1 variable at a time, please.')

    var = var[0]  # Convert list to str value

    # Get monthly mean if this is time-based data (i.e., not UFS which is already monthly meaned)
    if 'time' in ds.dims:
        ds = ds.resample(time='MS').mean()

    # Compute area mean and std
    if area_mean is True:
        results['monthly_std'] = ds.std(['lat', 'lon'])[var].compute()

        # save ds for for next step
        ds = ds.mean(['lat', 'lon'])
        results['monthly_mean'] = ds[var].compute()
    else:
        results['monthly_mean'] = ds[var]
        # ^ If this is UFS data and we're not calculating area mean, then 'monthly_mean' is identical to input data.
        # results['monthly_std'] is None

    # Compute climatological mean for each month of the year as a function of init.month
    if 'init' in ds.dims:
        results['climatology_mean'] = ds.groupby('init.month').mean(['init'])[var].compute()
        results['climatology_std'] = ds.groupby('init.month').std(['init'])[var].compute()

    # or per time.month
    elif 'time' in ds.dims:
        results['climatology_mean'] = ds.groupby('time.month').mean(['time'])[var].compute()
        results['climatology_std'] = ds.groupby('time.month').std(['time'])[var].compute()

    # Calculate Anomaly as a final step.
    anomaly = calc_anomaly(ds=results['monthly_mean'].to_dataset(),
                           var=var,
                           stats=results,
                           use_member_climatology=use_member_climatology)

    anomaly = anomaly[['anomaly']]  # the anomaly function returns the original data too.  Here, just take anomaly.

    # Assign anomaly results to dictionary
    results['anomaly'] = anomaly

    return results


def calc_anomaly(ds: xr.Dataset, var: str, stats: dict, use_member_climatology=True) -> xr.Dataset:

    '''climatology stats have already been computed, now calculate anomaly'''

    if use_member_climatology is False and 'init' in ds.dims:

        # We have a few special notebooks that deal with ECMWF hindcast data,
        # These data have init+month, but we only care about ensemble means, not members.
        if 'member' in list(ds.coords):
            for this_member in ds.member.values:
                stats['climatology_mean'] = stats['climatology_mean'].where(
                    stats['climatology_mean'].member == this_member, stats['climatology_mean'].sel(member=-1))

    ds = ds.assign(anomaly=xr.DataArray(np.nan, dims=ds.dims, coords=ds.coords))

    if 'init' in ds.dims:
        for i in range(len(ds[var].init.values)):

            this_init = ds.init.values[i]
            this_init_month = this_init.astype('datetime64[M]').astype(int) % 12 + 1
            this_climatology = stats['climatology_mean'].sel(month=this_init_month)
            this_anomaly = ds[var].sel(init=this_init) - this_climatology

            # Assign anomaly values to dataset
            ds['anomaly'].loc[{'init': this_init}] = this_anomaly

    else:
        for i in range(len(ds[var].time.values)):

            this_time = ds.time.values[i]
            this_month = this_time.astype('datetime64[M]').astype(int) % 12 + 1

            this_climatology = stats['climatology_mean'].sel(month=this_month)
            this_anomaly = ds[var].sel(time=this_time) - this_climatology

            ds['anomaly'].loc[{'time': this_time}] = this_anomaly

    return ds


def radius(lat: float) -> float:
    '''Calculate radius of ellipse as function of latitude in degrees'''
    # Values from wiki
    r_eq = 6378137.0  # Equatorial radius
    r_po = 6356752.3  # Polar radius

    lat_rad = math.radians(lat)  # Convert latitude to radians

    # Geometry
    c_ = (r_eq**2 * math.cos(lat_rad))**2
    d_ = (r_po**2 * math.sin(lat_rad))**2
    e_ = (r_eq * math.cos(lat_rad))**2
    f_ = (r_po * math.sin(lat_rad))**2

    # Radius as function of lat
    radius = math.sqrt((c_ + d_) / (e_ + f_))

    return radius


def calc_betastar_kwavenumber(ds: xr.Dataset, uvar: str) -> xr.Dataset:

    # If lev is a dimension in the data, then drop it temporarily, work on the underlying arrays, and add it back.
    stored_lev = None
    if 'lev' in ds.dims:
        stored_lev = ds.lev.values[0]  # upstream logic has already confirmed that these data are flat.
        ds = ds.squeeze(dim='lev')

    # Create a copy of the input dataset so we can include all data back in at the end.
    original_ds = ds.copy(deep=True)

    # Calculate beta-effect as function of latitude β=2Ωcos(latitude) / a
    omega = 7.292115e-5  # (rad/s) wiki
    radii = [radius(lat) for lat in ds.lat.values]  # Radius of Earth as a function of lat

    lats_radians = [math.radians(lat) for lat in ds.lat.values]
    lat_cosines = [math.cos(lat) for lat in lats_radians]

    beta_per_lat = [2 * omega * lat_cosines[k] / radii[k] for k in range(len(lats_radians))]
    beta_per_lat = np.array(beta_per_lat).reshape(len(beta_per_lat), 1)

    # Also make a grid of a*cos(phi)
    lat_cosines = np.array(lat_cosines)
    lat_cosines = lat_cosines.reshape(len(lat_cosines), 1)

    # Find position of lon dimension
    lon_position = list(ds[uvar].dims).index('lon')

    if 'time' in ds.dims:
        beta_per_latlon = np.tile(beta_per_lat, (1, ds[uvar].shape[lon_position]))  # lon is here
        lat_cosines_grid = np.tile(lat_cosines, (1, ds[uvar].shape[lon_position]))

    else:
        beta_per_latlon = np.tile(beta_per_lat, (1, ds[uvar].shape[lon_position]))  # lon is here
        lat_cosines_grid = np.tile(lat_cosines, (1, ds[uvar].shape[lon_position]))

    # Convert lats in degrees to lats in meters before differentiating.
    ds = ds.assign_coords(lat_meters=np.radians(ds.lat) * radii)
    ds = ds.swap_dims({"lat": "lat_meters"})

    slices = []

    # Perform calculations for each time step

    radii = np.array(radii).reshape(len(radii), 1)

    if 'time' in ds.dims:
        times = ds.time.values

        for this_time in times:

            # Calculate 2nd partial derivative of U with respect to north-south
            this_ds = ds[[uvar]].sel(time=this_time)
            d2u_dy2 = this_ds[uvar].differentiate(coord='lat_meters').differentiate(coord='lat_meters')

            # Calculate Restoring Effect β* = β-d2u_d2y.
            beta_star = (beta_per_latlon - d2u_dy2)  # / 10e-11
            this_ds = this_ds.assign(beta_star=beta_star)  # Assign results

            # Ks is undefined for negative U.
            neg_u_mask = (this_ds[uvar] < 0)

            # Mask out negative U wind for Ks calculation (do this ~after~ beta* has been calculated)
            this_ds[uvar] = this_ds[uvar].where(~neg_u_mask, drop=False)

            # Calculate stationary wave number Ks = (beta_star / U)^(1/2)
            Ks = radii * lat_cosines_grid * np.sqrt(beta_star.values / this_ds[uvar].values)

            # We expect singularities. Cap values at 11.
            Ks[Ks == np.inf] = 11
            Ks[Ks > 11] = 11  # Cap values at 11.

            # Assign results
            this_ds = this_ds.assign(Ks=(('lat', 'lon'), Ks))

            # Append results
            slices.append(this_ds)

        # Finished calculations
        final_result = xr.concat(slices, dim='time', coords='minimal', compat='equals').sortby('time')

    # Else this must be init+lead structure (UFS)
    else:
        inits = np.atleast_1d(ds['init'].values)
        leads = np.atleast_1d(ds['lead'].values.astype(int))

        stack = []
        for this_init in inits:
            lead_slices = []

            for this_lead in leads:
                # Calculate 2nd partial derivative of U with respect to north-south
                this_ds = ds[[uvar]].sel(init=this_init, lead=this_lead)
                d2u_dy2 = this_ds[uvar].differentiate(coord='lat_meters').differentiate(coord='lat_meters')

                # Calculate Restoring Effect β* = β-d2u_d2y.
                beta_star = (beta_per_latlon - d2u_dy2)  # / 10e-11
                this_ds = this_ds.assign(beta_star=beta_star)  # Assign results

                # Get month number
                this_month_number = this_init.astype('datetime64[M]').astype(int) % 12 + 1

                # Ks is undefined for negative U.
                neg_u_mask = (this_ds[uvar] < 0)

                # Mask out negative U wind for Ks calculation (do this ~after~ beta* has been calculated)
                this_ds[uvar] = this_ds[uvar].where(~neg_u_mask, drop=False)

                # Calculate stationary wave number Ks = (beta_star / U)^(1/2)
                Ks = radii * lat_cosines_grid * np.sqrt(beta_star.values / this_ds[uvar].values)

                # We expect singularities. Cap values at 11.
                Ks[Ks == np.inf] = 11
                Ks[Ks > 11] = 11

                # Assign results
                this_ds = this_ds.assign(Ks=(('lat', 'lon'), Ks))

                # Append results
                lead_slices.append(this_ds)

            lead_stack = xr.concat(lead_slices, dim='lead', coords='different', compat='equals')
            stack.append(lead_stack)

        # Finished calculations
        final_result = xr.concat(stack, dim='init', coords='different', compat='equals')
        final_result = final_result.assign_coords(init=('init', inits), lead=('lead', leads))

    final_result = final_result.swap_dims({'lat_meters': 'lat'})
    final_result = xr.merge([final_result, original_ds], compat='no_conflicts')

    # Before returning result, pass a filter of meanu for which U cannot be < 0.
    ds = ds.swap_dims({"lat_meters": "lat"})
    if 'time' in ds.dims:
        neg_u_mask = (ds[uvar].mean(dim='time', keepdims=False) < 0)
    else:
        neg_u_mask = (ds[uvar].mean(dim=['init', 'lead'], keepdims=False) < 0)

    # Apply the negative-U mask based on the overall mean U.
    final_result['Ks'] = final_result['Ks'].where(~neg_u_mask, drop=False)

    # Add back lev, if it existed.
    if stored_lev is not None:
        final_result = final_result.expand_dims(lev=[stored_lev])

    return final_result


def plot_index_spaghetti(ufs_stats: dict,
                         verif_stats: dict,
                         calc_anomaly=True,
                         use_member_climatology=False,
                         title='',
                         verif_label='Verification',
                         dpi=300):

    '''
    This monster function plot Index and anomaly for UFS and Verif for up to multiple decades.
    '''

    # freq_unit='MS', step=numpy.timedelta64(30,'D')
    freq_unit, step = timeutil.get_lead_resolution(ufs_stats['monthly_mean'])

    all_initmonths = list(ufs_stats['monthly_mean'].groupby('init.month').groups.keys())
    all_years = list(ufs_stats['monthly_mean'].groupby('init.year').groups.keys())

    # We will generate 1 panel per decade.
    decade_batches = timeutil.decade_batcher(all_years)
    print(f'Generating {len(decade_batches)} panel(s).')

    # Instantiate Figure
    fig, axs = plt.subplots(nrows=len(decade_batches), figsize=(12, 4 * len(decade_batches)), dpi=dpi)

    # Run this function once per decade-batch.
    def add_to_plot(this_decade_index,
                    ufs_stats=ufs_stats,
                    verif_stats=verif_stats,
                    calc_anomaly=calc_anomaly,
                    verif_label=verif_label):

        # Is this the very first iteration? (need this info for labeling)
        first_iteration = False
        if this_decade_index == 0:
            first_iteration = True

        min_y_value = np.nan
        max_y_value = np.nan

        # This plot panel will represent these years of data
        these_years = decade_batches[this_decade_index]
        print(f'Processing years {these_years[0]} to {these_years[-1]}')

        # Get all UFS forecasts for this decade-batch
        ufs_index_mask = (ufs_stats['monthly_mean']['init.year'].isin(these_years))\
            & (ufs_stats['monthly_mean']['init.month'].isin(all_initmonths))

        ufs_index = ufs_stats['monthly_mean'].where(ufs_index_mask, drop=True)

        # Get forecast times for every lead at every init
        inits = list(ufs_index.init.values)
        leads = list(ufs_index.lead.values)

        # Gather times, inits, and leads as iterables.
        forecast_times = []
        forecast_inits = []
        forecast_leads = []

        for this_init in inits:
            for this_lead in leads:
                # Calculate forecast datetime
                this_forecast_time = timeutil.time_offset(freq_unit, this_init, this_lead, step)

                # Append to record-keeping lists
                forecast_times.append(this_forecast_time)
                forecast_inits.append(this_init)
                forecast_leads.append(this_lead)

        # Generate separate series for each init
        for i in range(len(ufs_index.member.values)):

            this_member = ufs_index.member.values[i]

            # Get a color for this series.  Pick red if it's the ensemble mean.
            this_linestyle = 'solid'
            this_linewidth = 0.85
            this_label = None
            this_zorder = 4
            this_opacity = 0.6

            if isinstance(this_member, str):

                # Assign a different color to each member.  This will fail if number of members > 5
                colors = ['#4477AA', '#228833', '#CCBB44', '#66CCEE', '#AA3377']  # Paul Tol Bright

                this_color = colors[i]
                this_linewidth = 1.7
                this_opacity = 0.75
                if first_iteration is True:
                    this_label = str(this_member)
                else:
                    this_label = None

            # Group ensemble members into like colors.
            # Colors are based on Paul Tol's muted scheme.
            elif this_member >= 1 and this_member <= 3:
                this_color = '#4477aa'  # Paul Tol
                this_linestyle = 'solid'

                if this_member == 1 and first_iteration is True:
                    this_label = 'Earliest 3 ensemble members'

            elif this_member >= 8 and this_member <= 10:
                this_color = '#66ccee'  # Paul Tol
                this_linestyle = 'solid'

                if this_member == 8 and first_iteration is True:
                    this_label = 'Latest 3 ensemble members'

            elif this_member >= 0:
                this_color = '#ccbb44'  # Paul Tol
                this_linestyle = 'solid'

                if this_member == 0 and first_iteration is True:
                    this_label = 'Ensemble members (other)'

            elif this_member == -1:
                if first_iteration is True:
                    this_label = 'UFS Ensemble Mean'
                else:
                    this_label = None

                # this_color = 'black'
                this_color = '#228833'  # Paul Tol Bright
                this_linestyle = 'solid'
                this_linewidth = 1.75
                this_zorder = 5
                this_opacity = 1.0

            x_values = forecast_times

            if calc_anomaly is False:
                # Compute index (region was already sliced)
                y_values = ufs_index.sel(member=this_member).values

                # y_values is a lists of sublists, where each sublist represents the leads for an init
                # Flatten that list:
                y_values = [item for sublist in y_values for item in sublist]

            # Calculate anomaly
            elif calc_anomaly is True:

                # If plotting anomaly, draw a zero line.
                zero_line_properties = {
                    'y': 0,
                    'color': 'lightgrey',
                    'linestyle': 'solid',
                    'linewidth': 0.2,
                    'zorder': -1000}

                if len(decade_batches) == 1:
                    axs.axhline(**zero_line_properties)
                else:
                    axs[this_decade_index].axhline(**zero_line_properties)

                # Calculate the anomaly
                # ufs_climatology_ens_mean_values = []
                y_values = []
                # Loop over initmonths and their leads.
                for i in range(len(forecast_leads)):
                    this_init = forecast_inits[i]
                    this_lead = forecast_leads[i]

                    # Get anomaly value
                    this_anomaly = ufs_stats['anomaly'].sel(member=this_member,
                                                            init=this_init,
                                                            lead=this_lead)

                    this_anomaly = this_anomaly['anomaly'].values.item()
                    # Append climiatology value to list
                    y_values.append(this_anomaly)

            # Keep track of the minimum and maximum y values for axis standardization
            min_y_value = np.nanmin([np.nanmin(y_values), min_y_value])
            max_y_value = np.nanmax([np.nanmax(y_values), max_y_value])

            # Plot this init's forecast leads
            # Do so one init at a time so that series are discontinuous
            step_size = len(ufs_index.lead.values)
            for i in range(0, len(x_values), step_size):

                these_x_values = x_values[i:i + step_size]
                these_y_values = y_values[i:i + step_size]

                # Set labels to none for every series beyond the first.
                final_label = this_label
                if i != 0:
                    final_label = None

                plot_kwargs = {'color': this_color,
                               'linestyle': this_linestyle,
                               'linewidth': this_linewidth,
                               'alpha': this_opacity,
                               'label': final_label,
                               'zorder': this_zorder}

                if len(decade_batches) == 1:
                    this_ufs_series, = axs.plot(these_x_values, these_y_values, **plot_kwargs)
                else:
                    this_ufs_series, = axs[this_decade_index].plot(these_x_values, these_y_values, **plot_kwargs)

                this_ufs_series.set_solid_capstyle('round')
                this_ufs_series.set_solid_joinstyle('round')

        # Plot verif data
        # Get the few months leading up to this forecast.
        # Additionally:
        # Get 1 extra month ahead because the data point represents the entire next month.
        # and take off the final time step that would be the first date final month.
        # future_leads = leads + [max(leads) + 1]
        future_leads = list(range(12))
        past_leads = list(range(12))  # Want VERIF series to stretch across entire panel.

        # Foreward projection
        verif_forecast_times = [
            timeutil.time_offset(freq_unit, inits[-1], lead, step, 'forward')
            for lead in future_leads
        ]

        # Backward projection
        verif_hindcast_times = [
            timeutil.time_offset(freq_unit, inits[0], lead, step, 'backward')
            for lead in past_leads
        ]

        all_times = sorted(list(set(verif_hindcast_times + verif_forecast_times)))
        # Copy the index dataset into a new object to prevent reference interference.
        verif_index = copy.deepcopy(verif_stats['monthly_mean'].sel(time=slice(all_times[0], all_times[-1])))

        # Calculate nino3.4 area average and monthly average
        x_values = verif_index.time.values

        if calc_anomaly is False:
            y_values = verif_index.values

        elif calc_anomaly is True:
            y_values = [verif_stats['anomaly'].sel(time=this_time)['anomaly'].values.item() for this_time in x_values]

        min_y_value = np.nanmin([np.nanmin(y_values), min_y_value])
        max_y_value = np.nanmax([np.nanmax(y_values), max_y_value])

        if first_iteration is not True:
            verif_label = None

        # Plot verif
        verif_properties = {
            'color': '#ee6677',
            'alpha': 0.85,
            'linestyle': 'solid',
            'linewidth': 1.75,
            'label': verif_label,
            'zorder': 3}

        if len(decade_batches) == 1:
            this_verif_series, = axs.plot(x_values, y_values, **verif_properties)
        else:
            this_verif_series, = axs[this_decade_index].plot(x_values, y_values, **verif_properties)

        this_verif_series.set_solid_capstyle('round')
        this_verif_series.set_solid_joinstyle('round')

        return min_y_value, max_y_value

    # Add plot series decade at a time, for every year and every init+leads in that year.
    min_y_values = []
    max_y_values = []
    for i in range(len(decade_batches)):

        # Add series to plot
        this_min_y_value, this_max_y_value = add_to_plot(i)

        # Keep track of minimum and maximum y values (needed later)
        min_y_values.append(this_min_y_value)
        max_y_values.append(this_max_y_value)

    # Determine y-axis ranges.
    min_y_value = np.nanmin(min_y_values)
    max_y_value = np.nanmax(max_y_values)

    # Add a cushion
    difference = max_y_value - min_y_value
    cushion = difference / 10.0
    min_y_value -= cushion
    max_y_value += cushion

    # Max lead is used to set the rightmost ylim for each panel
    max_lead = np.nanmax(ufs_stats['monthly_mean'].lead.values)

    # Calcluate the axis ranges for each panel
    # It should be first day of decade to finalday of decade + max_lead[months]
    xlimits = []
    for this_decade in decade_batches:

        this_left_year = this_decade[0] - (this_decade[0] % 10)
        this_right_year = this_left_year + 10

        this_left_date = np.datetime64(f'{this_left_year}-01-01')
        this_right_date = np.datetime64(f'{this_right_year}-{str(max_lead).zfill(2)}-01')

        # Append results
        xlimits.append((this_left_date, this_right_date))

    if len(decade_batches) == 1:

        axs.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.5),
            ncol=2, fancybox=True, shadow=False
        )

        axs.set_ylim(min_y_value, max_y_value)
        axs.margins(x=0)

        axs.set_xlim(left=xlimits[0][0], right=xlimits[0][1])
        axs.grid(axis='x', which='major', linestyle='--', dashes=(5, 10), linewidth=0.5, color='lightgray')
        axs.set_axisbelow(True)

    else:
        axs[0].set_title(f"{title}")

        for i in range(len(axs)):
            axs[i].set_ylim(min_y_value, max_y_value)
            axs[i].set_xlim(xlimits[i][0], right=xlimits[i][1])
            axs[i].grid(axis='x', which='major', linestyle='--', dashes=(5, 10), linewidth=0.5, color='lightgray')
            axs[i].set_axisbelow(True)

        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=2, fancybox=True, shadow=False)

        axs[0].margins(x=0)

    return plt


def calc_rmse_spread(ufs_ds: xr.Dataset,
                     ufs_var: str,
                     ufs_stats: dict,
                     verif_ds: xr.Dataset,
                     verif_var: str,
                     verif_stats: dict) -> dict:

    # Get lead resolution of UFS
    freq_unit, step = timeutil.get_lead_resolution(ufs_ds)

    # Compute sample size as (M x N)
    members = sorted(list(ufs_ds.member.values))
    members = [mem for mem in members if mem != -1]  # -1 refers to the ensemble mean in our case.

    n_Members = len(members)
    years = list(ufs_ds.groupby('init.year').groups.keys())
    N_years = len(years)
    rmse_coef = 1 / (n_Members * N_years)
    # spread_coef = (n_Members * (n_Members - 1)) / (2 * N_years)
    # The number of degrees of freedom for spread are given by the binom(n_Members, 2)
    # This equation counts every unique (ignoring order) combo of members (which all represent a unique distance).
    # It does not include like-number combinations like (1,1), (2,2), etc... because spread is not computing here.
    # spread_members_dof = math.factorial(n_Members) / (math.factorial(2) * math.factorial(n_Members-2))
    spread_members_dof = (n_Members * (n_Members - 1)) / 2

    spread_coef = 1 / (spread_members_dof * N_years)

    print(f'Number of ensemble members: {n_Members}')
    print(f'Number of years: {N_years}')

    # Get initmonth values.
    initmonths = [label for label, _ in ufs_ds.groupby('init.month')]

    def rmse_spread_resid(year, initmonth, lead, member, next_member):
        '''Calculate the RMSE residual squared'''

        calc_rmse = True
        calc_spread = True

        if member == next_member:
            calc_spread = False
        else:
            calc_rmse = False

        ufs_mask = (ufs_stats['monthly_mean']['init.year'] == year)\
            & (ufs_stats['monthly_mean']['init.month'] == initmonth)

        # Get forecast indeX value
        this_ufs_index = ufs_stats['monthly_mean'].sel(lead=lead, member=member).where(ufs_mask, drop=True).values

        # This may happen if a forecast year only has 1 init.  It's okay.
        if len(this_ufs_index) == 0:
            return 0, 0
        elif len(this_ufs_index) > 1:
            msg = f'year {year} initmonth {initmonth} lead {lead} member {member}'
            raise ValueError(f'Why does this_ufs_index have more than one value? {this_ufs_index} {msg}')

        this_ufs_index = this_ufs_index[0]  # It should be a length-1 array.

        # Get this UFS climatology (based on the Ensemble mean)
        this_ufs_climatology = ufs_stats['climatology_mean'].sel(member=-1, month=initmonth, lead=lead).values.item()

        # Calculate this UFS anomaly
        this_ufs_index -= this_ufs_climatology

        # Calculate rmse (if member combo is actually same-member)
        this_rmse_resid_squared = 0
        if calc_rmse is True:
            # Calculate forecast month
            forecast_time = timeutil.time_offset(freq_unit,
                                                 np.datetime64(f'{year}-{str(initmonth).zfill(2)}-01'),
                                                 lead,
                                                 step,
                                                 'forward')
            # Get the month for this real time
            forecast_month = forecast_time.astype('datetime64[M]').astype(int)
            forecast_month = (forecast_month % 12) + 1

            # Get Observed value
            this_observed_index = verif_stats['monthly_mean'].sel(time=forecast_time).values.item()

            # Get this Era5 climatology
            this_verif_climatology = verif_stats['climatology_mean'].sel(month=forecast_month).values

            # Calculate this verif anomaly
            this_observed_index -= this_verif_climatology

            # Calculate residual
            this_rmse_resid = this_ufs_index - this_observed_index

            # Square the residual
            this_rmse_resid_squared = this_rmse_resid**2

        # Calculate spread (if dealing with 2 different members here)
        this_spread_resid_squared = 0
        if calc_spread is True:

            # Get next member's index and anomaly
            next_member_ufs_index = ufs_stats['monthly_mean'].sel(lead=lead, member=next_member)
            next_member_ufs_index = next_member_ufs_index.where(ufs_mask, drop=True).values[0]
            next_member_ufs_index -= this_ufs_climatology

            # Calculate distance and square the result
            this_spread_resid = this_ufs_index - next_member_ufs_index
            this_spread_resid_squared = this_spread_resid**2

        # Return rmse, spread
        return this_rmse_resid_squared, this_spread_resid_squared

    # Get lead values.
    leads = list(ufs_ds.lead.values)

    # These are the results, one per statistics per init per lead.
    # (lead dictionary is created inside the loop)
    results = dict.fromkeys(['rmse', 'spread'])
    for this_stat in results:
        results[this_stat] = dict.fromkeys(initmonths)

    print('Accumulating Statistics...')
    # Loop over experiment init month.
    for this_initmonth in initmonths:

        # These are the results for this particular init month
        rmse_lead_results = dict.fromkeys(leads)
        spread_lead_results = dict.fromkeys(leads)

        for this_lead in leads:
            # Begin SUM
            # rmse_sum_total = 0
            # spread_sum_total = 0

            # This is a set if unique member combinations as tuples.
            all_member_combinations = set([tuple(sorted(pair)) for pair in itertools.product(members, members)])
            # Convert all member combinations to a dictionary structured like {(1, 4): resid, (1, 5): resid, ...}
            all_member_combinations = dict.fromkeys(all_member_combinations, 0)

            # Statistics
            rmse_sum_total = 0
            spread_sum_total = 0

            for member_combo in all_member_combinations:

                this_member = member_combo[0]
                next_member = member_combo[1]

                # n_years_processed = 0 # Keep track of this for debugging purposes.
                for this_year in years:

                    # run RMSE_SPREAD_RESID() function and ADD RESULT to SUM
                    rmse_resid_squared, spread_resid_squared =\
                        rmse_spread_resid(this_year, this_initmonth, this_lead, this_member, next_member)

                    # Accumulate
                    rmse_sum_total += rmse_resid_squared
                    spread_sum_total += spread_resid_squared

            # Done with Accumulation for this lead.
            # Divide by degrees of freedom
            rmse_sum_total *= rmse_coef
            spread_sum_total *= spread_coef

            # Take square root
            rmse_sum_total_sqrt = math.sqrt(rmse_sum_total)
            spread_sum_total_sqrt = math.sqrt(spread_sum_total)

            # Append to list of results
            rmse_lead_results[this_lead] = rmse_sum_total_sqrt
            spread_lead_results[this_lead] = spread_sum_total_sqrt

        # Add this lead's results to the final results dictionary
        results['rmse'][this_initmonth] = rmse_lead_results
        results['spread'][this_initmonth] = spread_lead_results

    print('Finished.')
    return results


def plot_rmse_spread(rmses: dict,
                     ufs_experiments: list[str],
                     rmse_only: bool = False,
                     spread_only: bool = False,
                     verif_stats: dict = None,
                     title='',
                     dpi=300):
    '''
    rmses is a list of dictionaries.
    Each dict has 1 key per init month,
    and each value is a dictionary with 1 key per lead.
    '''

    fig, ax = plt.subplots(figsize=(14, 7), dpi=dpi)

    # This works as long as there aren't > 5 UFS models to plot.
    # colors = ['#EE6677', '#4477AA', '#228833', '#CCBB44', '#AA3377', '#66CCEE']  # Paul Tol Bright
    colors = ['#4477AA', '#228833', '#CCBB44', '#66CCEE', '#AA3377']  # Paul Tol Bright

    initmonths = sorted(list(rmses[0]['rmse'].keys()))

    for i in range(len(rmses)):

        this_rmse = rmses[i]  # Process this UFS model's statistics

        for k in range(len(initmonths)):

            this_initmonth = initmonths[k]

            # Plot rmse
            if spread_only is not True:
                x_values = sorted(list(this_rmse['rmse'][this_initmonth].keys()))
                y_values = [this_rmse['rmse'][this_initmonth][this_lead] for this_lead in x_values]

                # Convert x_values to their actual month number.
                x_values = [this_x + this_initmonth for this_x in x_values]

                this_label = None
                if k == 0:
                    this_label = f'RMSE {ufs_experiments[i]}'

                line_rmse, = ax.plot(x_values, y_values,
                                     color=colors[i],
                                     linestyle='solid',
                                     linewidth=2.5,
                                     alpha=0.9,
                                     label=this_label,
                                     zorder=0)

                line_rmse.set_solid_capstyle('round')
                line_rmse.set_solid_joinstyle('round')

            # Plot spread
            if rmse_only is not True:
                x_values = sorted(list(this_rmse['spread'][this_initmonth].keys()))
                y_values = [this_rmse['spread'][this_initmonth][this_lead] for this_lead in x_values]

                # Convert x_values to their actual month number.
                x_values = [this_x + this_initmonth for this_x in x_values]

                this_label = None
                if k == 0:
                    this_label = f'SPREAD {ufs_experiments[i]}'

                line_spread, = ax.plot(x_values, y_values,
                                       color=colors[i],
                                       linestyle='dashed',
                                       dashes=(5, 5),
                                       linewidth=1.2,
                                       alpha=0.75,
                                       label=this_label,
                                       zorder=100)

                line_spread.set_dash_capstyle('round')
                line_spread.set_dash_joinstyle('round')

    # Calculate x-axis
    max_init = max([max(rmse['rmse'].keys()) for rmse in rmses])
    max_lead = max([max(rmse['rmse'][max_init].keys()) for rmse in rmses])

    max_x_value = max_init + max_lead
    min_x_value = min(initmonths)

    # Plot verif saturation
    if verif_stats is not None:

        # These are the axis values (min_init to max_init + leads)
        x_values = list(range(min_x_value, max_x_value + 1))

        # Do this because our numerical range of months for plotting may extend beyond 12.
        real_month_values = [(x - 1) % 12 + 1 for x in x_values]

        y_values = [verif_stats['climatology_std'].sel(month=m).values.item()
                    for m in real_month_values]

        y_values = np.sqrt(2) * np.array(y_values)

        saturation, = ax.plot(x_values, y_values,
                              color='grey',
                              linestyle='solid',
                              linewidth=0.5,
                              zorder=-999,
                              label='Saturation')

        saturation.set_solid_capstyle('round')
        saturation.set_solid_joinstyle('round')

    # Numeric attributes of x-axis
    ax.set_xlim(min_x_value, max_x_value)
    ax.set_xticks(list(range(min_x_value, max_x_value + 1)))

    month_labels = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # Cycle through month labels if we go beyond 12
    xticks = [month_labels[i % len(month_labels)] for i in range(min_x_value - 1, max_x_value)]

    ax.set_xticklabels(xticks)

    ax.set_ylim(bottom=0)
    ax.set_xlabel('Lead time')

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.30),
        ncol=5, fancybox=True, shadow=False
    )

    ax.set_title(f"{title}")

    return plt
