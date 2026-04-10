from typing import Tuple, Union
import operator
import numpy as np
import pandas as pd
import xarray as xr

# from ..datareader import DataReader_Super, UFS_DataReader


def time_offset(freq_unit: str, init: str, lead: int, step: np.timedelta64, direction='forward') -> np.timedelta64:

    if direction == 'backward':
        direction_op = operator.sub

    elif direction == 'forward':
        direction_op = operator.add
    else:
        raise ValueError(f'direction must be forward or backward.')

    # Forecast time
    if freq_unit == 'MS':
        forecast_time = np.datetime64(direction_op(pd.Timestamp(init), pd.DateOffset(months=int(lead))))

    elif freq_unit == 'D':
        forecast_time = direction_op(np.datetime64(init),
                                     np.timedelta64(int(lead * step.astype('timedelta64[D]').astype(int)), 'D'))

    elif freq_unit == 'H':
        forecast_time = direction_op(np.datetime64(init),
                                     np.timedelta64(int(lead * step.astype('timedelta64[h]').astype(int)), 'h'))

    return forecast_time


def get_lead_resolution(ds: Union[xr.Dataset, xr.DataArray]) -> Tuple[str, np.timedelta64]:

    if 'lead' not in ds.dims:
        return (None, None)

    lead = ds['lead'].values.astype(int)
    lead_unit = ds['lead'].attrs.get("units", "hours")

    # print(f"\nDEBUG: Analyzing lead time units")
    # print(f"       Raw lead values: {self.model_ds['lead'].values[:5]}...")
    # print(f"       Metadata units: '{lead_unit}'")

    if lead_unit == 'months':
        return 'MS', np.timedelta64(30, 'D')

    elif lead_unit == 'days':
        return 'D', np.timedelta64(lead[1] - lead[0], 'D')

    elif lead_unit == 'hours':
        return 'H', np.timedelta64(lead[1] - lead[0], 'h')

    else:
        raise ValueError(f"Unsupported lead unit '{lead_unit}'")


def datetime_batcher(all_times: list[np.datetime64]) -> list[tuple]:
    '''
    Divide a list of np.datetime64 objects into monthly chunks, i.e.
    [(month_start, month_end), (next_month_start, next_month_end), ...]
    '''
    if len(all_times) <= 1:
        msg = f'datetime_batcher() needs more than one datetime to work with. Got length {len(all_times)}'
        raise ValueError(msg)

    start_time = None
    batches = []

    for i in range(len(all_times)):

        this_time = all_times[i]
        this_month = pd.to_datetime(this_time).month

        # If this is the FINAL datetime in the list, then our work is done.
        if i == len(all_times) - 1:
            batches.append((start_time, this_time))
            break

        if i == 0:
            start_time = this_time

        next_time = all_times[i + 1]
        next_month = pd.to_datetime(next_time).month

        if next_month != this_month:
            batches.append((start_time, this_time))
            start_time = next_time  # reset start_time

    return batches


def decade_batcher(year_list: list[int]) -> list[list]:

    '''Batch a list of years into sublists of years in the same decade.'''

    # First sort in ascending order
    year_list = sorted(year_list)

    # Determine which years are start-of-decade
    all_mods = np.array([this_year % 10 for this_year in year_list])

    # The start-of-decade years appear at these indices
    indices = list(np.where(all_mods == 0)[0])
    indices.append(len(year_list))
    indices.insert(0, 0)
    indices = list(set(indices))

    batches = []
    for i in range(len(indices)):
        if i == len(indices) - 1:
            continue

        this_index = indices[i]
        next_index = indices[i + 1]
        batches.append(year_list[this_index:next_index])

    return batches


def match_time_to_leads(verif_ds: xr.Dataset, ufs_ds: xr.Dataset) -> xr.Dataset:
    '''Temporally subset verification data (time) to exactly match UFS (init+lead)
    '''

    if 'init' not in ufs_ds.dims:
        raise ValueError(f'ufs_ds must have init+lead dimensions, got {ufs_ds.dims}')

    # The whole point of this function is to match time -> init+lead
    if 'init' in verif_ds.dims:
        inits = list(ufs_ds.init.values)
        leads = list(ufs_ds.lead.values)
        return verif_ds.sel(init=inits, lead=leads)

    # Get the final lead of the final init.
    # Add 1 because we want to grab all verif data for the final lead.
    # e.g. Montly data with init 2020-01-01 and +3 lead times.
    # The final lead time would thus be 2020-04-01.
    # To grab all verif data, then we need the entire month of 04.
    # Extend the final lead time to 2020-05-01 and treat it as an exclusive bound.
    # (TODO: What if we're dealing with 2 ufs dataset?)
    first_init = sorted(ufs_ds.init.values)[0]
    final_init = sorted(ufs_ds.init.values)[-1]

    # The final lead (int) from the final init
    final_lead = max(ufs_ds.sel(init=final_init).lead.values) + 1

    # Get temporal resolution of UFS data
    freq_unit, step = get_lead_resolution(ufs_ds)

    # Convert the final lead into a datetime based on our knowledge of UFS temporal resolution.
    final_time = time_offset(freq_unit, final_init, final_lead, step)

    # Subset verif data based on these times.
    verif_ds = verif_ds.sel(time=slice(str(first_init), str(final_time)))

    # Now, all we need to do is exclude the final time.
    all_times = list(verif_ds.time.values)

    matches = list(all_times >= final_time)
    final_index = matches.index(True) if True in matches else None

    # Do the exlusive time slice
    verif_ds = verif_ds.isel(time=slice(0, final_index))

    return verif_ds
