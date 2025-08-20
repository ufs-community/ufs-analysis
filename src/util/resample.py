import pandas as pd

def resample(ds, timeslice, freq):

    return ds.sel(time=timeslice).resample(time=freq).mean().sortby('time').compute()

def datetime_batcher(all_times):
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
        if i == len(all_times)-1:
            batches.append((start_time, this_time))
            break

        if i == 0:
            start_time = this_time
            
        next_time = all_times[i+1]
        next_month = pd.to_datetime(next_time).month

        if next_month != this_month:
            batches.append((start_time, this_time))
            start_time = next_time  # reset start_time

    return batches

