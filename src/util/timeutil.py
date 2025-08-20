import numpy as np
import pandas as pd

def time_offset(freq_unit, init, lead, step):

    if freq_unit == 'MS':
        forecast_time = np.datetime64(pd.Timestamp(init) + pd.DateOffset(months=int(lead)))

    elif freq_unit == 'D':
        forecast_time = np.datetime64(init) + np.timedelta64(int(lead * step.astype('timedelta64[D]').astype(int)), 'D')

    elif freq_unit == 'H':
        forecast_time = np.datetime64(init) + np.timedelta64(int(lead * step.astype('timedelta64[h]').astype(int)), 'h')

    return forecast_time
