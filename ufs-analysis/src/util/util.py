# ---------------------------------------------------------------------------------------------------------------------
#  Filename: util.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Define assorted utilitarian functions.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
from ..datareader import datareader as dr


def retrieve_ufs_dataset(ufs_data_reader,
                         ufs_var: str | list[str],
                         time_range: tuple[str, str],
                         members: list[int],
                         region: dict = None,
                         **kwargs) -> xr.Dataset:
    '''
    This function is a wrapper around UFS_DataReader.retrieve()
    Its main purpose is to return member data alongside the ensemble average (for all members).
    (Normally, we get one or the other, not both together.)

    members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'ens_avg']
    region = {
        'latmin': -5.0,
        'latmax': 5.0,
        'lonmin': 190.0,
        'lonmax':240.0
    }
    '''

    if region is None:
        region = {'latmin': -90, 'latmax': 90, 'lonmin': 0, 'lonmax': 360}

    datasets = []
    for i in range(len(members)):

        this_member = members[i]
        print(f'retrieving {this_member}')

        if this_member == 'ens_avg':
            this_ds = ufs_data_reader.retrieve(
                var=ufs_var,
                lat=(region['latmin'], region['latmax']),
                lon=(region['lonmin'], region['lonmax']),
                time=time_range,
                ens_avg=True,
                **kwargs
            )

            this_ds = this_ds.expand_dims('member')
            this_ds = this_ds.assign_coords({'member': [-1]})
            datasets.append(this_ds)

        else:
            this_ds = ufs_data_reader.retrieve(
                var=ufs_var,
                lat=(region['latmin'], region['latmax']),
                lon=(region['lonmin'], region['lonmax']),
                time=time_range,
                member=this_member,
                **kwargs
            )

            datasets.append(this_ds)

    # End Loop
    if len(datasets) == 0:
        ds = None
    elif len(datasets) == 1:
        ds = this_ds
    else:
        ds = xr.concat(datasets, dim='member', coords='minimal', compat='equals')

    return ds


def combine_ufs_means(ufs_experiments_list: list[str],
                      ufs_vars_list: list[str],
                      time_range: tuple[str, str],
                      region: dict = None,
                      **kwargs) -> xr.Dataset:
    '''
    Merge the ensemble means of multiple UFS models together into 1 dataset
    In the merged dataset, each model will have a 'member' coordinate
    '''
    data_reader_list = []
    for this_experiment in ufs_experiments_list:
        # this_filename = f"experiments/phase_1/{this_experiment}/atm_monthly.zarr"

        this_data_reader = dr.getDataReader(datasource='UFS',
                                            experiment=this_experiment,
                                            # filename=this_filename,
                                            model='atm')

        data_reader_list.append(this_data_reader)

    members = ['ens_avg']
    ds_list = []

    for i in range(len(data_reader_list)):

        this_dr = data_reader_list[i]
        this_member = ufs_experiments_list[i]
        ufs_var = None

        for this_var in ufs_vars_list:
            if this_var in list(this_dr.dataset().keys()):
                ufs_var = this_var

        if ufs_var is None:
            raise ValueError(f"Couldn't find variable in ufs dataset: {ufs_vars_list}")

        # Get the dataset
        this_ds = retrieve_ufs_dataset(this_dr, ufs_var, time_range, members, region=region, **kwargs)
        this_ds = this_ds.assign_coords(member=('member', [this_member]))

        this_ds = this_ds.rename_vars({ufs_var: ufs_vars_list[0]})

        ds_list.append(this_ds)

    ds = xr.concat(ds_list, dim='member', coords='minimal', compat='equals')

    return ds


def print_fixed_width(list_of_strings: list[str]) -> str:

    '''return a printable fixed-width table of 2 or 3 columns from a list of strings'''

    max_string_len = len(max(list_of_strings, key=len))

    # Guesstimate how much space columns will need. Goal is to minimize width.
    if max_string_len > 30:
        n_cols = 2
    elif max_string_len > 20:
        n_cols = 3
    else:
        n_cols = 4

    # Organize list elements alphabetically down the columns.
    list_of_strings = sorted(list_of_strings)

    # Determine how many rows are needed
    n_rows = int(np.ceil(len(list_of_strings) / n_cols))

    # Determine remainder after inserting data into columns
    remainder = (n_rows * n_cols) - len(list_of_strings)

    # Add blank list elements to completely fill columns
    list_of_strings.extend(['' for i in range(remainder)])

    # A list of lists, where each list contains the values for a column
    cols = [list_of_strings[(colnum * n_rows):(colnum * n_rows) + n_rows] for colnum in range(n_cols)]

    # Pandas to_string() method always justifies values to the right
    # (the 'justify' parameter is for headers only)
    # Left-justification is easier to read, but must be done manually.
    # Insert spaces to the right of each value to fill out the column's width.
    for i in range(len(cols)):
        col_list = cols[i]
        max_length = 0

        # Find the max string length in this column
        for this_value in col_list:
            max_length = max(max_length, len(this_value))

        # Iterate again and add blanks to reach the width
        new_values = []
        for this_value in col_list:
            n_spaces_to_add = max_length - len(this_value)
            spaces = ' ' * n_spaces_to_add
            new_values.append(this_value + spaces)

        cols[i] = new_values

    # Construct dataframe
    df = pd.DataFrame()
    for colnum in range(len(cols)):
        this_col = cols[colnum]
        df[f'{colnum}'] = this_col

    # Make all column headers blank, for printing to console.
    df.columns = [''] * n_cols

    return df.to_string(index=False)
