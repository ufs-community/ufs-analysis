# ---------------------------------------------------------------------------------------------------------------------
#  Filename: nao.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Calculate positive and negative NAO phases.
# ---------------------------------------------------------------------------------------------------------------------

from typing import Optional, Union, Tuple, List
import xarray as xr
import pandas as pd

from . import stats, timeutil


class NAO:

    # For NAO, there are 2 reference locations:
    # (This naming convention is used throughout the repository, so for consistency it is also used here.)
    REGION_1 = {'latmin': 65.0, 'lonmin': 331.2}
    REGION_2 = {'latmin': 37.7, 'lonmin': 334.3}

    def __init__(self, ds: xr.Dataset):

        if 'init' in ds.dims and 'lead' in ds.dims:
            self.data_type = 'model'
            self.negative_exclude_initleads = []
            self.positive_exclude_initleads = []

        elif 'time' in ds.dims:
            self.data_type = 'verif'
            self.negative_exclude_months = []
            self.positive_exclude_months = []
        else:
            raise ValueError(f'da must have either time or init+lead dimensions, got {da.dims}')

        self.ds = ds

    def calc_phases(self, var):
        if self.data_type == 'verif':
            return self._calc_phases_verif(var)

        elif self.data_type == 'model':
            return self._calc_phases_model(var)

    def _calc_phases_model(self, var):

        ds_1 = self.ds[[var]].sel(lat=self.REGION_1['latmin'], lon=self.REGION_1['lonmin'], method='nearest').load()
        ds_2 = self.ds[[var]].sel(lat=self.REGION_2['latmin'], lon=self.REGION_2['lonmin'], method='nearest').load()

        stats_1 = stats.calc_climatology_anomaly(ds_1, area_mean=False, use_member_climatology=False)
        stats_2 = stats.calc_climatology_anomaly(ds_2, area_mean=False, use_member_climatology=False)

        da_1 = stats.normalize(da=stats_1[var], stats=stats_1)
        da_2 = stats.normalize(da=stats_2[var], stats=stats_2)

        nao_da = (da_2 - da_1)

        # This list shows when monthly NAO is positive or negative.
        # It is a list of np.datetime64 objects, like:
        # [(numpy.datetime64('1994-05-01T00'), ...]
        # This is an *exclusionary* list due to downstream analysis codes; easier this way.

        # This is when ERA5 NAO is positive or negative
        positive_exclude_initleads = []
        negative_exclude_initleads = []

        for this_init in nao_da.init.values:
            for this_lead in nao_da.lead.values:

                # This NAO value
                this_nao_value = nao_da.sel(init=this_init, lead=this_lead).values.item()
                # Is this NAO value non-positive or non-negative?
                if this_nao_value >= 0:
                    negative_exclude_initleads.append((this_init, this_lead))

                elif this_nao_value <= 0:
                    positive_exclude_initleads.append((this_init, this_lead))

        # Assign results to self
        self.negative_exclude_initleads = negative_exclude_initleads
        self.positive_exclude_initleads = positive_exclude_initleads

        print('Results stored in:')
        print('<self>.positive_exclude_initleads')
        print('and')
        print('<self>.negative_exclude_initleads')

    def _calc_phases_verif(self, var):

        ds_1 = self.ds[[var]].sel(lat=self.REGION_1['latmin'], lon=self.REGION_1['lonmin'], method='nearest').load()
        ds_2 = self.ds[[var]].sel(lat=self.REGION_2['latmin'], lon=self.REGION_2['lonmin'], method='nearest').load()

        stats_1 = stats.calc_climatology_anomaly(ds_1, area_mean=False)
        stats_2 = stats.calc_climatology_anomaly(ds_2, area_mean=False)

        da_1 = stats.normalize(da=stats_1['monthly_mean'], stats=stats_1)
        da_2 = stats.normalize(da=stats_2['monthly_mean'], stats=stats_2)

        nao_da = (da_2 - da_1)

        # This list shows when monthly NAO is positive or negative.
        # It is a list of np.datetime64 objects, like:
        # [(numpy.datetime64('1994-05-01T00'), ...]
        # This is an *exclusionary* list due to downstream analysis codes; easier this way.
        positive_exclude_months = []
        negative_exclude_months = []

        for this_time in nao_da.time.values:

            # This NAO value
            this_nao_value = nao_da.sel(time=this_time).values.item()

            # Is this NAO value non-positive or non-negative?
            if this_nao_value >= 0:
                negative_exclude_months.append(this_time)

            elif this_nao_value <= 0:
                positive_exclude_months.append(this_time)

        # Assign results to self
        self.verif_negative_exclude_months = negative_exclude_months
        self.verif_positive_exclude_months = positive_exclude_months

        # Reset to null
        self.negative_exclude_initleads = []
        self.positive_exclude_initleads = []

        print('Results stored in:')
        print('<self>.positive_exclude_months')
        print('and')
        print('<self>.negative_exclude_months')







































