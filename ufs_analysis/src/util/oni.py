# ---------------------------------------------------------------------------------------------------------------------
#  Filename: oni.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Define a class that organizes information about Oceanic Niño Index events.
# ---------------------------------------------------------------------------------------------------------------------

import os
import sys
import warnings
from typing import Optional, Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from ..regridder import Regrid
from ..datareader import datareader as dr
from ..datareader import DataReader_Super
from . import stats, rws, timeutil, cmaps


# Year and highest ONI recorded *in its strength category*
elnino_events = (
    (1951, 1.2),
    (1952, 0.8),
    (1953, 0.8),
    (1957, 1.8),
    (1958, 0.6),
    (1963, 1.4),
    (1965, 1.9),
    (1968, 1.1),
    (1969, 0.9),
    (1972, 1.8),
    (1976, 0.9),
    (1977, 0.8),
    (1979, 0.6),
    (1982, 2.2),
    (1986, 1.2),
    (1987, 1.7),
    (1991, 1.7),
    (1994, 1.1),
    (1997, 2.4),
    (2002, 1.3),
    (2004, 0.7),
    (2006, 0.94),
    (2009, 1.36),
    (2014, 0.93),
    (2015, 2.64),
    (2018, 0.90),
    (2023, 1.95)
)

# Year and highest ONI recorded *in its strength category*
lanina_events = (
    (1954, -0.9),
    (1955, -1.4),
    (1964, -0.8),
    (1970, -1.4),
    (1971, -0.9),
    (1973, -1.9),
    (1974, -0.8),
    (1975, -1.7),
    (1983, -0.9),
    (1984, -0.9),
    (1988, -1.8),
    (1995, -1.0),
    (1998, -1.6),
    (1999, -1.7),
    (2000, -0.7),
    (2005, -0.85),
    (2007, -1.64),
    (2008, -0.85),
    (2010, -1.64),
    (2011, -1.09),
    (2016, -0.69),
    (2017, -0.97),
    (2020, -1.27),
    (2021, -1.06),  # We sure about this one?
    (2022, -0.99)
)


class ONI:
    '''
    Oceanic Niño Index
    Weak:        0.5 to 0.9 SST anomaly
    Moderate:    1.0 to 1.4 SST anomaly
    Strong:      1.5 to 1.9 SST anomaly
    Very Strong: ≥ 2.0      SST anomaly
    '''

    def __init__(self, year: int, oni: float):

        if not isinstance(year, int):
            raise ValueError(f'Wrong data type year=integer')

        if year <= 1900 or year >= 2100:
            raise ValueError(f"year value ({year}) is outside our range of consideration.")

        if not isinstance(oni, (float, int)):
            raise ValueError(f'oni must be a number.')

        if oni <= -10 or oni >= 10:
            raise ValueError(f"oni value ({oni}) is physically unlikely.")

        self._oni = oni
        self._oni_magnitude = abs(oni)
        self._year = year

        # Default values for normal seasons.
        self._event_code = 0
        self._event = ''
        self._strength_code = 0  # 1=weak, 2=moderate, 3=strong, 4=very strong

        if oni < 0:
            self._event = 'LaNina'
            self._event_code = 1
        elif oni > 0:
            self._event = 'ElNino'
            self._event_code = -1

        # Calculate strength
        if self._event_code != 0:
            if 0.5 < self._oni_magnitude < 1.0:
                self._strength_code = 1

            elif 1.0 <= self._oni_magnitude < 1.5:
                self._strength_code = 2

            elif 1.5 <= self._oni_magnitude < 2.0:
                self._strength_code = 3

            elif self._oni_magnitude >= 2.0:
                self._strength_code = 4

        # Readable label
        self._strength = [None, 'Weak', 'Moderate', 'Strong', 'Very Strong'][self._strength_code]

    def __repr__(self):

        msg = f'Oceanic Niño Index object\n'
        msg += f'Event:    {self._event}\n'
        msg += f'Year:     {self._year}\n'
        msg += f'ONI:      {self._oni}\n'
        msg += f'Strength: {self._strength}\n'
        msg += f'Get characteristics of this ONI object with: <your_oni_object>.get(<attribute_name>)\n'

        return msg

    def get(self, att: str):
        '''Every attribute must be lower case.'''

        att = att.lower()

        try:
            return getattr(self, att)
        except AttributeError:
            att = f'_{att}'
            return getattr(self, att)


def prep_oni_datasets(data_reader1: DataReader_Super.DataReader,
                      var1: Union[str, List[str]],
                      data_reader2: DataReader_Super.DataReader,
                      var2: Union[str, List[str]],
                      statistics: Union[str, List[str]],
                      elnino_years: list,
                      lanina_years: list) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:

    '''Prepare datasets used for enso-teleconnections diagnostics.'''

    # First, process statistics argument
    available_statistics = ['anomaly',
                            'restoring effect',
                            'stationary wave number',
                            'rossby wave source']

    # Check data types:
    # Input variables.
    if isinstance(var1, str):
        var1 = [var1]
    elif not isinstance(var1, list):
        msg = "var1 must be a string (1 variable) or list of strings (2 variables)."
        raise ValueError(msg)

    if isinstance(var2, str):
        var2 = [var2]
    elif not isinstance(var2, list):
        msg = "var2 must be a string (1 variable) or list of strings (2 variables)."
        raise ValueError(msg)

    # Statistics requested.
    if isinstance(statistics, str):
        statistics = [statistics]

    if not isinstance(statistics, list):
        msg = f"statistics must be a string value or a list of string values in {available_statistics}"
        raise ValueError(msg)

    if len(statistics) == 0:
        raise ValueError(f'Must enter statistics= one or more of {available_statistics}')

    # Coerce every statistic name to lower case
    statistics = [stat.lower() for stat in statistics]

    # if statistic not in available_statistics:
    not_available = set(statistics).difference(available_statistics)
    if len(not_available) > 0:
        msg = f"{not_available} is not available. statistics must be one or more of {available_statistics}'"
        raise ValueError(msg)

    # Get types of datareaders (turn something like "src.datareader.UFS_DataReader.UFS_DataReader" into "UFS")
    type1 = str(type(data_reader1)).split('.')[-1].split('_')[0]
    type2 = str(type(data_reader2)).split('.')[-1].split('_')[0]

    # Check if a type is UFS. If so, get its releast tag and append to 'type'
    if type1 == 'UFS' and hasattr(data_reader1, 'experiment'):
        type1 += str(data_reader1.experiment)

    if type2 == 'UFS' and hasattr(data_reader2, 'experiment'):
        type2 += str(data_reader2.experiment)

    # Extract Xarray datasets.
    ds1 = data_reader1.dataset()
    ds2 = data_reader2.dataset()

    # Subset Data based on these years
    # UFS
    ds1_elnino_mask = (ds1.init.dt.year.isin(elnino_years))
    ds1_lanina_mask = (ds1.init.dt.year.isin(lanina_years))

    ds1_elnino = ds1.where(ds1_elnino_mask, drop=True)
    ds1_lanina = ds1.where(ds1_lanina_mask, drop=True)

    # Verif (may also be UFS)
    ds2_elnino_mask = (ds2.init.dt.year.isin(elnino_years))
    ds2_lanina_mask = (ds2.init.dt.year.isin(lanina_years))

    ds2_elnino = ds2.where(ds2_elnino_mask, drop=True)
    ds2_lanina = ds2.where(ds2_lanina_mask, drop=True)

    # Confirm that we have perfectly matching forecast times.
    n_ds1_elnino = len(ds1_elnino.init.values) * len(ds1_elnino.lead.values)
    n_ds2_elnino = len(ds2_elnino.init.values) * len(ds2_elnino.lead.values)

    n_ds1_lanina = len(ds1_lanina.init.values) * len(ds1_lanina.lead.values)
    n_ds2_lanina = len(ds2_lanina.init.values) * len(ds2_lanina.lead.values)

    # Check
    if n_ds1_elnino != n_ds2_elnino or n_ds1_lanina != n_ds2_lanina:

        msg = "Something went wrong... VERIF data and ds1 don't have identical time periods."
        raise ValueError(msg)

    if 'restoring effect' in statistics or 'stationary wave number' in statistics:

        print('Calculating restoring effect (Beta star) and stationary wave number (Ks)')

        # We must first check that U_WIND has been specified by the user.
        U_WIND_FOUND = False
        for wind_set in data_reader1.WINDS:

            if var1[0] == wind_set['U_WIND']:
                U_WIND_FOUND = True
                use_this_var1 = var1[0]

            if var1[1] == wind_set['U_WIND']:
                U_WIND_FOUND = True
                use_this_var1 = var1[1]

        if U_WIND_FOUND is False:
            msg = f'restoring effect and/or stationary wave number require U wind component, got {var1}'
            raise ValueError(msg)

        # Do this again for ds2
        U_WIND_FOUND = False
        for wind_set in data_reader2.WINDS:

            if var2[0] == wind_set['U_WIND']:
                U_WIND_FOUND = True
                use_this_var2 = var2[0]

            if var2[1] == wind_set['U_WIND']:
                U_WIND_FOUND = True
                use_this_var2 = var2[1]

        if U_WIND_FOUND is False:
            msg = f'restoring effect and/or stationary wave number require U wind component, got {var2}'
            raise ValueError(msg)

        # Calculate UFS Beta* and Ks
        ds1_elnino = stats.calc_betastar_kwavenumber(ds1_elnino, uvar=use_this_var1)
        ds1_lanina = stats.calc_betastar_kwavenumber(ds1_lanina, uvar=use_this_var1)

        # VERIF VERIF Beta* and Ks
        ds2_elnino = stats.calc_betastar_kwavenumber(ds2_elnino, uvar=use_this_var2)
        ds2_lanina = stats.calc_betastar_kwavenumber(ds2_lanina, uvar=use_this_var2)

    if 'anomaly' in statistics:
        print("Calculating climatology statistics and anomalies.")

        # Compute climatology statistics
        ds1_stats = stats.calc_climatology_anomaly(ds1[[var1[0]]], area_mean=False)
        ds2_stats = stats.calc_climatology_anomaly(ds2[[var2[0]]], area_mean=False)

        # Calculate UFS Anomaly
        ds1_elnino = stats.calc_anomaly(ds=ds1_elnino, var=var1[0], stats=ds1_stats)
        ds1_lanina = stats.calc_anomaly(ds=ds1_lanina, var=var1[0], stats=ds1_stats)

        # Calculate VERIF Anomaly
        ds2_elnino = stats.calc_anomaly(ds=ds2_elnino, var=var2[0], stats=ds2_stats)
        ds2_lanina = stats.calc_anomaly(ds=ds2_lanina, var=var2[0], stats=ds2_stats)

    if 'rossby wave source' in statistics:

        # RWS Components across entire data record.
        print('Calculating Rossby Wave Source (RWS) components.')
        ds1 = rws.calc_rws_components(ds1, var1[0], var1[1])
        ds2 = rws.calc_rws_components(ds2, var2[0], var2[1])

        # -----------
        # STATISTICS
        # -----------
        print('Calculating RWS component climatology statistics and anomalies.')
        # Climatologies
        ds1_absvrt_stats = stats.calc_climatology_anomaly(ds1[['absvrt']], area_mean=False)
        ds1_uchi_stats = stats.calc_climatology_anomaly(ds1[['uchi']], area_mean=False)
        ds1_vchi_stats = stats.calc_climatology_anomaly(ds1[['vchi']], area_mean=False)

        ds2_absvrt_stats = stats.calc_climatology_anomaly(ds2[['absvrt']], area_mean=False)
        ds2_uchi_stats = stats.calc_climatology_anomaly(ds2[['uchi']], area_mean=False)
        ds2_vchi_stats = stats.calc_climatology_anomaly(ds2[['vchi']], area_mean=False)

        # Anomalies
        ds1_absvrt_anomaly = stats.calc_anomaly(ds=ds1, var='absvrt', stats=ds1_absvrt_stats)
        ds1_uchi_anomaly = stats.calc_anomaly(ds=ds1, var='uchi', stats=ds1_uchi_stats)
        ds1_vchi_anomaly = stats.calc_anomaly(ds=ds1, var='vchi', stats=ds1_vchi_stats)

        ds2_absvrt_anomaly = stats.calc_anomaly(ds=ds2, var='absvrt', stats=ds2_absvrt_stats)
        ds2_uchi_anomaly = stats.calc_anomaly(ds=ds2, var='uchi', stats=ds2_uchi_stats)
        ds2_vchi_anomaly = stats.calc_anomaly(ds=ds2, var='vchi', stats=ds2_vchi_stats)

        # ---------------
        # END STATISTICS
        # ---------------

        # Compute RWS
        ds1_elnino = rws.calc_rws(ds1_elnino,
                                  absvrt_stats=ds1_absvrt_stats,  # Absolute Vorticity
                                  absvrt_anomaly=ds1_absvrt_anomaly,
                                  uchi_stats=ds1_uchi_stats,  # UCHI
                                  uchi_anomaly=ds1_uchi_anomaly,
                                  vchi_stats=ds1_vchi_stats,  # VCHI
                                  vchi_anomaly=ds1_vchi_anomaly)

        ds1_lanina = rws.calc_rws(ds1_lanina,
                                  absvrt_stats=ds1_absvrt_stats,  # Absolute Vorticity
                                  absvrt_anomaly=ds1_absvrt_anomaly,
                                  uchi_stats=ds1_uchi_stats,  # UCHI
                                  uchi_anomaly=ds1_uchi_anomaly,
                                  vchi_stats=ds1_vchi_stats,  # VCHI
                                  vchi_anomaly=ds1_vchi_anomaly)

        ds2_elnino = rws.calc_rws(ds2_elnino,
                                  absvrt_stats=ds2_absvrt_stats,  # Absolute Vorticity
                                  absvrt_anomaly=ds2_absvrt_anomaly,
                                  uchi_stats=ds2_uchi_stats,  # UCHI
                                  uchi_anomaly=ds2_uchi_anomaly,
                                  vchi_stats=ds2_vchi_stats,  # VCHI
                                  vchi_anomaly=ds2_vchi_anomaly)

        ds2_lanina = rws.calc_rws(ds2_lanina,
                                  absvrt_stats=ds2_absvrt_stats,  # Absolute Vorticity
                                  absvrt_anomaly=ds2_absvrt_anomaly,
                                  uchi_stats=ds2_uchi_stats,  # UCHI
                                  uchi_anomaly=ds2_uchi_anomaly,
                                  vchi_stats=ds2_vchi_stats,  # VCHI
                                  vchi_anomaly=ds2_vchi_anomaly)

    print("ONI Datasets Ready.")

    return ds1_elnino.load(), \
        ds1_lanina.load(), \
        ds2_elnino.load(), \
        ds2_lanina.load()


def plot_composite(da: xr.DataArray,
                   shading: xr.DataArray = None,
                   shading_threshold: float = 0.05,
                   title: str = '',
                   subtitle: str = '',
                   vmin: float = None,
                   vmax: float = None,
                   cmap: str = 'BuPu',
                   cmap_label: str = None,
                   topleft_label: str = None,
                   bottomright_label: str = None,
                   region: dict = None,
                   dpi=200):
    '''
    Generate shaded contour plot for composite statistics.
    '''

    # Drop lev dimension if it exists. Upstream logic has already confirmed that these data are flat.
    if 'lev' in da.dims:
        da = da.squeeze(dim='lev')

    cmap_center = False
    if vmin is not None and vmax is not None:
        if vmin == -1 * vmax:
            cmap_center = True

    center = 180
    projection = ccrs.PlateCarree(central_longitude=center)

    crs = ccrs.PlateCarree()

    # Instantiate plot
    plt.figure(figsize=(14, 7), dpi=dpi)
    ax = plt.axes(projection=projection)
    # ax.set_global()

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.3,
                      linestyle='--')  # dashes=(5, 1))

    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.top_labels = False
    gl.right_labels = False

    # Remove degree symbol from gridline labels
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')

    cbar_kwargs = {
        'orientation': 'horizontal',
        'shrink': 0.7,
        'pad': 0.05
    }

    # Preserve the name of the cmap input. The following logic may coerce cmap to different variable type.
    cmap_string = cmap

    # Load custom cmaps
    CUSTOM_CMAPS = cmaps.process_cmaps_yaml()

    if cmap in CUSTOM_CMAPS:
        # Adjust n_levels
        n_levels = len(CUSTOM_CMAPS[cmap]) + 1
        # Load up the color map.  This variable is now a matplotlib object.
        # cmap = mcolors.LinearSegmentedColormap.from_list('', CUSTOM_CMAPS[cmap])
        cmap = ListedColormap(CUSTOM_CMAPS[cmap])
    else:
        n_levels = 20

    plot_args = {
        'ax': ax,
        'transform': crs,
        'cmap': cmap,  # cmap could be a string or a ListedColormap, at this point.
        'levels': n_levels,
        'extend': 'neither'  # Disable colorbar pointed extensions
    }

    plot_args['cbar_kwargs'] = cbar_kwargs

    if vmin is not None and vmax is not None:
        plot_args.update({'vmin': vmin, 'vmax': vmax})

    # We will add a label for the min, max, and average values across this field.
    min_value = da.min().values.item()
    avg_value = da.mean().values.item()
    max_value = da.max().values.item()
    std_value = da.std().values.item()

    # Cap values at the color bar range
    # (there is a matplotlib bug where values that deviate greatly from colorbar range show up as white)
    if vmin is not None:
        da = da.clip(min=vmin)

    if vmax is not None:
        da = da.clip(max=vmax)

    # Make plot
    p = da.plot.contourf(**plot_args)

    # Draw contour lines with hardcoded expectations for certain custom cmaps.
    if cmap_string == 'beta_star':
        da_for_lines = (da >= 0).astype(int)
        lines = da_for_lines.plot.contour(ax=ax, transform=crs, colors='black', linewidths=0.5, levels=1)

    if cmap_string in ['Ks', 'Ks_diff']:
        da_for_lines = da.notnull().astype(int)
        lines = da_for_lines.plot.contour(ax=ax, transform=crs, colors='black', linewidths=0.5, levels=1)

    # Center the colormap about 0
    if vmin is not None and vmax is not None:

        ticks = np.linspace(vmin, vmax, n_levels)
        cbar = p.colorbar

        tick_locations = []
        tick_labels = []

        tick_locations.append(vmin)
        tick_labels.append(f'{vmin:.1f}')

        for i in range(n_levels):

            # Skip ends.
            if i == 0 or i == (n_levels - 1):
                continue

            # Display 0 at center.
            if cmap_center is True and i == (n_levels / 2) - 1:
                tick_locations.append(0)
                tick_labels.append('0')
                continue

            # Don't display a value directly adjacent to 0.
            if cmap_center is True and i == (n_levels / 2):
                continue

            if i < (n_levels / 2):
                if i % 2 == 1:
                    continue
                tick_locations.append(ticks[i])
                tick_labels.append(f'{ticks[i]:.1f}')

            elif i > ((n_levels - 1) / 2):
                if i % 2 != 1:
                    continue
                tick_locations.append(ticks[i])
                tick_labels.append(f'{ticks[i]:.1f}')

        if cmap_string == 'beta_star':
            tick_labels[0] = ''

        tick_locations.append(vmax)
        tick_labels.append(f'{vmax:.1f}')

        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels(tick_labels)

        if cmap_label is not None:
            cbar.set_label(cmap_label, size=12)  # , weight='bold')

    ax.coastlines()

    # Draw square if a region is specified (e.g. nino 3.4)
    if region is not None:
        rect = mpatches.Rectangle((region['lonmin'], region['latmin']),
                                  width=(region['lonmax'] - region['lonmin']),
                                  height=(region['latmax'] - region['latmin']),
                                  color='black', fill=None, linewidth=0.5, alpha=0.75, zorder=1000,
                                  transform=ccrs.PlateCarree())

        ax.add_patch(rect)  # Add patch

    plt.title(f'{title}')

    # Add label to bottom right
    lower_left_values_label = f'max:\nmin:'
    lower_left_values_text = f'{max_value:.3f}\n{min_value:.3f}'

    if topleft_label is not None:
        ax.text(0.000001, 0.99999, topleft_label, ha='left', va='bottom', fontweight='bold', transform=ax.transAxes)

    if bottomright_label is not None:
        ax.text(0.99, 0.01, bottomright_label, ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)

    top_right_values_label = f'mean:\nstdev:'
    top_right_values_text = f'{avg_value:.3f}\n{std_value:.3f}'

    ax.text(0.01, 0.01, lower_left_values_label, ha='left', va='bottom', fontweight='bold', transform=ax.transAxes)
    ax.text(0.15, 0.01, lower_left_values_text, ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)

    ax.text(0.85999, 0.99999, top_right_values_label, ha='left', va='bottom', fontweight='bold', transform=ax.transAxes)
    ax.text(1.000001, .99999, top_right_values_text, ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)

    # Place title and subtitle
    plt.title(f'{title}\n', pad=12)
    ax.text(0.5, 1, subtitle, ha='center', va='bottom', fontweight='bold', transform=ax.transAxes)

    # Add shaded layer based on threshold.
    if shading is not None:

        # Convert values to binary 0-1
        shading = (shading <= shading_threshold).astype(int)

        p = shading.plot.contourf(colors='None',
                                  hatches=['', '...'],
                                  levels=[0, 0.5, 1],
                                  add_colorbar=False,
                                  add_labels=False,
                                  ax=ax,
                                  transform=crs)

    return plt
