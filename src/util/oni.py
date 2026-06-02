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
import copy
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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
