# ---------------------------------------------------------------------------------------------------------------------
#  Filename: UFS_DataReader.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Define the UFS DataReader subclass.
# ---------------------------------------------------------------------------------------------------------------------

from typing import Union, List, Tuple
from posixpath import join as urljoin
import xarray as xr

from .DataReader_Super import DataReader


class UFS_DataReader(DataReader):
    WINDS = [
        {'U_WIND': 'uprs', 'V_WIND': 'vprs'},
        {'U_WIND': 'u', 'V_WIND': 'v'},
        {'U_WIND': 'u10m', 'V_WIND': 'v10m'}
    ]

    '''Concrete: Reads and contains UFS datasets.'''

    def __init__(self, **kwargs):

        file_url = kwargs.get('filename', None)
        # process model
        model = kwargs.get('model', None)
        allowed_models = ["atm", "ocn", "lnd", "ice", "wav"]
        if model not in allowed_models:
            raise ValueError(f'Must supply model= one of {", ".join(allowed_models)}')

        freq = kwargs.get('freq', None)
        if freq is None:
            print(f"freq: deferring to 'monthly' forecast data")
            freq = 'monthly'
        allowed_freqs = ['monthly', 'daily']
        if freq not in allowed_freqs:
            raise ValueError(f"'freq' must be one of {', '.join(allowed_freqs)} or None")

        self.experiment = kwargs.get('experiment', 'baseline')  # <-- unique to UFS
        self.model = model  # <-- unique to UFS, atm or ocn
        self.freq = freq
        self._base_url = 's3://noaa-oar-sfsdev-pds/'
        self._default_file = f'experiments/phase_1/{self.experiment}/{self.model}_{self.freq}.zarr'

        super().__init__(file_url=file_url)

    def _read_dataset(self):
        try:
            self._dataset = xr.open_zarr(self._dataset_url,
                                         storage_options={"anon": True},
                                         consolidated=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load UFS dataset at address: {self._dataset_url} {e}")
