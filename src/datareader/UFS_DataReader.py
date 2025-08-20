##
## UFS Data Reader
##

from typing import Union, List, Tuple
from posixpath import join as urljoin
import xarray as xr

from .DataReader_Super import DataReader

class UFS_DataReader(DataReader):
    U_WIND, V_WIND = 'uprs', 'vprs'

    '''Concrete: Reads and contains UFS datasets.'''

    def __init__(self, **kwargs):

        file_url = kwargs.get('filename', None)
        # process model
        model = kwargs.get('model', None)
        allowed_models = ["atm", "ocn", "lnd", "ice", "wav"]
        if model not in allowed_models:
            raise ValueError(f'Must supply model= one of {", ".join(allowed_models)}')

        self.model = model  # <-- unique to UFS
        self._base_url = 's3://noaa-oar-sfsdev-pds/'
        self._default_file = 'experiments/phase_1/baseline/atm_monthly.zarr'

        super().__init__(file_url=file_url)

    def _read_dataset(self):
        try:
            self.dataset = xr.open_zarr(self._dataset_url,
                                        storage_options={"anon": True},
                                        consolidated=True)
            print(f"UFS dataset loaded.")

        except Exception as e:
            raise RuntimeError(f"Failed to load UFS dataset at address: {self._dataset_url} {e}")
