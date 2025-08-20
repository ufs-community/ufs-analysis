##
## ERA5 Data Reader
##

from typing import Union, List, Tuple
from posixpath import join as urljoin
import xarray as xr
from xarray.groupers import TimeResampler

from .DataReader_Super import DataReader

class ERA5_DataReader(DataReader):
    U_WIND, V_WIND = 'u_component_of_wind', 'v_component_of_wind'

    '''Concrete: Reads and contains ERA5 datasets.'''

    def __init__(self, **kwargs):

        file_url = kwargs.get('filename', None)
        self._base_url = 'gs://gcp-public-data-arco-era5/ar/'
        self._default_file = '1959-2022-6h-512x256_equiangular_conservative.zarr/'

        super().__init__(file_url=file_url)

    def _read_dataset(self):
        try:
            self.dataset = xr.open_zarr(self._dataset_url,
                                        storage_options={"token": "anon"},
                                        consolidated=True,
                                        decode_timedelta=False,
                                        decode_times=True,
                                        chunks=None)
            print(f"ERA5 dataset loaded.")

            # Chunk by month by default.  Needed for transpose laziness.
            self.dataset = self.dataset.chunk(time=TimeResampler("MS"))

        except Exception as e:
            raise RuntimeError(f"Failed to load ERA5 dataset at address: {self._dataset_url} {e}")
