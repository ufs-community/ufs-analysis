##
## SUPPLIED Data Reader
##

from typing import Union, List, Tuple
from posixpath import join as urljoin
import xarray as xr
from xarray.groupers import TimeResampler

from .DataReader_Super import DataReader

class SUPPLIED_DataReader(DataReader):
    '''Concrete: Reads and contains SUPPLIED datasets.'''

    def __init__(self, **kwargs):

        self._base_url = ''
        self._default_file = ''

        dataset = kwargs.get('dataset', None)
        if not isinstance(dataset, xr.Dataset):
            msg = f'''dataset must be of xarray.Dataset class for a supplied datasource.
            Got {type(dataset)}.
            '''
            raise ValueError(msg)

        self.dataset = dataset

        super().__init__()

    def _read_dataset(self):
        pass
