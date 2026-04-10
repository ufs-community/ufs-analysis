# ---------------------------------------------------------------------------------------------------------------------
#  Filename: DataReader_Factory.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Instantiate and return the appropriate DataReader subclass.
# ---------------------------------------------------------------------------------------------------------------------

import os
import importlib

from .DataReader_Super import DataReader


class DataReader_Factory:

    def create_DataReader(datasource: str, **kwargs) -> DataReader:

        # Infer data reader subclasses per file naming template <datasource>_DataReader.py
        wd = os.path.abspath(os.path.dirname(__file__))
        filelist = os.listdir(wd)
        data_reader_filelist = [this for this in filelist if '_DataReader.py' in this]

        # Extract the data reader subclass names.  These are all available data readers.
        data_readers = [os.path.splitext(this)[0] for this in data_reader_filelist]
        datasources = [this.split('_')[0] for this in data_readers]

        # Validate user input.
        datasource = datasource.upper()
        if datasource not in datasources:
            raise ValueError(f'datasource must be one of {", ".join(datasources)}')

        # A data reader for this datasource exists.
        module_name = f'{datasource}_DataReader'

        # Import the module for this data reader.
        module = importlib.import_module(f'src.datareader.{module_name}')
        data_reader = getattr(module, f'{module_name}')

        return data_reader(**kwargs)
