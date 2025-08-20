import importlib

def getDataReader(datasource, **kwargs):

    module = importlib.import_module('src.datareader.DataReader_Factory')
    fact = getattr(module, 'DataReader_Factory')

    data_reader = fact.create_DataReader(datasource, **kwargs)

    return data_reader
