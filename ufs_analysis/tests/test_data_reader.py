# ---------------------------------------------------------------------------------------------------------------------
#  Filename: test_data_reader.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Test DataReader class.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import pytest

from src.datareader import datareader as dr
from src.datareader import DataReader_Super, UFS_DataReader, ERA5_DataReader, SUPPLIED_DataReader

case1 = {
    'datasource': 'UFS',
    'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
    'model': 'atm',
    'dataset_url': 's3://noaa-oar-sfsdev-pds/experiments/phase_1/baseline/atm_monthly.zarr',
    'class_name': UFS_DataReader.UFS_DataReader,
    'retrieve_params': {
        'var': ['uprs', 'vprs'],
        'lat': (30, 50),
        'lon': (240, 280),
        'time': ('2021-05-01', '2021-08-31T23'),
        'lead': (0, 2),
        'lev': (300, 500),
        'member': (2, 5)
    }
}

case2 = {
    'datasource': 'ERA5',
    'filename': '1959-2022-6h-512x256_equiangular_conservative.zarr',
    'dataset_url': 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr',
    'class_name': ERA5_DataReader.ERA5_DataReader,
    'retrieve_params': {
        'var': ['u_component_of_wind', 'v_component_of_wind'],
        'lat': (30, 50),
        'lon': (240, 280),
        'time': ('2021-05-01', '2021-08-31T23'),
        'lev': (300, 500),
    }
}

case3 = {
    'datasource': 'SUPPLIED'
}


@pytest.mark.parametrize('kwargs', [case1, case2])
def test_data_reader(kwargs):

    print('________ TEST DATA READER ________')

    # Validate .getDataReader()
    print('asserting error Failed to load')
    with pytest.raises(RuntimeError, match='Failed to load'):
        data_reader = dr.getDataReader(datasource=kwargs.get('datasource'), filename='asdf', model=kwargs.get('model'))

    # Get data
    data_reader = dr.getDataReader(datasource=kwargs.get('datasource'),
                                   filename=kwargs.get('filename'),
                                   model=kwargs.get('model'))

    print('asserting data_reader type')
    assert isinstance(data_reader, kwargs.get('class_name'))

    print('asserting dataset url')
    assert data_reader.dataset_url() == kwargs.get('dataset_url')

    # Validate .standardize_coords()
    coord_map = {'latitude': 'lat', 'y': 'lat', 'longitude': 'lon', 'x': 'lon', 'level': 'lev'}
    unpermitted_coordnames = set(coord_map.keys())

    print('asserting coordinate names')
    assert set(data_reader.dataset().dims).isdisjoint(unpermitted_coordnames)

    print('asserting lat-lon coordinate sorting')
    lats = list(data_reader.dataset().lat.values)
    lons = list(data_reader.dataset().lon.values)

    assert lats == sorted(lats, reverse=True)
    assert lons == sorted(lons, reverse=False)

    print('assert dimension order')
    dims = list(data_reader.dataset().dims)
    lat_ind = dims.index('lat')
    lon_ind = dims.index('lon')
    assert lon_ind == lat_ind + 1

    if kwargs.get('datasource') == 'UFS':
        init_ind = dims.index('init')
        lead_ind = dims.index('lead')
        assert lead_ind != init_ind
        assert lead_ind == init_ind + 2  # We expect init  member  lead

    # Validate .retrieve()
    print('asserting error Missing retrieve parameter: var')
    with pytest.raises(ValueError, match='Missing retrieve parameter: var'):
        data_reader.retrieve(lon=50.0)

    print('asserting error Variables not found')
    with pytest.raises(ValueError, match='Variables not found'):
        data_reader.retrieve(var='asdf')

    print('asserting mean dimension not found')
    # Get first variable, doesn't matter what it is.
    var = list(data_reader.dataset().variables)[0]
    with pytest.raises(ValueError, match='Mean: Dimension'):
        data_reader.retrieve(var=var, mean='asdf')

    print('asserting std dimension not found')
    with pytest.raises(ValueError, match='Std: Dimension'):
        data_reader.retrieve(var=var, std='asdf')

    print('asserting save_path unsupported format')
    with pytest.raises(ValueError, match='Unsupported format'):
        data_reader.retrieve(var=var, save_path='asdf.asdf')

    print('asserting retrieve results')
    retrieve_params = kwargs.get('retrieve_params')
    retrieved_ds = data_reader.retrieve(**retrieve_params)

    for param in retrieve_params:
        print('asserting variables')
        if param == 'var':
            assert retrieve_params[param] == list(retrieved_ds.keys())

        print('asserting lev')
        if param == 'lev':
            assert np.min(retrieved_ds.lev.values) >= retrieve_params[param][0]
            assert np.max(retrieved_ds.lev.values) <= retrieve_params[param][1]

        print('asserting lat')
        if param == 'lat':
            assert np.min(retrieved_ds.lat.values) >= retrieve_params[param][0]
            assert np.max(retrieved_ds.lat.values) <= retrieve_params[param][1]

        print('asserting lon')
        if param == 'lon':
            assert np.min(retrieved_ds.lon.values) >= retrieve_params[param][0]
            assert np.max(retrieved_ds.lon.values) <= retrieve_params[param][1]

        print('asserting time')
        if param == 'time':
            if 'init' in retrieved_ds.dims:
                assert np.min(retrieved_ds.init.values) >= np.datetime64(retrieve_params[param][0])
                assert np.max(retrieved_ds.init.values) <= np.datetime64(retrieve_params[param][1])
            else:
                assert np.min(retrieved_ds.time.values) >= np.datetime64(retrieve_params[param][0])
                assert np.max(retrieved_ds.time.values) <= np.datetime64(retrieve_params[param][1])

        print('asserting lead')
        if param == 'lead':
            assert np.min(retrieved_ds.lead.values) >= retrieve_params[param][0]
            assert np.max(retrieved_ds.lead.values) <= retrieve_params[param][1]

        print('asserting member')
        if param == 'member':
            assert np.min(retrieved_ds.member.values) >= retrieve_params[param][0]
            assert np.max(retrieved_ds.member.values) <= retrieve_params[param][1]
