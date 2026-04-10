# ---------------------------------------------------------------------------------------------------------------------
#  Filename: test_regrid.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Test Regrid class, especially .resample(), .regrid(), and .align() methods.
#           Results are compared against target files stored in the /test_in/ directory.
# ---------------------------------------------------------------------------------------------------------------------

import os
import sys
import yaml
import numpy as np
import pytest
import xarray as xr

from src.datareader import datareader as dr
from src.regridder import Regrid

# Note: time_range is set to init+1 for a reason. Leads 2 and 3 should be filled with NA when verif data are aligned.
case1 = {
    'case_number': 'case1',
    'case_message': 'ERA5 (256x512) -> UFS (192x384), Scalar variable, No lev',
    'system_to_regrid': 'ERA5',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-512x256_equiangular_conservative.zarr'
    },
    'resample_var': 'sea_surface_temperature',
    'regrid_var': 'sea_surface_temperature',
    'lev': None,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case2 = {
    'case_number': 'case2',
    'case_message': 'ERA5 (256x512) -> UFS (192x384), Scalar variable, At lev',
    'system_to_regrid': 'ERA5',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-512x256_equiangular_conservative.zarr'
    },
    'resample_var': 'temperature',
    'regrid_var': 'temperature',
    'lev': 500,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case3 = {
    'case_number': 'case3',
    'case_message': 'ERA5 (256x512) -> UFS (192x384), Wind vectors, No lev',
    'system_to_regrid': 'ERA5',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-512x256_equiangular_conservative.zarr'
    },
    'resample_var': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    'regrid_var': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    'lev': None,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case4 = {
    'case_number': 'case4',
    'case_message': 'ERA5 (256x512) -> UFS (192x384), Wind vectors, At lev',
    'system_to_regrid': 'ERA5',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-512x256_equiangular_conservative.zarr'
    },
    'resample_var': ['u_component_of_wind', 'v_component_of_wind'],
    'regrid_var': ['u_component_of_wind', 'v_component_of_wind'],
    'lev': 500,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case5 = {
    'case_number': 'case5',
    'case_message': 'UFS (192x384) -> ERA5 (121x240), Scalar variable, No lev',
    'system_to_regrid': 'UFS',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'
    },
    'resample_var': '2m_temperature',
    'regrid_var': 'tmp2m',
    'lev': None,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case6 = {
    'case_number': 'case6',
    'case_message': 'UFS (192x384) -> ERA5 (121x240), Scalar variable, At lev',
    'system_to_regrid': 'UFS',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'
    },
    'resample_var': 'temperature',
    'regrid_var': 'tprs',
    'lev': 500,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

# UFS does not have surface wind vector variables, so this case shall be a placeholder and not actually tested.
case7 = {
    'case_number': 'case7',
    'case_message': 'UFS (192x384) -> ERA5 (121x240), Wind vectors, No lev',
    'system_to_regrid': 'UFS'
}

case8 = {
    'case_number': 'case8',
    'case_message': 'UFS (192x384) -> ERA5 (121x240), Wind vectors, At lev',
    'system_to_regrid': 'UFS',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'ERA5',
        'filename': '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr',
    },
    'resample_var': ['u_component_of_wind', 'v_component_of_wind'],
    'regrid_var': ['uprs', 'vprs'],
    'lev': 500,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case9 = {
    'case_number': 'case9',
    'case_message': 'UFS (192x384) -> UFS (181x360), Scalar variable, No lev',
    'system_to_regrid': 'UFS',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/cpc_ics/atm_monthly.zarr',
        'model': 'atm'
    },
    'resample_var': None,
    'regrid_var': 'tmp2m',
    'lev': None,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

case10 = {
    'case_number': 'case10',
    'case_message': 'UFS (192x384) -> UFS (181x360), Scalar variable, At lev',
    'system_to_regrid': 'UFS',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/cpc_ics/atm_monthly.zarr',
        'model': 'atm'
    },
    'resample_var': None,
    'regrid_var': 'tprs',
    'lev': 500,
    'time_range': ('2021-05-01', '2021-06-30T23')
}

# UFS does not have surface wind vector variables, so this case shall be a placeholder and not actually tested.
case11 = {
    'case_number': 'case11',
    'case_message': 'UFS (192x384) -> UFS (181x360), Wind vectors, No lev'
}

case12 = {
    'case_number': 'case12',
    'case_message': 'UFS (192x384) -> UFS (181x360), Wind vectors, At lev',
    'system_to_regrid': 'UFS',
    'data_reader1': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/baseline/atm_monthly.zarr',
        'model': 'atm'
    },
    'data_reader2': {
        'datasource': 'UFS',
        'filename': 'experiments/phase_1/cpc_ics/atm_monthly.zarr',
        'model': 'atm'
    },
    'resample_var': None,
    'regrid_var': ['uprs', 'vprs'],
    'lev': 500,
    'time_range': ('2021-05-01', '2021-06-30T23')
}


def assert_equal_arrays(case_number, method, source):

    print(f'asserting {method} equivalence for {case_number}')

    # Get file handle
    this_dir = os.path.abspath(os.path.dirname(__file__))
    test_in_dir = os.path.join(this_dir, 'test_in')
    filename = f"{case_number}_{method}.nc"
    filepath = str(os.path.join(test_in_dir, filename))

    # Load file
    loaded_ds = xr.open_dataset(filepath)

    # In cases where None value is expected (e.g. there should be no 'aligned' data if regridding 2 UFS datasets)
    # if loaded_ds is None:
    #     assert loaded_ds == source
    if loaded_ds.equals(xr.Dataset()):
        assert source is None

    else:
        # Assert equivalence
        # Target datasets may differ in how lev is indexed.
        # Every array should be "flat" (i.e. single vertical level).
        # An index for vertical level may exist or not, as the code evolves,
        # but the underlying data should still be the same.

        # Copy source
        source_ds = source.dataset().copy()

        if 'lev' in loaded_ds.dims:
            loaded_ds = loaded_ds.squeeze(dim='lev').copy()

        if 'lev' in source.dataset().dims:
            source_ds = source_ds.squeeze(dim='lev')

        print(source_ds)
        print(loaded_ds)

        for thisvar in list(loaded_ds.keys()):
            assert np.array_equal(loaded_ds[thisvar].values, source_ds[thisvar].values, equal_nan=True)


@pytest.mark.parametrize('kwargs', [case1, case2, case3, case4, case5, case6, case8, case9, case10, case12])
def test_regrid(kwargs):

    # This is needed for comparing results against precomputed datasets.
    case_number = kwargs.get('case_number')
    case_message = kwargs.get('case_message', '')
    system_to_regrid = kwargs.get('system_to_regrid')
    print(f'{case_message}')

    # For now, we expect either UFS or ERA5 systems. This expectation could change in the future.
    if system_to_regrid not in ['UFS', 'ERA5']:
        raise ValueError(f'Expected case to have a system_to_regrid of either UFS or ERA5, got {system_to_regrid}')

    # Get data readers
    data_reader1 = dr.getDataReader(datasource=kwargs['data_reader1']['datasource'],
                                    filename=kwargs['data_reader1']['filename'],
                                    model=kwargs['data_reader1'].get('model', None))

    data_reader2 = dr.getDataReader(datasource=kwargs['data_reader2']['datasource'],
                                    filename=kwargs['data_reader2']['filename'],
                                    model=kwargs['data_reader2'].get('model', None))

    # Initialize Regrid object
    regridder = Regrid.Regrid(data_reader1=data_reader1,
                              data_reader2=data_reader2,
                              method=kwargs.get('method', 'linear'))
    # Run resample
    regridder.resample(var=kwargs.get('resample_var'),
                       lev=kwargs.get('lev', None),
                       time=kwargs.get('time_range'),
                       use_mp=False)

    # Assert
    assert_equal_arrays(case_number, 'resample', regridder.resampled)

    # Assert regrid error capturing
    print('asserting error regrid var empty')
    with pytest.raises(ValueError, match='Must supply var as a string or list of two wind vectors'):
        regridder.regrid(var=[])

    print('asserting error regrid var more than 2')
    with pytest.raises(ValueError, match='If var is a list, then it can only contain a maximum of two variables'):
        regridder.regrid(var=['asdf', 'qwer', 'zxcv'])

    print('asserting error regrid only 1 wind vector')
    with pytest.raises(ValueError, match='You supplied one wind vector'):

        if system_to_regrid == 'ERA5':
            regridder.regrid(var=['u_component_of_wind', '10m_v_component_of_wind'])

        elif system_to_regrid == 'UFS':
            regridder.regrid(var=['uprs'])

    print('asserting error regrid no wind vectors')
    with pytest.raises(ValueError, match='You supplied a list of variables that are not registered as wind vectors'):
        regridder.regrid(var=['asdf', 'qwer'])

    print('asserting error regrid global domain')
    with pytest.raises(ValueError, match='Spherical harmonics can only be run on the full global domain'):

        if system_to_regrid == 'ERA5':
            regridder.regrid(var=['u_component_of_wind', 'v_component_of_wind'], lat=(30, 50), lon=(240, 280))

        elif system_to_regrid == 'UFS':
            regridder.regrid(var=['uprs', 'vprs'], lat=(30, 50), lon=(240, 280))

    print('asserting error regrid var as string')
    with pytest.raises(ValueError, match='var must be a string representing a single variable in the dataset'):
        regridder.regrid(var=set())

    print('asserting error regrid var is 1 wind vector')
    with pytest.raises(ValueError, match='You supplied one wind vector'):

        if system_to_regrid == 'ERA5':
            regridder.regrid(var='u_component_of_wind')

        elif system_to_regrid == 'UFS':
            regridder.regrid(var='uprs')

    print('asserting member or ensemble average')
    if system_to_regrid == 'UFS':
        with pytest.raises(ValueError, match='To regrid an ensemble model'):
            regridder.regrid(var=kwargs.get('regrid_var'), ens_avg=False)

    # Run regrid
    if system_to_regrid == 'UFS':
        regridder.regrid(var=kwargs.get('regrid_var'),
                         lev=kwargs.get('lev', None),
                         time=kwargs.get('time_range'),
                         ens_avg=True)
    else:
        print('system_to_regrid is not UFS')
        regridder.regrid(var=kwargs.get('regrid_var'),
                         lev=kwargs.get('lev', None),
                         time=kwargs.get('time_range'))

    # Assert
    assert_equal_arrays(case_number, 'regrid', regridder.regridded)

    # Run align
    regridder.align()

    # Assert
    assert_equal_arrays(case_number, 'align', regridder.aligned)
