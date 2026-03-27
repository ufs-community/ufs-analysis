# ---------------------------------------------------------------------------------------------------------------------
#  Filename: test_pep8.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Test entire repository for PEP8 code style adherence.
# ---------------------------------------------------------------------------------------------------------------------

import os
import sys
import pycodestyle


def test_pep8():

    print('\n')
    print(f'________ TEST PEP8 ________')

    # Get absolute path to this directory
    this_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(__file__, "../.."))

    print(f'Validating pep8 standards for all Python files in {parent_dir}.')

    # Get all python files in a list
    py_file_list = []

    # Iterate over repo directory
    for dir_path, dir_names, filenames in os.walk(parent_dir):

        # Iterate over files in the directory
        for filename in filenames:

            # Get file extensions
            file_splits = filename.split('.')

            # Filter out non-Python files
            if file_splits[-1] != 'py':
                continue

            # Python file list
            py_file_list.append(os.path.join(dir_path, filename))

    # Check that we'll validating all Python files in our repo.
    # You will need to adjust this number if/when more Python files are created in the future.
    n_python_files_in_repository = 23
    assert len(py_file_list) == n_python_files_in_repository

    # Get config file and define style
    config_file = os.path.join(this_dir, 'tox.ini')
    style = pycodestyle.StyleGuide(quiet=False, config_file=config_file)

    # Check files
    print(f'Number files being validated for PEP8 standards: {len(py_file_list)}')
    print('\n'.join(py_file_list), '\n')
    results = style.check_files(py_file_list)

    # Assert
    assert results.total_errors == 0
