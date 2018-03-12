"""
pytest for evaluate_tree.py.
93% statement coverage.
"""

import os

from core.evaluate_tree import prepare_dataset_and_classify


def test_prepare_dataset_and_classify_dt():
    base_dir = r'D:/Dropbox/Fall2016/412/MP/'
    file_path_real = ('balance-scale.train', 'balance-scale.train')
    path = [os.path.join(base_dir, p) for p in file_path_real]
    matrix = prepare_dataset_and_classify(*path, is_rf=False, pre_pruning=True, max_tree_height=7)
    assert matrix is not None


def test_prepare_dataset_and_classify_rf():
    base_dir = r'D:/Dropbox/Fall2016/412/MP/'
    file_path_real = ('balance-scale.train', 'balance-scale.train')
    path = [os.path.join(base_dir, p) for p in file_path_real]
    matrix = prepare_dataset_and_classify(*path, is_rf=True, number_of_tree=100,
                                          pre_pruning=True, max_tree_height=4)
    assert matrix is not None
