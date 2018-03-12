"""
pytest for random_forest.py.
83% statement coverage(some rare exception cases are missing).
"""

import pytest
import os
import random

from core.random_forest import RandomForest
from core.evaluate_tree import import_data, parser_libsvm_format

base_dir = r'D:/Dropbox/Fall2016/412/MP/'
file_path = ('libsvmguide1.test', 'libsvmguide1.train')
file_path_real = ('balance-scale.train', 'balance-scale.train')


@pytest.fixture
def random_forest():
    path = [os.path.join(base_dir, p) for p in file_path_real]
    test_lists, train_lists = import_data(*path)
    train_label, train_params = parser_libsvm_format(train_lists)
    return RandomForest.create_random_forest(
        train_label, train_params,
        pre_pruning=True, max_tree_height=7,
        number_of_tree=10
    )


def test_create_random_forest(random_forest):
    assert random_forest.tree_list != []


def test_classify_random_forest(random_forest):
    path = [os.path.join(base_dir, p) for p in file_path_real]
    test_lists, train_lists = import_data(*path)
    train_label, train_params = parser_libsvm_format(train_lists)
    result = random_forest.classify(train_label, train_params)
    assert result is not None

