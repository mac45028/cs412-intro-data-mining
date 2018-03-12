"""
pytest for decision_tree.py.
83% statement coverage(some rare exception cases are missing).
"""

import pytest
import os
import random

from core.decision_tree import DecisionTree, InvalidDatasetError
from core.evaluate_tree import import_data, parser_libsvm_format

base_dir = r'D:/Dropbox/Fall2016/412/MP/'
file_path = ('libsvmguide1.test', 'libsvmguide1.train')
file_path_real = ('balance-scale.train', 'balance-scale.train')


@pytest.fixture
def decision_tree_sample():
    return DecisionTree()


@pytest.fixture
def decision_tree_real():
    path = [os.path.join(base_dir, p) for p in file_path_real]
    test_lists, train_lists = import_data(*path)
    train_label, train_params = parser_libsvm_format(train_lists)
    return DecisionTree.create_decision_tree(
        train_label, train_params
    )


def test_import_files_none():
    with pytest.raises(AssertionError):
        import_data(None, None)


def test_import_files_non_valid():
    path = os.path.join(base_dir, 'all_blank_lines.train')
    test_lists, train_lists = import_data(path, '')
    assert (test_lists is None
            and train_lists is None)


def test_import_files_blank_lines():
    path = os.path.join(base_dir, 'all_blank_lines.train')
    test_lists, train_lists = import_data(path, path)
    assert test_lists == [] and train_lists == []


def test_import_files_valid():
    path = [os.path.join(base_dir, p) for p in file_path]
    test_lists, train_lists = import_data(*path)
    assert (len(test_lists) > 0 and len(train_lists) > 0)


def test_parser_libsvm_format():
    path = [os.path.join(base_dir, p) for p in file_path]
    test_lists, train_lists = import_data(*path)
    train_label, train_params = parser_libsvm_format(train_lists)
    assert train_label[0] == '1'
    assert (
            train_params[0] == ('2.617300e+01', '5.886700e+01', '-1.894697e-01', '1.251225e+02')
    )


def test_parser_libsvm_format_garbled():
    train_lists = ['asdit283vn8nu r082u3 0r82nu: :']
    train_label, train_params = parser_libsvm_format(train_lists)
    assert train_label == []
    assert train_params == []


def test__no_splitting_attribute_false(decision_tree_sample):
    assert decision_tree_sample.no_splitting_attribute(['1', '2']) is False


def test__no_splitting_attribute_true(decision_tree_sample):
    assert decision_tree_sample.no_splitting_attribute([]) is True


def test__all_same_class_exception(decision_tree_sample):
    with pytest.raises(InvalidDatasetError):
        decision_tree_sample.all_same_label([])


def test__all_same_class_true(decision_tree_sample):
    assert decision_tree_sample.all_same_label(['1', '1']) is True


def test__all_same_class_false(decision_tree_sample):
    assert decision_tree_sample.all_same_label(['1', '2']) is False


def test__detect_feature_type_true(decision_tree_sample):
    assert decision_tree_sample._detect_feature_type([('1', '2')]) is True


def test__detect_feature_type_false(decision_tree_sample):
    assert decision_tree_sample._detect_feature_type([('1.1', '3')]) is False


def test__split_dataset_by_feature():
    yes_index, no_index = DecisionTree._split_dataset_by_feature(
        features_list=[('1', '2'), ('3', '4')],
        target_feature='1'
    )
    assert yes_index == [0]
    assert no_index == [1]


def test__find_majority_class_invalid():
    with pytest.raises(InvalidDatasetError):
        DecisionTree.find_majority_class([])


def test__find_majority_class_valid():
    assert DecisionTree.find_majority_class(['1', '1', '2', '3']) == '1'


def test__calculate_probability_valid():
    assert DecisionTree._calculate_probability(['1', '1', '2'], '1') == 2/3


def test__calculate_gini_index():
    assert DecisionTree._calculate_gini_index(['1', '1', '3', '2']) == 0.625


def test__calculate_candidate_gini_value():
    train_label = ['1', '2', '1', '3']
    train_params = [('1', '2', '1'), ('2', '2', '2'), ('1', '1', '2'), ('1', '2', '2')]
    tree = DecisionTree.create_decision_tree(
        train_label, train_params
    )
    output = tree._calculate_candidate_gini_value(
        label_list=train_label,
        current_feature_list=train_params
    )

    assert output is not None


def test_create_decision_tree(decision_tree_real):
    assert decision_tree_real is not None


def test_classify(decision_tree_real):

    random.seed(1)

    expected_confusion_matrix = {
        'B': {'B': 49, 'L': 0, 'R': 0},
        'L': {'B': 0, 'L': 288, 'R': 0},
        'R': {'B': 0, 'L': 0, 'R': 288}
    }
    path = [os.path.join(base_dir, p) for p in file_path_real]
    test_lists, train_lists = import_data(*path)
    train_label, train_params = parser_libsvm_format(train_lists)

    output_confusion_matrix = decision_tree_real.classify(train_label, train_params)

    assert expected_confusion_matrix == output_confusion_matrix
