from warnings import warn

from .decision_tree import DecisionTree
from .random_forest import RandomForest


def prepare_dataset_and_classify(train_filepath, test_filepath, *,
                                 is_rf=False, number_of_tree=100,
                                 pre_pruning=False, max_tree_height=None):
    """
    Helper function for tree object creation and classification.

    Parameters
    ----------
    train_filepath : str
        Filepath of training dataset.
    test_filepath : str
        Filepath of testing dataset.
    is_rf : bool
        Creating random forest or decision tree.
    number_of_tree : int
        Number of trees to grow; ignore if `is_rf` is False.
    pre_pruning : bool
        Is pre-pruning being used or not.
    max_tree_height : int
        Maximum height of tree to grow.

    Returns
    -------
    Confusion matrix : dict of dict
        Confusion matrix of the testing dataset.
    """

    test_list, train_list = import_data(test_filepath, train_filepath)

    test_label, test_params = parser_libsvm_format(test_list)
    train_label, train_params = parser_libsvm_format(train_list)

    if is_rf:
        rf = RandomForest.create_random_forest(train_label, train_params,
                                               pre_pruning=pre_pruning,
                                               number_of_tree=number_of_tree,
                                               max_tree_height=max_tree_height)
        return rf.classify(test_label, test_params)
    else:
        dt = DecisionTree.create_decision_tree(train_label, train_params,
                                               pre_pruning=pre_pruning,
                                               max_tree_height=max_tree_height)
        return dt.classify(test_label, test_params)


def import_data(test_file, train_file):
    """
    Import and return list of pre-processed data.

    Parameters
    ----------
    test_file : str
        Filepath of training dataset.
    train_file : str
        Filepath of testing dataset.

    Returns
    -------
    list_test_file : list
        List of `test_file` data.
    list_train_file: list
        List of `train_file` data.
    """
    assert (test_file is not None
            and train_file is not None)
    try:
        with open(train_file, 'r') as f:
            list_train_file = f.readlines()
        with open(test_file, 'r') as f:
            list_test_file = f.readlines()
    except IOError as exc:
        warn(f'one of the input cannot be read or located.')
        warn(str(exc))
        return None, None

    try:
        list_train_file = [line.strip() for line in list_train_file if not line.isspace()]
        list_test_file = [line.strip() for line in list_test_file if not line.isspace()]
    except Exception as exc:
        print(exc)
        raise

    return list_test_file, list_train_file


def parser_libsvm_format(input_file):
    """
    Parse and return the label and features list from the pre-processed data in libsvm format.

    Parameters
    ----------
    input_file : list
        Pre-processed dataset from `import_data` function.

    Returns
    -------
    label_list : list
        List of data label.
    features_list : list of tuple
        List of tuple of data features.
    """

    label_list = []
    features_list = []

    try:
        for line in input_file:
            label, *parameters = line.split()
            label_list.append(label)
            features_list.append(
                tuple(arg.split(':')[1] for arg in parameters)
            )

    except Exception as exc:
        print(str(exc))
        label_list = []
        features_list = []

    return label_list, features_list
