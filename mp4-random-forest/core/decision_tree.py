"""
This module is my implementation of a simple decision tree based on the Gini impurity index.
Currently, only discrete feature is being supported. it is implemented in pure python due to
course requirement.
"""

import random
from math import sqrt, ceil
from collections import (
    Counter, namedtuple, defaultdict
)

from .binary_tree import BinaryTree


selection_result = namedtuple(
    'selection_result',
    [
        'candidate', 'gini_index', 'yes_index', 'no_index'
    ]
)


class InvalidDatasetError(Exception):
    """ Raised when dataset is empty or not in expected format. """
    pass


class DecisionTree(BinaryTree):
    """Decision tree implementation based on a simple binary tree."""

    def __init__(self):
        super().__init__()

    @property
    def label(self):  # pragma: no cover
        return self._label[1]

    @label.setter
    def label(self, label):  # pragma: no cover
        self._label = label

    @staticmethod
    def all_same_label(label_list):
        """
        Return `True` if all data have identical label; `False` otherwise.

        Parameters
        ----------
        label_list : list
            Label list of dataset.

        Returns
        -------
        predicate : bool
        """

        if not label_list:
            raise InvalidDatasetError('Empty label list.')

        if len(set(label_list)) == 1:
            return True
        else:
            return False

    @staticmethod
    def _detect_feature_type(features_list):
        """
        Return possible data type of input features list.

        Parameters
        ----------
        features_list : list of tuple
            List containing tuple of features.

        Returns
        -------
        data_type : int or float
            return `'int'` if data is discrete and `'float'` if continuous.
        """

        try:
            int(features_list[0][0])
        except ValueError:
            return False
        return True

    @staticmethod
    def _split_dataset_by_feature(features_list, target_feature):
        """
        Return two lists containing indices of data based on the split target.

        Parameters
        features_list : list of tuple
            List of tuple of data features.
        target_feature : str
            Splitting target.

        Returns
        -------
        yes_index : list
            List of dataset indices conforming to target value.
        no_index : list
            List of dataset indices not conforming to target value.
        """

        yes_index = [
            idx for idx, row in enumerate(features_list)
            if target_feature in row
        ]

        no_index = [
            idx for idx, _ in enumerate(features_list)
            if idx not in yes_index
        ]

        return yes_index, no_index

    @staticmethod
    def find_majority_class(label_list):
        """
        Return majority label of dataset.

        Parameters
        ----------
        label_list : list
            List of data label.

        Returns
        -------
        label : str
            label of a majority class.
        """

        if not label_list:
            raise InvalidDatasetError('Empty label list')
        else:
            result = Counter(label_list).most_common(1)[0]
            return result[0]

    @staticmethod
    def no_splitting_attribute(split_features_list):
        """Verify whether the splitting candidates are available."""
        for tuple_ in split_features_list:
            if tuple_:
                return False
        return True

    @staticmethod
    def _calculate_probability(label_list, target_label):
        """
        Calculate and return probability of the target label.
        Raises `InvalidDatasetError` if `label_list` is empty.
        """

        if not label_list:
            raise InvalidDatasetError('Empty dataset.')
        else:
            return label_list.count(target_label) / len(label_list)

    @classmethod
    def _calculate_gini_index(cls, label_list):
        """
        Calculate and return gini impurity index of input list.

        Parameters
        ----------
        label_list - list
            List of label data.

        Returns
        -------
        gini_index : float
            calculated gini impurity value.
        """
        gini_index = 1.0
        existing_class = frozenset(label_list)
        for class_ in existing_class:
            try:
                prob = cls._calculate_probability(
                    label_list, class_
                )
                gini_index -= prob ** 2
            except InvalidDatasetError:
                raise RuntimeError('Impossible case occurred.') # Then probability = 0
        return gini_index

    def _calculate_candidate_gini_value(self, label_list, current_feature_list):
        """
        Calculated and return suitable splitting-feature from specific feature index.

        Parameters
        ----------
        label_list : list
            List of data label.
        current_feature_list : list
            List of features at specific dimension.

        Returns
        -------
        selection_result : namedTuple
            namedtuple containing
            `('candidate', 'gini_index', 'feature_index', 'yes_index', 'no_index')`.
        """

        stored_candidate = None
        stored_yes_index = None
        stored_no_index = None
        min_gini_index = 1.0

        dataset_length = len(label_list)

        for candidate in set(current_feature_list):

            yes_index, no_index = self._split_dataset_by_feature(current_feature_list, candidate)

            yes_label_list = [
                label for index, label in enumerate(label_list)
                if index in yes_index
            ]
            no_label_list = [
                label for index, label in enumerate(label_list)
                if index in no_index
            ]

            tmp_gini = (
                    (len(yes_index) / dataset_length)
                    * self._calculate_gini_index(yes_label_list)
            )

            tmp_gini += (
                    (len(no_index) / dataset_length)
                    * self._calculate_gini_index(no_label_list)
            )

            if min_gini_index > tmp_gini:
                min_gini_index = tmp_gini
                stored_candidate = candidate
                stored_yes_index = yes_index
                stored_no_index = no_index

        return selection_result(
            stored_candidate, min_gini_index,
            stored_yes_index, stored_no_index
        )

    def attribute_selection_method(self, label_list, features_list, *,
                                   data_type='str', is_randomize=False):
        """
        Calculate and return splitting-feature for current node based on gini impurity index.

        Parameters
        ----------
        label_list : list
            List of data label.
        features_list : list of tuple
            List of tuple of data features.
        data_type : str
            Input datatype. Default to `str` for auto-detection.
        is_randomize : bool

        Returns
        -------
        selection_result : namedTuple
            namedtuple containing
            `('candidate', 'gini_index', 'feature_index', 'yes_index', 'no_index')`.

        """

        if data_type == 'str':
            feature_is_discrete = self._detect_feature_type(features_list)
        else:
            feature_is_discrete = data_type == 'int'

        if feature_is_discrete:

            split_by_attribute = []

            for feature_dimension, _ in enumerate(features_list[0]):
                # Select splitting candidate from each dimension
                current_feature_list = [row[feature_dimension] for row in features_list]

                if is_randomize:  # for random subspace method in random forest module
                    unique_features = set(row[feature_dimension] for row in features_list)
                    randomly_selected_features = random.sample(
                        unique_features,
                        int(ceil(sqrt(len(unique_features))))
                    )
                    current_feature_list = [
                        feature for feature in current_feature_list
                        if feature in randomly_selected_features
                    ]

                current_dimension_candidate = self._calculate_candidate_gini_value(
                    label_list, current_feature_list
                )

                # Add feature dimension for classification later
                prefix_candidate = ':'.join(
                    [str(feature_dimension),
                     current_dimension_candidate.candidate]
                )
                current_dimension_candidate = current_dimension_candidate._replace(
                    candidate=prefix_candidate
                )

                split_by_attribute.append(current_dimension_candidate)

            return min(split_by_attribute, key=lambda t: t.gini_index)
        else:
            raise NotImplementedError('Only support discrete feature for now.')

    @classmethod
    def create_decision_tree(cls, label_list, features_list, *,
                             is_randomize=False, pre_pruning=False,
                             max_tree_height=None, current_height=0):
        """
        Create a decision tree based on the input `label_list` and `features_list`.
        Pre-pruning is not performed by default.

        Parameters
        ----------
        label_list : list
            List of data label.
        features_list : list of tuple
            List of tuple of data features.
        is_randomize : bool

        pre_pruning : bool
            predicate for pre-pruning. False by default.
        max_tree_height : int
            Desired tree height. Ignored if `pre_pruning` is False.
        current_height : int
            Current tree height. Ignored if `pre_pruning` is False.

        Returns
        -------
        Tree : DecisionTree
            A fully-grown or trimmed tree, depending on `pre_pruning` parameter.
        """

        node = cls()

        # if pre-pruning
        if pre_pruning and current_height > max_tree_height:
            try:
                majority_class = cls.find_majority_class(label_list)
                node._label = ('class', majority_class[0])  # set to that class
            except InvalidDatasetError:
                node._label = ('class', random.sample(label_list, 1))
            return node

        if node.all_same_label(label_list):
            try:
                node._label = ('class', label_list[0])  # set to that class
            except InvalidDatasetError:
                node._label = ('class', random.sample(label_list, 1))
            return node

        if node.no_splitting_attribute(features_list):
            majority_class = cls.find_majority_class(label_list)
            node._label = ('class', majority_class[0])
            return node

        split_sub_set = node.attribute_selection_method(
            label_list, features_list, is_randomize=is_randomize)

        node._label = ('split', split_sub_set.candidate)

        yes_features_list = [
            feature_row for index, feature_row in enumerate(features_list)
            if index in split_sub_set.yes_index
        ]
        no_features_list = [
            feature_row for index, feature_row in enumerate(features_list)
            if index in split_sub_set.no_index
        ]

        yes_labels_list = [
            label for index, label in enumerate(label_list)
            if index in split_sub_set.yes_index
        ]

        no_labels_list = [
            label for index, label in enumerate(label_list)
            if index in split_sub_set.no_index
        ]

        if not yes_labels_list:
            left_node = cls()
            left_node._label = ('class', left_node.find_majority_class(label_list))
            left_node._parent = node
            node._left_node = left_node
        else:
            left_node = cls.create_decision_tree(
                yes_labels_list, yes_features_list,
                pre_pruning=pre_pruning, max_tree_height=max_tree_height,
                current_height=current_height + 1
            )
            left_node._parent = node
            node._left_node = left_node

        if not no_labels_list:
            right_node = cls()
            right_node._label = ('class', right_node.find_majority_class(label_list))
            right_node._parent = node
            node._right_node = right_node

        else:
            right_node = cls.create_decision_tree(
                no_labels_list, no_features_list,
                pre_pruning=pre_pruning, max_tree_height=max_tree_height,
                current_height=current_height + 1
            )
            right_node._parent = node
            node._right_node = right_node

        return node

    def classify_a_data(self, feature):
        """
        Classify a single data.

        Parameters
        ----------
        feature : tuple
            tuple of data features

        Returns
        -------
        label : str
            predicted label
        """

        root = self

        while root is not None:

            if root.is_leaf():
                return root.label

            else:
                node_label = root.label
                split_index, split_attribute = node_label.split(':')

                if feature[int(split_index)] == split_attribute:
                    root = root.left_node
                else:
                    root = root.right_node

    def classify(self, label_list, features_list):
        """
        Classify the dataset based on input decision tree, then return the confusion matrix.

        Parameters
        ----------
        label_list : list
            List of data label.
        features_list : list of tuple
            List of tuple of data features.

        Returns
        -------
        confusion_table : dict of dict
            Confusion matrix of classification.
        """

        unique_labels = set(label_list)

        confusion_table = defaultdict(dict)

        for x in unique_labels:
            for y in unique_labels:
                confusion_table[x][y] = 0

        for idx, feature in enumerate(features_list):

            target_label = label_list[idx]
            predicted_label = self.classify_a_data(feature)
            confusion_table[target_label][predicted_label] += 1

        return confusion_table
