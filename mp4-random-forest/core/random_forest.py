"""
Random forest implementation based on the decision tree module. The idea is straightforward
since random forest is the ensemble of decision tree. Hence, we have to generate desired number
of trees and perform classification on each tree to get the majority vote.
"""
import concurrent.futures
import math
import multiprocessing
import random
import functools
from collections import Counter, defaultdict

from .decision_tree import DecisionTree


class RandomForest:
    """Random forest implementation based on decision tree."""
    def __init__(self):  # pragma: no cover
        self._tree_list = []

    @property
    def tree_list(self):  # pragma: no cover
        return self._tree_list

    @tree_list.setter
    def tree_list(self, tree):  # pragma: no cover
        self._tree_list.append(tree)

    def extend_tree_list(self, tree_list):  # pragma: no cover
        self._tree_list.extend(tree_list)

    @classmethod
    def create_random_forest(cls, label_list, features_list, *,
                             pre_pruning=False, max_tree_height=None,
                             number_of_tree=100):
        """
        Create a random forest based on the input `train_labels` and `features_list`.

        Parameters
        ----------
        label_list : list of list
             List of data label list.
        features_list : list of list
            List of list of data features tuple.
        number_of_tree : int
            Number of trees to grow. Default is 100 trees.
        pre_pruning : bool
            predicate for pre-pruning. False by default.
        max_tree_height : int
            Desired tree height. Ignored if `pre_pruning` is False.

        Returns
        -------
        forest : RandomForest
            A fully grown forest.
        """

        train_label_sample, train_params_sample = next(
                cls._create_random_train_data(label_list, features_list, number_of_tree)
            )

        forest = cls()

        create_decision_func = functools.partial(
            DecisionTree.create_decision_tree,
            is_randomize=True,
            pre_pruning=pre_pruning,
            max_tree_height=max_tree_height
        )

        try:

            process_count = (multiprocessing.cpu_count() - 1)

            with concurrent.futures.ProcessPoolExecutor(max_workers=max(2, process_count)) as pool:

                trees = pool.map(
                    create_decision_func,
                    train_label_sample, train_params_sample,
                    chunksize=process_count
                )

            forest.extend_tree_list(trees)

        except RuntimeError:
            for idx, _ in enumerate(train_label_sample):
                tree = create_decision_func(
                    train_label_sample[idx], train_params_sample[idx]
                )
                forest.tree_list = tree

        return forest

    @staticmethod
    def _create_random_train_data(label_list, features_list, number_of_sample):
        """
        Return a prepared dataset for growing forest. The data will be randomly select
        with replacement and each tree receives around 2/3 of original dataset.

        Parameters
        ----------
        label_list : list
            List of data label.
        features_list : list of tuple
            List of tuple of data features.
        number_of_sample : int
            Number of samples to generate

        Returns
        -------
        train_label_sample : list of list
            List of class's label list.
        train_params_sample : list of list
            List of list of data features tuple.
        """

        train_label_sample = []
        train_params_sample = []

        training_data_length = len(label_list)

        number_of_elements = int(math.floor(training_data_length * (2 / 3)))

        # generate random indices
        all_indices = [x for x in range(number_of_elements)]

        for rnd in range(number_of_sample):

            selected_indices = random.sample(all_indices, number_of_elements)

            train_label_sample.append(
                [label for idx, label in enumerate(label_list)
                 if idx in selected_indices]
            )
            train_params_sample.append(
                [tuple_ for idx, tuple_ in enumerate(features_list)
                 if idx in selected_indices]
            )

        yield train_label_sample, train_params_sample

    def classify(self, label_list, features_list):
        """
        Classify the dataset based on random forest, then return the confusion matrix.

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

        for idx, _ in enumerate(label_list):
            target_label = label_list[idx]

            predicted_label = Counter(
                label for label in
                (tree.classify_a_data(features_list[idx]) for tree in self.tree_list)
            )

            majority_label = predicted_label.most_common()[0][0]

            confusion_table[target_label][majority_label] += 1

        return confusion_table

