"""
pytest for binary_tree.py.
100% statement coverage.
"""

import pytest

from core.binary_tree import BinaryTree


@pytest.fixture
def binary_tree():
    root = BinaryTree()
    left_node = BinaryTree()

    left_node._label = '1'
    left_node._left_node = None
    left_node._right_node = None

    root._left_node = left_node
    root._right_node = BinaryTree()

    return root

@pytest.fixture
def binary_tree_unbalanced():
    root = BinaryTree()
    left_node = BinaryTree()

    left_node._label = '1'
    left_node._left_node = None
    left_node._right_node = None
    left_node._left_node = BinaryTree()

    root._left_node = left_node
    root._right_node = BinaryTree()

    return root


def test_is_leaf(binary_tree):
    assert binary_tree.is_leaf() is False
    assert binary_tree._right_node.is_leaf() is True


def test_get_max_height(binary_tree):
    assert binary_tree.get_max_height() == 1


def test_get_max_height_unbalanced(binary_tree_unbalanced):
    assert binary_tree_unbalanced.get_max_height() == 2