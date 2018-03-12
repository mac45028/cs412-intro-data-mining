"""Sample binary tree implementation in python."""


class BinaryTree:

    def __init__(self):
        self._left_node = None
        self._right_node = None
        self._parent = None
        self._label = None

    @property
    def label(self):  # pragma: no cover
        return self._label

    @label.setter
    def label(self, label):  # pragma: no cover
        self._label = label

    @property
    def parent(self):  # pragma: no cover
        return self._parent

    @parent.setter
    def parent(self, parent):  # pragma: no cover
        self._parent = parent

    @property
    def left_node(self):  # pragma: no cover
        return self._left_node

    @left_node.setter
    def left_node(self, left_node):  # pragma: no cover
        self._left_node = left_node

    @property
    def right_node(self):  # pragma: no cover
        return self._right_node

    @right_node.setter
    def right_node(self, right_node):  # pragma: no cover
        self._right_node = right_node

    def is_leaf(self):
        return (self._left_node is None
                and self._right_node is None)

    def get_max_height(self):

        tmp_height_left = 0
        tmp_height_right = 0

        if self._left_node is not None:
            tmp_height_left = self._left_node.get_max_height() + 1
        if self._right_node is not None:
            tmp_height_right = self._right_node.get_max_height() + 1

        if tmp_height_left > tmp_height_right:
            return tmp_height_left
        else:
            return tmp_height_right

    def show_tree(self, *, level=0):  # pragma: no cover
        print('\t' * level + repr(self._label))
        if self._left_node is not None:
            self._left_node.show_tree(level=level + 1)
        if self._right_node is not None:
            self._right_node.show_tree(level=level + 1)
