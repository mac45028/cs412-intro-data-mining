import os

from core.evaluate_tree import prepare_dataset_and_classify


base_directory = r'C:/412/MP/'
files_path = ('balance-scale.train', 'balance-scale.train')

path = [os.path.join(base_directory, p) for p in files_path]


def main():
    # matrix = prepare_dataset_and_classify(*path, is_rf=False, pre_pruning=True, max_tree_height=7)
    matrix = prepare_dataset_and_classify(*path, is_rf=True, number_of_tree=200,
                                 pre_pruning=True, max_tree_height=4)
    print(matrix)


if __name__ == '__main__':
    main()
