from collections import Counter
from operator import itemgetter


def import_data(paper_filepath):
    """load the `paper.txt` and return pattern and support value."""

    with open(paper_filepath, 'r') as f:
        raw_text = f.readlines()

    # get tokens from each line, ignoring first token
    token_generator = (
        tokens for _, *tokens in
        (line.split() for line in raw_text)
    )

    data_dict = {
        index: Counter(tokens) for index, tokens in
        enumerate(line for line in token_generator)
    }

    data_set = set(
        token for each_dict in data_dict.values()
        for token in each_dict.elements()
    )

    return list(data_set), data_dict


def generate_vocab(data_set):
    """create vocab.txt"""
    output = '\n'.join(data_set)

    with open('vocab.txt', 'w') as f:
        f.write(output)


def generate_title(data_list, data_dict):
    """create and return `title.txt`."""
    title_map = {token: index for index, token in enumerate(data_list)}

    output = []
    for each_dict in data_dict.values():
        tmp_list = [str(len(each_dict))]

        sorted_titles = sorted(  # sort title for comparison later in step4
            [
                (title_map[token], count)
                for token, count in each_dict.most_common()
            ], key=itemgetter(0)
        )

        tmp_list.extend(
                ':'.join(
                        [str(title_index), str(support)]
                ) for title_index, support in sorted_titles
        )

        output.append(' '.join(tmp_list))

    with open('title.txt', 'w') as f:
        f.write('\n'.join(output))

    # reverse mapping for later usage...
    title_map = {index: token for token, index in title_map.items()}

    return title_map
