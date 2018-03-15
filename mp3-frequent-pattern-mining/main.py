"""
This is a PoC of frequent pattern mining for MP3, so no testing and exception handling will be
implemented. Tested on python 3.6.
"""

import subprocess

from core.step2 import (
    import_data, generate_vocab, generate_title
)
from core.step3 import separate_by_terms
from core.step4 import apriori
from core.step5 import generate_max_or_closed_pattern_output
from core.step6 import rank_by_purity
from core.step7 import rank_by_complete_coverage_purity_and_phrase

lda_location = r'/Users/sthk/lda_binary'


def index_to_word(pattern_to_word_dict, data):

    tmp = []

    for line in data:
        support, *patterns = line.split()

        map_words = ' '.join(
            pattern_to_word_dict[int(pattern)]
            for pattern in patterns
        )

        tmp.append(' '.join([support, map_words]))

    output = '\n'.join(tmp)

    return output


def convert_pattern_index_to_word(index_to_word_dict):

    for topic_index in range(5):

        # read base data to convert
        with open(f'combined_ranking/pattern-{str(topic_index)}.txt') as f:
            combined_data = f.readlines()

        with open(f'closed/closed-{str(topic_index)}.txt') as f:
            closed_data = f.readlines()

        with open(f'max/max-{str(topic_index)}.txt') as f:
            max_data = f.readlines()

        with open(f'purity/purity-{str(topic_index)}.txt') as f:
            purity_data = f.readlines()

        with open(f'patterns/pattern-{str(topic_index)}.txt') as f:
            patterns_data = f.readlines()

        # convert and write output
        output_str = index_to_word(index_to_word_dict, combined_data)
        with open(f'combined_ranking/pattern-{str(topic_index)}.txt.phrase',
                  'w') as f:
            f.write(output_str)

        output_str = index_to_word(index_to_word_dict, closed_data)
        with open(f'closed/closed-{str(topic_index)}.txt.phrase', 'w') as f:
            f.write(output_str)

        output_str = index_to_word(index_to_word_dict, max_data)
        with open(f'max/max-{str(topic_index)}.txt.phrase', 'w') as f:
            f.write(output_str)

        output_str = index_to_word(index_to_word_dict, purity_data)
        with open(f'purity/purity-{str(topic_index)}.txt.phrase', 'w') as f:
            f.write(output_str)

        output_str = index_to_word(index_to_word_dict, patterns_data)
        with open(f'patterns/pattern-{str(topic_index)}.txt.phrase', 'w') as f:
            f.write(output_str)


def main():
    # Step 2.1
    data_list, data_dict = import_data('paper.txt')
    generate_vocab(data_list)

    # Step 2.2
    title_map = generate_title(data_list, data_dict)

    # Step 3.1
    # Uncomment to call the lda binary
    # subprocess.call(
    #     [lda_location, 'est', '0.001', '5', 'settings.txt', 'title.txt', 'random', 'result']
    # )
    #
    # Step 3.2
    separate_by_terms()

    # Step 4
    apriori(0.005)  # relative minimum support ~ 0.5%

    # Step 5
    generate_max_or_closed_pattern_output(is_closed_mining=False)
    generate_max_or_closed_pattern_output(is_closed_mining=True)

    # Step 6
    rank_by_purity()

    # Step 7
    rank_by_complete_coverage_purity_and_phrase(0.5, 0.5)

    # Create readable output from index
    convert_pattern_index_to_word(title_map)


if __name__ == '__main__':
    main()
