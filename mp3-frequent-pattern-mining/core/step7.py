from copy import deepcopy
from operator import mul, itemgetter
from functools import reduce
import math

from core.step5 import import_pattern_list
from core.step6 import rank_by_purity


def rank_by_completeness(*, threshold=0.5):
    all_patterns = import_pattern_list()
    all_patterns_completeness = deepcopy(all_patterns)

    for topic_index, freq_dict_current in enumerate(all_patterns):

        # generate base pattern set for finding super-pattern
        freq_list = list(freq_dict_current.keys())

        base_set = list(
            frozenset(pattern.split(':')) for pattern in freq_list
        )

        for current_pattern, current_support in freq_dict_current.items():

            cur_pattern_set = set(current_pattern.split(':'))

            tmp_list = []

            for test_set in base_set:

                if cur_pattern_set < test_set:  # super-pattern exist
                    tmp_list.append(
                            ':'.join(token for token in sorted(test_set, key=int))
                    )

            if tmp_list:  # calculate completeness value if super-pattern exist
                completeness = max(
                    freq_dict_current.get(super_pattern) / current_support
                    for super_pattern in tmp_list
                )

            else:
                completeness = 0

            if completeness > threshold:
                all_patterns_completeness[topic_index][current_pattern] = 1 - completeness
            else:
                all_patterns_completeness[topic_index][current_pattern] = 0

        # prune the non-complete pattern
        for pattern in list(all_patterns_completeness[topic_index]):
                if all_patterns_completeness[topic_index][pattern] == 0:
                    del all_patterns_completeness[topic_index][pattern]

    return all_patterns_completeness


def rank_by_coverage():
    # calculate coverage(probability) value
    all_patterns_coverage = import_pattern_list()

    for current_freq_dict in all_patterns_coverage:

        dict_size = len(current_freq_dict)

        for pattern, support in current_freq_dict.items():
            current_freq_dict[pattern] = support / dict_size
    return all_patterns_coverage


def rank_by_phraseness(all_patterns_coverage):
    # calculate phraseness value
    all_patterns_phraseness = deepcopy(all_patterns_coverage)

    for topic_idx, current_freq_dict in enumerate(all_patterns_phraseness):
        for pattern, prob in current_freq_dict.items():

            prob_each_tokens = reduce(
                mul,
                (all_patterns_coverage[topic_idx][token] for token in pattern.split(':')),
                1
            )

            try:
                phraseness = math.log(prob/prob_each_tokens)
            except ZeroDivisionError:
                print(pattern)  # Unlikely to happen, just in case
                phraseness = - float('inf')
            current_freq_dict[pattern] = phraseness
    return all_patterns_phraseness


# step7 this function will calculate combined ranking measure based on the paper
def rank_by_complete_coverage_purity_and_phrase(completeness_threshold, phraseness_weight):

    all_patterns_purity = rank_by_purity()
    all_patterns_completeness = rank_by_completeness(threshold=completeness_threshold)
    all_patterns_coverage = rank_by_coverage()
    all_patterns_phraseness = rank_by_phraseness(all_patterns_coverage)

    # Ranking pattern based on the formula in paper
    for topic_idx, current_freq_dict in enumerate(all_patterns_completeness):
        for pattern in current_freq_dict:
            rank = all_patterns_coverage[topic_idx][pattern] * (
                    ((1-phraseness_weight) * all_patterns_purity[topic_idx][pattern])
                    + (phraseness_weight * all_patterns_phraseness[topic_idx][pattern])
            )
            all_patterns_completeness[topic_idx][pattern] = rank

    # sort patterns based on ranking value
    result = list(
        sorted(current_freq_dict.items(), key=itemgetter(1), reverse=True)
        for current_freq_dict in all_patterns_completeness
    )

    # generate output
    for topic_index, topic_list in enumerate(result):

        output = '\n'.join(
            ' '.join([str(support), pattern.replace(':', ' ')])
            for pattern, support in topic_list
        )

        with open(f'combined_ranking/pattern-{str(topic_index)}.txt', 'w') as f:
            f.write(output)

    return result

