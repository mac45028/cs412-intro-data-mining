from copy import deepcopy
from math import log
from collections import OrderedDict
from operator import itemgetter

from core.step5 import import_pattern_list


def rank_by_purity():
    """rank pattern based on purity measure."""
    all_patterns = import_pattern_list()
    all_patterns_purity = deepcopy(all_patterns)

    for topic_index, freq_dict_current in enumerate(all_patterns):
        for pattern, support in freq_dict_current.items():

            prob_current_topic = log(support/len(freq_dict_current))

            try:
                prob_other_topics = log(max(
                    (support+freq_dict_other.get(pattern, 0.0)) / (len(freq_dict_current)+len(freq_dict_other))
                    for idx, freq_dict_other in enumerate(all_patterns)
                    if topic_index != idx
                ))
            except ValueError:
                prob_other_topics = 0

            all_patterns_purity[topic_index][pattern] = prob_current_topic - prob_other_topics

        # sort frequent dict, based on support, for printing
        all_patterns_purity[topic_index] = OrderedDict(
            sorted(all_patterns_purity[topic_index].items(),
                   key=itemgetter(1),
                   reverse=True)
        )

        output = '\n'.join(
            ' '.join(
                [
                    str(support),
                    pattern.replace(':', ' ')
                ]
            )
            for pattern, support in all_patterns_purity[topic_index].items()
        )

        with open(f'purity/purity-{str(topic_index)}.txt', 'w') as f:
            f.write(output)

    return all_patterns_purity

