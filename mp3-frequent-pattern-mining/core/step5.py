from collections import OrderedDict


def import_pattern_list():

    freq_pattern_list = []

    for idx in range(5):

        with open(f'patterns/pattern-{str(idx)}.txt', 'r') as f:
            raw_text = f.readlines()

        freq_dict = OrderedDict(
            (':'.join(pattern), int(support))
            for support, *pattern in (line.split() for line in raw_text)
        )

        freq_pattern_list.append(freq_dict)

    return freq_pattern_list


# this function will create max-pattern from pattern-i.txt
def generate_max_or_closed_pattern_output(*, is_closed_mining=False):

    topic_pattern = import_pattern_list()

    all_pattern = list()

    for topic_index, freq_dict in enumerate(topic_pattern):

        freq_list = list(freq_dict.keys())

        base_set = list(
            frozenset(pattern.split(':')) for pattern in freq_list
        )

        result_list = []

        for pattern in freq_list:

            is_max_or_closed_pattern = True

            for test_set in base_set:

                cur_pattern_set = set(pattern.split(':'))

                if cur_pattern_set < test_set:  # is proper subset

                    if is_closed_mining:

                        if freq_dict.get(pattern) \
                                == freq_dict.get(':'.join(sorted(test_set, key=int))):
                            is_max_or_closed_pattern = False
                            break
                    else:
                        is_max_or_closed_pattern = False
                        break

            if is_max_or_closed_pattern:
                result_list.append(pattern)

        result_list.sort(
            reverse=True,
            key=lambda t: freq_dict.get(t)
        )

        all_pattern.append(result_list)

        output = '\n'.join(
            ' '.join(
                [
                    str(freq_dict.get(pattern)),
                    pattern.replace(':', ' ')
                ]
            )
            for pattern in result_list
        )

        if is_closed_mining:
            with open(f'max/max-{str(topic_index)}.txt', 'w') as f:
                f.write(output)
        else:
            with open(f'closed/closed-{str(topic_index)}.txt', 'w') as f:
                f.write(output)

    return all_pattern
