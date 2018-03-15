from collections import OrderedDict, Counter
from itertools import combinations
from operator import itemgetter


def generate_candidate(freq_pattern_list, k):
    """generate and return candidate list for apriori algorithm."""

    if k <= 2:
        candidates = list(':'.join(pattern) for pattern in combinations(freq_pattern_list, 2))
        return candidates

    else:
        candidates = list(' '.join(pattern) for pattern in combinations(freq_pattern_list, 2))
        tmp_list = []
        for pattern in candidates:

            fst, snd = pattern.split(' ')

            *fst_rest, fst_tail = fst.split(':', maxsplit=1)
            *snd_rest, snd_tail = snd.split(':', maxsplit=1)

            if fst_rest == snd_rest:
                tmp_list.append(':'.join([fst, snd_tail]))

        tmp_list.sort()

        return tmp_list


def prune_data(min_sup, candidate_list, tmp_list):
    """prune the candidate list and return the frequent pattern dictionary."""
    tmp_dict = OrderedDict()

    for pattern in candidate_list:

        base_set = set(pattern.split(':'))

        sup = sum(
            1 for data in tmp_list
            if set(data.split()).issuperset(base_set)
        )

        if sup > min_sup:
            tmp_dict[pattern] = sup

    return OrderedDict(sorted(tmp_dict.items(), key=itemgetter(1), reverse=True))


def generate_pattern_output(freq_all_pattern, idx):
    """ create pattern-i.txt file."""
    flat_freq_dict = {
        ' '.join(
            sorted(
                pattern.split(':'),
                key=int
            )
        ): str(support)
        for freq_length_i in freq_all_pattern
        for pattern, support in freq_length_i.items()
    }

    flat_freq_dict = OrderedDict(
        sorted(flat_freq_dict.items(), key=lambda t: int(t[1]), reverse=True)
    )

    output = '\n'.join(
        ' '.join([support, pattern])
        for pattern, support in flat_freq_dict.items()
    )

    output = output.strip()

    with open(f'patterns/pattern-{str(idx)}.txt', 'w') as f:
        f.write(output)


def apriori(relative_min_support):
    """ Main function for the Apriori algorithm."""
    for idx in range(5):

        print(f'Mining file : topic-{str(idx)}.txt')

        freq_pattern_list = []

        with open(f'topic-{str(idx)}.txt', 'r') as f:
            raw_text = f.readlines()

        min_support = round(relative_min_support * len(raw_text))

        """Generate frequent pattern of length 1."""
        freq_dict = Counter(
            tokens
            for line in raw_text
            for tokens in line.split()
        )

        for pattern, support in list(freq_dict.items()):
            if support < min_support:
                del freq_dict[pattern]

        rnd = 2

        freq_dict = OrderedDict(sorted(freq_dict.items(), key=itemgetter(1), reverse=True))

        freq_pattern_list.append(freq_dict)

        while freq_dict:

            candidate_list = generate_candidate(
                list(freq_dict.keys()),
                rnd
            )

            if not candidate_list:
                break

            freq_dict = prune_data(min_support, candidate_list, raw_text)

            if freq_dict:
                freq_pattern_list.append(freq_dict)

            rnd += 1

        """generate output file."""
        generate_pattern_output(freq_pattern_list, idx)
