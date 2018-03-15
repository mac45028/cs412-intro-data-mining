def separate_by_terms():
    """Read `word-assignments.dat` and generate topic-i.txt file from that data."""

    with open('result/word-assignments.dat', 'r') as f:
        data = f.readlines()

    list_result_str = [[] for _ in range(5)]

    for line in data:
        _, *tokens = line.split()

        list_temp_str = [[] for _ in range(5)]

        for x in tokens:
            idx, term = x.split(':')
            list_temp_str[int(term)].append(idx)

        for idx, list_ in enumerate(list_temp_str):
            if list_:
                list_result_str[idx].append(list_)

    for idx, list_ in enumerate(list_result_str):
        with open(f'topic-{idx}.txt', 'w') as f:
            for inner_list in list_:
                f.write(' '.join(inner_list)+'\n')
