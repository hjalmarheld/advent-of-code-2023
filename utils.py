def input(
        day: int,
        sample: bool=True,
        listify: bool=False,
        arrayify: bool=False):
    if sample:
        filename='sample'
    else:
        filename='input'
    with open(f'data/{day}/{filename}.txt') as file:
        data = file.read()

    if arrayify:
        data = [list(i) for i in data.splitlines()]
    elif listify:
        data = data.splitlines()

    return data


class matrix:
    @staticmethod
    def get_locs(
            M:list[list],
            s
        ):
        "find all loc"
        locs = []
        for x, row in enumerate(M):
            for y, char in enumerate(row):
                if char==s:
                    locs.append((x, y))
        return locs
    
    @staticmethod
    def transpose(
            M:list[list]
        ):
        return map(list, zip(*M))