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
            s:...
        ) -> list[tuple]:
        "find all locations of s"
        locs = []
        for x, row in enumerate(M):
            for y, char in enumerate(row):
                if char==s:
                    locs.append((x, y))
        return locs
    

    @staticmethod
    def transpose(
            M:list[list]
        ) -> list[list]:
        """
        transpose a list of list matrix
        """
        return list(map(list, zip(*M)))
    
    @staticmethod
    def turn(
            M:list[list]
        ) -> list[list]:
        """
        turn list of lists 90 deg clockwise
        """
        return list(map(list,(zip(*reversed(M)))))
    

    @staticmethod
    def get_neighbors(
            M:list[list],
            loc:tuple[int],
            diag:bool=False
        ):
        """
        get loc of neighbouring cells
        """
        # check point inside cell
        assert 0<=loc[0]<len(M) and 0<=loc[1]<len(M[1])
        neighbors = []
        # add horizontal and vertical neighbours
        x = [1, 0, -1, 0]
        y = [0, 1, 0, -1]
        for i in range(4):
            x_ = loc[0] + x[i]
            y_ = loc[1] + y[i]
            if 0<=x_<len(M) and 0<=y_<len(M[1]):
                neighbors.append((x_, y_))
        # add diagonal neighbors
        if diag:
            x = [1, 1, -1, -1]
            y = [1, -1, 1, -1]
            for i in range(4):
                x_ = loc[0] + x[i]
                y_ = loc[1] + y[i]
                if 0<=x_<len(M) and 0<=y_<len(M[1]):
                    neighbors.append((x_, y_))
        return neighbors
    
    
    @staticmethod
    def floodfill(
            M:list[list],
            loc:tuple[int],
            border:...=1,
            diag:bool=False
        ):
        """
        flood fill algo for list of lists matrix
        """
        assert M[loc[0]][loc[1]]!=border
        Q = set([loc])
        while Q:
            loc = Q.pop()
            M[loc[0]][loc[1]] = border
            neighbors = matrix.get_neighbors(M, loc, diag=diag)
            for neighbor in neighbors:
                if M[neighbor[0]][neighbor[1]] != border:
                    Q.add(neighbor)
        return M
    

#class Node()