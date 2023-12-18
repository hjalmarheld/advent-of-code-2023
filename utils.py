def input(
        day: int,
        sample: bool=True,
        listify: bool=False,
        arrayify: bool=False,
        matrixify: bool=False):
    if sample:
        filename='sample'
    else:
        filename='input'
    with open(f'data/{day}/{filename}.txt') as file:
        data = file.read()

    if arrayify:
        data = [list(i) for i in data.splitlines()]
    elif matrixify:
        data = Matrix([list(i) for i in data.splitlines()])
    elif listify:
        data = data.splitlines()

    return data


class matrix:
    """
    utility class for matrix methods
    """
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
    def replace(
            M:list[list],
            s1=...,
            s2=...
        ) -> list[list]:
        for x, y in matrix.get_locs(M, s1):
            M[x][y] = s2
        return M


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
    def count(
            M:list[list],
            s:...
        ) -> int:
        """
        return count of specific item in matrix
        """
        return len(matrix.get_locs(M, s))
    

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
        queue = set([loc])
        while queue:
            loc = queue.pop()
            M[loc[0]][loc[1]] = border
            neighbors = matrix.get_neighbors(M, loc, diag=diag)
            for neighbor in neighbors:
                if M[neighbor[0]][neighbor[1]] != border:
                    queue.add(neighbor)
        return M


class Matrix(list):
    """
    proper matrix class with some multiindex support
    """
    def __init__(self, M:list[list]):
        self.M = M
        self.index = -1

    def __getitem__(self, loc):
        """
        multiindexing to get items
        """
        assert isinstance(loc, (tuple, int, float))
        if type(loc)==tuple:
            return self.M[loc[0]][loc[1]]
        else:
            return self.M[int(loc)]
    
    def __setitem__(self, loc, item):
        """
        multiindexing to get items
        """
        assert isinstance(loc, (tuple, int, float))
        if type(loc)==tuple:
            self.M[loc[0]][loc[1]] = item
        else:
            self.M[int(loc)] = item
    
    def __iter__(self):
        yield from self.M
    
    def __str__(self):
        m = f'{len(self.M)}x{len(self.M[0])} matrix: \n'
        m += '\n'.join([''.join([str(i) for i in m]) for m in self.M])
        return m
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, s:...):
        return Matrix([[item+s for item in row] for row in self.M])
    
    def __mul__(self, s:...):
        return Matrix([[item*s for item in row] for row in self.M])
    
    def __sub__(self, s:...):
        return Matrix([[item*s for item in row] for row in self.M])
    
    def __pow__(self, s:...):
        return Matrix([[item**s for item in row] for row in self.M])

    def get_locs(self, s:...):
        return matrix.get_locs(self.M, s)
    
    def count(self, s:...):
        return matrix.count(self.M, s)
    
    def get_neighbors(self, loc:tuple[int], diag:bool):
        return matrix.get_neighbors(self.M, loc, diag)

    def replace(self, s1:..., s2:...):
        return Matrix(matrix.replace(self.M, s1, s2))
    
    def transpose(self):
        return Matrix(matrix.transpose(self.M))
    
    def turn(self):
        return Matrix(matrix.turn(self.M))
    
    def floodfill(self, loc:tuple[int], border:..., diag:bool=False):
        return Matrix(matrix.floodfill(self.M, loc, border, diag))
    
sample_matrix = Matrix([[1,1,1,1], [2,2,2,2], [3,3,3,3]])