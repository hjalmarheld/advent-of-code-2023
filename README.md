### Finished days:

[Day 1](#day-1)

[Day 2](#day-2)

[Day 3](#day-3)

[Day 4](#day-4)

[Day 5](#day-5)

[Day 6](#day-6)

[Day 7](#day-7)

[Day 8](#day-8)

[Day 9](#day-9)

[Day 10](#day-10)

[Day 11](#day-11)

[Day 12](#day-12)

[Day 13](#day-13)

[Day 14](#day-14)

[Day 15](#day-15)

[Day 16](#day-16)

[Day 17](#day-17)

[Day 18](#day-18)

[Day 19](#day-19)

[Day 20](#day-20)

[Day 21](#day-21)

[Day 22](#day-22)

[Day 23](#day-23)

[Day 23](#day-24)

[Day 25](#day-25)


```python
import re
import sys
import heapq
import random
from collections import Counter, deque, defaultdict
from itertools import combinations
from functools import cache
from math import lcm
from utils import *
```

# Day 1


```python
inp = input(1, listify=True, sample=False)

total = 0 
for line in inp:
    nums = [char for char in line if char.isnumeric()]
    total += int(nums[0]+nums[-1])

print('question 1')
print(total)

numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

total = 0 
for line in inp:
    for i, num in enumerate(numbers):
        line = line.replace(num, num+str(i+1)+num)
    nums = [char for char in line if char.isnumeric()]
    total += int(nums[0]+nums[-1])

print('question 2')
print(total)
```

    question 1
    53921
    question 2
    54676


# Day 2


```python
inp = input(2, listify=True, sample=True)

# parse input such that
# games = {
#   1: ['3 blue', '4 red', '1 red', '2 green'...]
#   ...
#   }
games = {
    int(game.split(':')[0].split()[1]):
    game.split(':')[1].replace('; ',', ').strip().split(', ')
    for game in inp}

# get max draw of each colour for each game such that
# games = {
#   1: {'blue': 6, 'red': 4, 'green': 2}
#   ...
#   }
for no, game in games.items():
    colours={}
    for draw in game:
        draw = draw.split()
        colours[draw[-1]] = max(
            int(draw[0]),
            colours.get(draw[-1], 0)
            )
    games[no]=colours

# compare each draw with top colors
top = {'red':12, 'green':13, 'blue':14}
possible_games = []
for no, max_draws in games.items():
    possible=True
    for colour, count in top.items():
        if max_draws[colour]>count:
            possible=False
            break
    if possible:
        possible_games.append(no)

print('question 1:')
print(sum(possible_games))

# multiply...
maximum=[]
for max_draws in games.values():
    total=1
    for count in max_draws.values():
        total*=count
    maximum.append(total)

print('question 2:')
print(sum(maximum))
```

    question 1:
    8
    question 2:
    2286


# Day 3


```python
inp = input(3, listify=True, sample=True)

class Num:
    def __init__(self, min_x, max_x, y, value, M):
        self.min_x = min_x
        self.max_x = max_x
        self.y = y
        self.value = int(value)
        self.M = M
        self.geared = False
        self.get_coords()
        self.check_value()

    # get coordinates for adjacent values
    def get_coords(self):
        _coords = []
        for i in range(self.min_x-1, self.max_x+1):
            _coords.append((i, self.y-1))
            _coords.append((i, self.y+1))
        _coords.append((self.min_x-1, self.y))
        _coords.append((self.max_x, self.y))
        self.coords = []
        for i in _coords:
            if (len(self.M[0]))>i[0]>=0 and len(self.M)>i[1]>=0:
                self.coords.append(i)

    # check adjacent if attached or geared
    def check_value(self):
        keep = False
        for coord in self.coords:
            if self.M[coord[1]][coord[0]]!='.':
                keep = True
            if self.M[coord[1]][coord[0]]=='*':
                self.geared=coord
        if not keep:
            self.value=0

q1 = 0
q2 = 0
seen_gears = {}

for y, row in enumerate(inp):
    for i in re.finditer('[0-9]+', row):
        num = Num(
            min_x=i.start(),
            max_x=i.end(),
            y=y,
            value=i.group(),
            M=inp
        )

        q1 += num.value
        
        if num.geared:
            if num.geared in seen_gears:
                q2 += num.value*seen_gears[num.geared]
            else:
                seen_gears[num.geared] = num.value

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    4361
    question 2:
    467835


# Day 4


```python
inp = input(4, listify=True, sample=False)
winning = [i.split(':')[1].split('|')[0].split() for i in inp]
have = [i.split(':')[1].split('|')[1].split() for i in inp]
wins = [len(set(h).intersection(set(w)))for h, w in zip(winning, have)]
q1 = sum([2**(i-1) for i in wins if i>0])

count = [1 for _ in wins]
for i, c in enumerate(count):
    w = wins[i]
    for j in range(i+1, i+1+w):
        if j < len(count):
            count[j]+=c
q2 = sum(count)

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    23941
    question 2:
    5571760


# Day 5


```python
inp = input(5, sample=False)
seeds = [int(i) for i in inp.split('\n\n')[0].split()[1:]]
recipes = inp.split('\n\n')[1:]
recipes = [i.splitlines()[1:] for i in recipes]

class Instruction:
    def __init__(self, lines):
        self.maps = [[int(i) for i in line.split()] for line in lines]

    def match(self, seed):
        for (destination, source, size) in self.maps:
            if source<=seed<source+size:
                return seed+destination-source
        return seed

    def match2(self, seeds):
        answers = []
        for (destination, source, size) in self.maps:
            source_end = source+size
            new_seeds = []
            
            while seeds:
                start, end = seeds.pop()
                
                before_range = (start, min(end, source))
                if before_range[1]>before_range[0]:
                    new_seeds.append(before_range)

                after_range = (max(source_end, start), end)
                if after_range[1]>after_range[0]:
                    new_seeds.append(after_range)

                intersection = (max(start, source), min(source_end, end))
                if intersection[1]>intersection[0]:
                    answers.append((intersection[0]-source+destination, intersection[1]-source+destination))

            seeds = new_seeds
        return answers+new_seeds

instructions  = [Instruction(recipe) for recipe in recipes]

destinations = []
for seed in seeds:
    dest = seed
    for instruction in instructions:
        dest = instruction.match(dest)
    destinations.append(dest)
q1 = min(destinations)


destinations = []
for start, size in zip(seeds[::2], seeds[1::2]):
  seed_range = [(start, start+size)] 
  for instruction in instructions:
    seed_range = instruction.match2(seed_range)
  destinations.append(min(seed_range)[0])
q2 = min(destinations)

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    910845529
    question 2:
    77435348


# Day 6


```python
inp = input(6, listify=True, sample=False)
races = [[int(k) for k in re.findall('\d+', i )] for i in  inp]
ways = 1
for time, dist in zip(races[0], races[1]):
    _ways = 0
    for i in range(1, time):
        if (time-i)*i>dist:
            _ways += 1
    ways*=_ways

q1 = ways

time, dist = [int(i.split(':')[1].replace(' ','')) for i in inp]
# i**2 - time*i + dist == 0 =>
a = time/2 + ((time/2)**2 - dist)**.5
b = time/2 - ((time/2)**2 - dist)**.5

q2 = int(a)-int(b)

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    1084752
    question 2:
    28228952


# Day 7


```python
inp = input(7, listify=True, sample=True)
inp = {i.split()[0]:int(i.split()[1]) for i in inp}

def pair_val(key):
    hand = Counter(key).values()
    if 5 in hand:
        pair = 6
    elif 4 in hand:
        pair = 5
    elif 3 in hand and 2 in hand:
        pair = 4
    elif 3 in hand:
        pair = 3
    elif list(hand).count(2)==2:
        pair = 2
    elif 2 in hand:
        pair = 1
    else:
        pair = 0
    return pair*10**11

cards = [
    '2', '3', '4',
    '5', '6', '7',
    '8', '9', 'T',
    'J', 'Q', 'K',
    'A']

def order_val(key):
    order = ''
    for k in key:
        val = str(cards.index(k))
        if len(val)==1:
            order+='0'+val
        else:
            order+=val
    return int(order)


q1 = 0
values = {}
for key in inp:
    pair = pair_val(key)
    order = order_val(key)
    values[pair+order]=key
for i, key in enumerate(sorted(values.keys())):
    q1 += (i+1)*inp[values[key]]

q2 = 0
values = {}
cards = [
    'J', '2', '3',
    '4', '5', '6',
    '7', '8', '9',
    'T', 'Q', 'K',
    'A']
for key in inp:
    if 'J' in key and len(set(key))>1:
        key2 = key.replace(
            'J',
            Counter(key.replace('J','')).most_common(1)[0][0]
            )
    else:
        key2 = key
    pair = pair_val(key2)
    order = order_val(key)
    values[pair+order]=key
for i, key in enumerate(sorted(values.keys())):
    q2 += (i+1)*inp[values[key]]

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    6440
    question 2:
    5905


# Day 8


```python
directions, nodes = input(8, sample=False).split('\n\n')

graph = {}
for node in nodes.splitlines():
    name, left, right = re.findall('[0-9A-Z]+', node)
    graph[name] = {}
    graph[name]['L'] = left
    graph[name]['R'] = right

position, goal, steps = 'AAA', 'ZZZ', 0
while position!=goal:
    position = graph[position][directions[steps%len(directions)]]
    steps+=1

q1 = steps

start_nodes, end_nodes, steps = set(), set(), []
for node in graph:
    if node[-1]=='A':
        start_nodes.add(node)
    if node[-1]=='Z':
        end_nodes.add(node)

for node in start_nodes:
    position = node
    _steps = 0
    while position not in end_nodes:
        position = graph[position][directions[_steps%len(directions)]]
        _steps+=1
    steps.append(_steps)

q2 = lcm(*steps)

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    20659
    question 2:
    15690466351717


# Day 9


```python
inp = input(9, listify=True, sample=False)
inp = [list(map(int, i.split())) for i in inp]

q1, q2 = 0, 0
for i in inp: 
    levels = [i]
    while set(levels[-1])!=set([0]):
        level = []
        for i in range(len(levels[-1])-1):
            level.append(levels[-1][i+1]-levels[-1][i])
        levels.append(level)
    
    next_val = sum([l[-1] for l in levels])
    q1+=next_val

    ls = [l[0] for l in levels]
    next_val2 = 0
    for l in ls[::-1]:
        next_val2 = l-next_val2
    q2+=next_val2

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    2101499000
    question 2:
    1089


# Day 10


```python
M = input(10, sample=False, arrayify=True)

for x in range(len(M)):
  for y in range(len(M[0])):
    if M[x][y]=='S':
      x1, y1 = x, y
      up = (M[x-1][y] in ['|','7','F'])
      right = (M[x][y+1] in ['-','7','J'])
      down = (M[x+1][y] in ['|','L','J'])
      left = (M[x][y-1] in ['-','L','F'])

      if up and down:
        M[x][y]='|'
        d1 = 0
      elif up and right:
        M[x][y]='L'
        d1 = 0
      elif up and left:
        M[x][y]='J'
        d1 = 0
      elif down and right:
        M[x][y]='F'
        d1 = 2
      elif down and left:
        M[x][y]='7'
        d1 = 2
      elif left and right:
        M[x][y]='-'
        d1 = 1

x, y, d = x1, y1, d1
dist = 0
while True:
  dist += 1
  x += [-1,0,1,0][d]
  y += [0,1,0,-1][d]
  if (x,y) == (x1,y1):
    q1=dist//2
    break
  elif M[x][y] in ['-', '|']:
    pass
  elif M[x][y]=='L':
    if d==2:
      d = 1
    else:
      d = 0
  elif M[x][y]=='J':
    if d==1:
      d = 0
    else:
      d = 3
  elif M[x][y]=='7':
    if d==0:
      d = 3
    else:
      d = 2
  elif M[x][y]=='F':
    if d==0:
      d = 1
    else:
      d = 2
  

R2 = 3*len(M)
C2 = 3*len(M[0])
M2 = [['.' for _ in range(C2)] for _ in range(R2)]
for x in range(len(M)):
  for y in range(len(M[0])):
    if M[x][y]=='|':
      M2[3*x+0][3*y+1] = '#'
      M2[3*x+1][3*y+1] = '#'
      M2[3*x+2][3*y+1] = '#'
    elif M[x][y]=='-':
      M2[3*x+1][3*y+0] = '#'
      M2[3*x+1][3*y+1] = '#'
      M2[3*x+1][3*y+2] = '#'
    elif M[x][y]=='7':
      M2[3*x+1][3*y+0] = '#'
      M2[3*x+1][3*y+1] = '#'
      M2[3*x+2][3*y+1] = '#'
    elif M[x][y]=='F':
      M2[3*x+2][3*y+1] = '#'
      M2[3*x+1][3*y+1] = '#'
      M2[3*x+1][3*y+2] = '#'
    elif M[x][y]=='J':
      M2[3*x+1][3*y+0] = '#'
      M2[3*x+1][3*y+1] = '#'
      M2[3*x+0][3*y+1] = '#'
    elif M[x][y]=='L':
      M2[3*x+0][3*y+1] = '#'
      M2[3*x+1][3*y+1] = '#'
      M2[3*x+1][3*y+2] = '#'


to_visit = deque()
seen = set()
# up right down left
for x in range(R2):
  to_visit.append((x,0))
  to_visit.append((x,C2-1))
for y in range(C2):
  to_visit.append((0,y))
  to_visit.append((R2-1,y))
while to_visit:
  x,y = to_visit.popleft()
  if (x,y) in seen or not (0<=x<R2 and 0<=y<C2):
    continue
  seen.add((x,y))
  if M2[x][y]=='#':
    continue
  for d in range(4):
    to_visit.append((x+[-1,0,1,0][d],y+[0,1,0,-1][d]))

q2 = 0
for x in range(len(M)):
  for y in range(len(M[0])):
    visited = False
    for xx in [0,1,2]:
      for yy in [0,1,2]:
        if (3*x+xx,3*y+yy) in seen:
          visited = True
    if not visited:
      q2 += 1

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    6867
    question 2:
    595


# Day 11


```python
inp = input(11, matrixify=True, sample=False)

def expand_rows(M):
    M2 = []
    for row in M:
        M2.append(row)
        if not '#' in row:
            M2.append(row)
    return Matrix(M2)

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

# expand rows, flip, expand cols, flip again
future = expand_rows(expand_rows(inp).transpose()).transpose()

# get star locations
locs_start = inp.get_locs('#')
locs_future = future.get_locs('#')

# calculate manhattan distances and multiply with delta for q2
q1 = sum([manhattan(a, b) for a, b in combinations(locs_future, 2)])
q2 = sum([manhattan(a, b) for a, b in combinations(locs_start, 2)])
q2 = q2+(q1-q2)*(10**6-1)

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    9563821
    question 2:
    827009909817


# Day 12


```python
inp = [i.split() for i in input(12, listify=True, sample=False)]
rows = [(row, tuple(map(int, size.split(",")))) for row, size in inp]
@cache
def get_num(row, sizes, done=0):
    # if not enough left return
    if len(row)+done<sum(sizes):
        return False
    # if end of row return if finished
    if not row:
        return not sizes and not done
    num = 0
    # create branching if unknown
    chars = [".", "#"] if row[0] == "?" else row[0]
    for c in chars:
        # if broken keep going on group
        if c == "#":
            num += get_num(row[1:], sizes, done + 1)
        # if not then check if previous group done
        else:
            if done:
                if sizes and sizes[0] == done:
                    num += get_num(row[1:], sizes[1:])
            else:
                num += get_num(row[1:], sizes)
    return num

q1=sum(get_num(group + ".", sizes) for group, sizes in rows)
q2=sum(get_num("?".join([group] * 5) + ".", sizes * 5) for group, sizes in rows)
print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    7705
    question 2:
    50338344809230


# Day 13


```python
def check_reflection(M, smudges=0):
    for i in range(1,len(M)):
        if type(M[0])==list:
            M = [''.join(i) for i in M]
        a = ''.join(M[max(0, 2*i-len(M)):i])
        b = ''.join(M[i:2*i][::-1])
        if sum([1 for a_, b_ in zip(a, b) if a_!=b_])==smudges:
            return i
    return 0

q1 = q2 = 0
for scan in [i.splitlines() for i in input(13, sample=False).split('\n\n')]:
    q1 += max(
        100*check_reflection(scan, 0),
        check_reflection(matrix.transpose(scan), 0))
    q2 += max(
        100*check_reflection(scan, 1),
        check_reflection(matrix.transpose(scan), 1))

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    37561
    question 2:
    31108


# Day 14


```python
M = input(14, arrayify=True, sample=False)

def tilt(M):
    round = deque(matrix.get_locs(M, 'O'))
    while round:
        x, y = round.popleft()
        if x>0 and M[x-1][y]=='.':
            M[x-1][y], M[x][y] = 'O', '.'
            round.append((x-1, y))
    return M

q1 = sum([len(M) - loc[0] for loc in matrix.get_locs(tilt(M), 'O')])

i = 1
seen = {}
while i <= 1000000000:
    for spin in range(4):
        rocks = tuple(matrix.get_locs(tilt(M), 'O'))
        if rocks in seen:
            old_i = seen[rocks]
            skip = (1000000000 - i) // (i - old_i)
            i += (i - old_i) * skip
            seen = {}
        else:
            seen[rocks] = i
        M = matrix.turn(tilt(M))
    i+=1
q2 = sum([len(M) - loc[0] for loc in matrix.get_locs(M, 'O')])

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    106990
    question 2:
    100531


# Day 15


```python
inp = input(15, sample=False).split(',')

def hashmap(code):
    val = 0
    for char in code:
        val = ((val+ord(char))*17)%256
    return val

q1 = sum([hashmap(code) for code in inp])

boxes = [[] for _ in range(256)]
for code in inp:
    name = re.findall('[a-z]+', code)[0]
    box = boxes[hashmap(name)]
    if code[-1]=='-':
        box = [(n,v) for (n,v) in box if n!=name]
    elif code[-2]=='=':
        len_ = int(code[-1])
        if name in [n for (n,v) in box]:
            box = [(n, len_ if name==n else v) for (n,v) in box]
        else:
            box.append((name, len_))
    boxes[hashmap(name)] = box 

q2 = 0
for i,box in enumerate(boxes):
  for j,(n,v) in enumerate(box):
    q2 += (i+1)*(j+1)*v

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    513643
    question 2:
    265345


# Day 16


```python
inp = input(16, matrixify=True, sample=False)

def walk(r, c, d):
    '''
    position and direction after 
    taking one step
    '''
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]
    return (r+dr[d], c+dc[d], d)

def moves(r, c, d, M):
    positions = [(r, c, d)]
    visited = set()
    done = set()
    while True:
        new_positions = []
        if not positions:
            break
        for r, c, d in positions:
            if 0<=r<len(M) and 0<=c<len(M[0]):
                visited.add((r, c))
                if (r, c, d) in done:
                    continue
                done.add((r,c,d))
                char = M[r,c]
                if char=='.':
                    new_positions.append(
                        walk(r,c,d))
                elif char=='/':
                    new_positions.append(
                        walk(r,c,{0:1, 1:0, 2:3, 3:2}[d]))
                elif char=='\\':
                    new_positions.append(
                        walk(r,c,{0:3, 1:2, 2:1, 3:0}[d]))
                elif char=='|':
                    if d in [0,2]:
                        new_positions.append(
                            walk(r,c,d))
                    else:
                        new_positions.append(
                            walk(r, c, 0))
                        new_positions.append(
                            walk(r, c, 2))
                elif char=='-':
                    if d in [1,3]:
                        new_positions.append(
                            walk(r,c,d))
                    else:
                        new_positions.append(
                            walk(r, c, 1))
                        new_positions.append(
                            walk(r, c, 3))
        positions = new_positions
    return len(visited)

q1 = moves(0,0,1, inp)

q2 = 0
rows = len(inp)
cols = len(inp[0])
for r in range(rows):
    q2 = max(moves(r, 0, 1, inp), q2)
    q2 = max(moves(r, cols-1, 3, inp), q2)
for c in range(cols):
    q2 = max(moves(0, c, 2, inp), q2)
    q2 = max(moves(rows-1, c, 0, inp), q2)

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    7111
    question 2:
    7831


# Day 17


```python
inp = input(17, matrixify=True, sample=False)
rows, cols = inp.shape

def traverse(part2=False):
    queue = [(0, 0, 0, -1, -1)]
    costs = {}
    while queue:
        dist, r, c, dir, steps = heapq.heappop(queue)
        if (r, c, dir, steps) in costs:
            continue
        else:
            costs[(r, c, dir, steps)] = dist
        for i in range(4):
            rr, cc, new_dir = walk(r, c, i)
            new_steps = 1 if new_dir != dir else steps + 1
            isnt_reverse = (new_dir + 2) % 4 != dir
            p1 = new_steps <= 3
            p2 = new_steps <= 10 and (new_dir == dir or steps >= 4 or steps == -1)
            valid = p2 if part2 else p1

            if 0 <= rr < rows and 0 <= cc < cols and isnt_reverse and valid:
                cost = int(inp[rr, cc])
                if (rr, cc, new_dir, new_steps) not in costs:
                    heapq.heappush(queue, (dist + cost, rr, cc, new_dir, new_steps))
    ans = 10**9
    for (r, c, dir, steps), v in costs.items():
        if r == rows - 1 and c == cols - 1 and (steps >= 4 or not part2):
            ans = min(ans, v)
    return ans

q1 = traverse()
q2 = traverse(True)
print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    1110
    question 2:
    1294


# Day 18


```python
def walk(position, d, steps=1):
    r, c = position
    dr = {'D':1, 'R':0, 'U':-1, 'L':0}[d]*int(steps)
    dc = {'D':0, 'R':1, 'U':0, 'L':-1}[d]*int(steps)
    return (r+dr, c+dc)

def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
        for ((x0, y0), (x1, y1))
        in zip(p, p[1:] + [p[0]])))

position = position2 = (0, 0)
seen1, seen2 = [], []
walked1 = walked2 = 0
for row in input(18, listify=True, sample=False):
    dir, steps, colour = row.split()

    seen1.append(position:=walk(
        position,
        dir,
        steps))
    walked1 += int(steps)

    seen2.append(position2:=walk(
        position2,
        'RDLU'[int(colour[-2])],
        steps:=int(colour[1:-2].replace('#', '0'), 16)))
    walked2 += steps

q1 = area(seen1)+walked1/2+1
q2 = area(seen2)+walked2/2+1

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    42317.0
    question 2:
    83605563360288.0


# Day 19


```python
def cond_parser(conds):
    conditions = {}
    for cond in conds:
        name, conds = cond.split('{')
        conds = conds[:-1].split(',')
        parsed = []
        for cond in conds[:-1]:
            cond, dest = cond.split(':')
            key, num = cond.split(sep:='>' if '>' in cond else '<')
            parsed.append({'var':key, 'num':int(num), 'sep':sep, 'dest':dest})
        parsed.append(conds[-1])
        conditions[name] = parsed
    return conditions

def part_parser(parts):
    parts_parsed = []
    for part in parts:
        items = {}
        items['total'] = 0
        for item in part[1:-1].split(','):
            name, value = item.split('=')
            items[name] = int(value)
            items['total'] += int(value)
        parts_parsed.append(items)
    return parts_parsed

def passing(conditions, parts):
    passing = 0
    for part in parts:
        flow = 'in'
        while True:
            # see if finished all rules of a flow
            if flow in ('A', 'R'):
                if flow=='A':
                    passing += part['total']
                break
            cond = conditions[flow]
            # else try rules
            for rule in cond[:-1]:
                # see if passing a rule
                if passed:= eval(f'{part[rule["var"]]} {rule["sep"]} {rule["num"]}'):
                    break
            # return destination if passed
            if passed:
                flow = rule['dest']
            else:
                flow = cond[-1]
    return passing

def new_ranges(rule, ranges):
    # create ranges inside and outside of conditions
    if rule['sep'] == ">":
        var_in = (rule['num']+1, ranges[rule['var']][1])
        var_out = (ranges[rule['var']][0], rule['num'])
    else:
        var_in = (ranges[rule['var']][0], rule['num']-1)
        var_out = (rule['num'], ranges[rule['var']][1])
    # avoid changing mutable objects...
    inside, outside = ranges.copy(), ranges.copy()
    inside[rule['var']], outside[rule['var']] = var_in, var_out
    return inside, outside

def combos(flow, ranges, conditions):
    total = 0
    # end of recursion
    if flow == "A":
        total = 1
        for low, high in ranges.values():
            total*=(high-low+1)
        return total
    elif flow == "R":
        return 0
    # get rules for a flow
    for rule in conditions[flow][:-1]:
        # get range passing and outside of rule
        inside, outside = new_ranges(rule, ranges)
        # send succesful to destination if inside
        total += combos(rule['dest'], inside, conditions)
        # save outside to try next condition
        ranges = outside
    # get flow for ranges missing all conditios
    total += combos(conditions[flow][-1], outside, conditions)
    return total

conds, parts = (i.splitlines() for i in input(19, sample=True).split('\n\n'))
conditions = cond_parser(conds)
parts = part_parser(parts)
q1 = passing(conditions, parts)
q2 = combos('in', {var:(1,4000) for var in 'xmas'}, conditions=conditions)
print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    19114
    question 2:
    167409079868000


# Day 20


```python
class FlipFlop:
    def __init__(self, recipients, name):
        self.name = name
        self.state = 'off'
        self.recipients = recipients

    def actions(self, signal, name):
        if signal=='h':
            pass
        else:
            self.state= 'on' if self.state=='off' else 'off'
            if self.state =='on':
                return ('h', self.recipients, self.name)
            else:
                return ('l', self.recipients, self.name)
            
class Conjunction:
    def __init__(self, recipients, name):
        self.name = name
        self.states = {}
        self.recipients = recipients

    def actions(self, signal, name):
        self.states[name] = signal
        if set(self.states.values())==set('h'):
            return ('l', self.recipients, self.name)
        else:
            return ('h', self.recipients, self.name)
        
class Broadcast:
    def __init__(self, recipients, name):
        self.name = name
        self.recipients = recipients
        self.name = name

    def actions(self, signal, name):
        return (signal, self.recipients, self.name)

    
def parser(input):
    # create dict of all components
    components = {}
    for component in input:
        component, recipients = component.split(' -> ')
        recipients = recipients.split(', ')
        if '%' in component:
            components[component[1:]] = FlipFlop(recipients, component[1:])
        elif '&' in component:
            components[component[1:]] = Conjunction(recipients, component[1:])
        else:
            components[component] = Broadcast(recipients, component)
    # set states for all Conjunctions
    for component in components.values():
        for recipient in component.recipients:
            try:
                if type(components[recipient])==Conjunction:
                    components[recipient].states[component.name]='l'
            except:
                pass
    return components

inp = input(20, listify=True, sample=False)    
components = parser(inp)

l = 0
h = 0
for _ in range(1000):
    button = ('l', ['broadcaster'], 'button')
    queue = [button]
    while queue:
        signal, recipients, name = queue.pop(0)
        for recipient in recipients:
            if signal=='h':
                h+=1
            else:
                l+=1
            if recipient in components:
                reponse = components[recipient].actions(signal, name)
                if reponse: queue.append(reponse)

components = parser(inp)

# find component giving signal to rx
for component in components.values():
    if 'rx' in component.recipients:
        comp = component

min_turns = {}
done = False
i = 1
while True:
    button = ('l', ['broadcaster'], 'button')
    queue = [button]
    while queue:
        signal, recipients, name = queue.pop(0)
        for recipient in recipients:
            if recipient in components:
                reponse = components[recipient].actions(signal, name)
                if reponse: queue.append(reponse)
            if recipient==comp.name and signal=='h':
                if name not in min_turns:
                    min_turns[name]=i
                    if len(min_turns)==len(comp.states):
                        done = True
    if done:
        break
    else:
        i+=1

q1 = h*l
q2 = lcm(*list(min_turns.values()))
print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    856482136
    question 2:
    224046542165867


# Day 21


```python
M = input(21, arrayify=True, sample=False)
size = len(M)

# make matrix 5x bigger to stay inside
M = Matrix(5*[row*5 for row in M])
M = M.replace('S','.')

# set up start and queues
start = (len(M)//2, len(M[0])//2)
visited = set()
queue = [start]

# save values for interpolation
y = []
x = [65, 65+size, 65+2*size]

for i in range(1, x[-1]+1):
    to_visit = set()
    while queue:
        loc = queue.pop()
        visited.add(loc) 
        neighbours = M.get_neighbors(loc, diag=False)
        for n in neighbours:
            if M[n] in ('.','S'):
                to_visit.add(n)
    queue = to_visit
    if i in x:
        y.append(len(to_visit))
    if i==64:
        q1 = len(to_visit)

# quadratic interpolation from
# f(0) = 65 steps, f(1) = 65+131 steps and f(2) = 65+2*131 
x0, x1, x2 = 0, 1, 2
diff10 = (y[1]-y[0])/(x1-x0)
diff21 = (y[2]-y[1])/(x2-x1)
a, b, c = (diff21-diff10)/(x2-x0), diff10, y[0]
x = 26501365//size
q2 = a*(x-x1)*(x-x0)+b*(x-x0)+c
print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    3758
    question 2:
    621494544278648.0


# Day 22


```python
inp = input(22, listify=True, sample=True)

class Brick:
    def __init__(self, line, name):
        x0, y0, z0, x1, y1, z1 = map(int, re.findall(r'\d+', line))
        self.h = z1 - z0 + 1
        self.z = z0
        self.name = name
        self.cubes = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

# parse all bricks
bricks = sorted([Brick(line, i) for i, line in enumerate(inp)], key=lambda x: x.z)

# get max dimensions for grid
xmax, ymax, zmax = 0, 0, 0
for line in inp:
    x0, y0, z0, x1, y1, z1 = map(int, re.findall(r'\d+', line))
    xmax = max(xmax, x1)
    ymax = max(ymax, y1)
    zmax = max(zmax, z1)

# create stack which is a list of matrixes
# q1 - return stack
# q2 - return height of all bricks
def stack_bricks(bricks, q1=True):
    stack = [Matrix([['.' for _ in range(ymax + 1)] for _ in range(xmax + 1)]) for _ in range(zmax)]
    all_heights = {}
    for brick in bricks:
        height = len(stack) - 1
        while height > 0:
            level = stack[height - 1]
            if any(level[cube] != '.' for cube in brick.cubes):
                break
            height -= 1
        all_heights[brick.name] = height
        for h in range(height, height + brick.h):
            for cube in brick.cubes:
                stack[h][cube] = brick.name
    return stack if q1 else all_heights

# create stack
stack = stack_bricks(bricks)
q1 = 0
for level, under in zip(stack[::-1], stack[-2::-1]):
    # get bricks on level and level below
    chars = level.unique()
    chars_under = under.unique()
    supps = set()
    # get bricks which are single-handedly supporting another brick
    for char in chars:
        if char == '.':
            continue
        supp = {under[loc] for loc in level.get_locs(char) if under[loc] != '.'}
        if len(supp) == 1:
            supps |= supp
    # add all bricks which were below and found to not be carrying
    q1 += sum(char != '.' and char not in supps for char in chars_under)

# get height of all original bricks
starting = stack_bricks(bricks, False)
q2 = 0
for i in range(len(bricks)):
    # create stack without a single bricks
    total_height = stack_bricks(bricks[:i] + bricks[i + 1:], False)
    # check how many bricks have moved
    for j in starting.keys():
        if j not in total_height:
            total_height[j] = starting[j]
        if starting[j] > total_height[j]:
            q2 += 1

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    5
    question 2:
    7


# Day 23


```python
def create_graph(input_map):
    graph = {}
    slopes = {'^':'U', '>':'R', 'v':'D', '<':'L'}
    for r in range(len(input_map)):
        for c in range(len(input_map[0])):
            if input_map[r, c] == '.':
                graph[(r, c)] = set()
                neighbors = input_map.get_neighbors((r, c), direction=True)
                for r_, c_, dir_ in neighbors:
                    val = input_map[r_, c_]
                    if val == '.':
                        graph[(r, c)].add((r_, c_))
                    elif val in slopes and slopes[val] == dir_:
                        graph[(r, c)].add((r_, c_))
            elif input_map[(r, c)] != '#':
                graph[(r, c)] = set()
                neighbors = input_map.get_neighbors((r, c), direction=True) 
                for r_, c_, dir_ in neighbors:
                    if slopes[input_map[(r, c)]] == dir_ and input_map[r_, c_] != '#':
                        graph[(r, c)].add((r_, c_))
    return graph

def condense_graph(graph, part2=False):
    new_graph = {node: {neighbor: 1 for neighbor in neighbors} for node, neighbors in graph.items()}
    if part2:
        while True:
            found = 0
            new_graph2 = new_graph.copy()
            for node, neighbours in new_graph.items():
                try:
                    if len(neighbours) == 2:
                        found += 1
                        n1, n2 = neighbours
                        new_dist = new_graph2[node][n1] + new_graph2[node][n2]
                        del new_graph2[node], new_graph2[n1][node], new_graph2[n2][node]
                        new_graph2[n1][n2] = new_graph2[n2][n1] = new_dist
                    else:
                        continue
                except:
                    continue
            new_graph = new_graph2
            if found == 0:
                break
    return new_graph

def find_all_paths(graph, start, end, path=[], length=0, visited=None):
    if visited is None:
        visited = set()
    path = path + [start]
    visited.add(start)
    if start == end:
        return [(path, length)]
    paths = []
    for neighbor, dist in graph[start].items():
        if neighbor not in visited:
            extended_paths = find_all_paths(graph, neighbor, end, path, length + dist, visited.copy())
            paths.extend(extended_paths)
    return paths

inp = input(23, matrixify=True, sample=False)

graph = condense_graph(create_graph(inp))
start, end = (0, 1), (140, 139)
paths = find_all_paths(graph, start, end)
q1 = max([path[1] for path in paths])

for char in '<>v^':
    inp.replace(char, '.')
graph = condense_graph(create_graph(inp), True)
paths = find_all_paths(graph, start, end)
q2 = max([path[1] for path in paths])

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    2186
    question 2:
    6802


# Day 24


```python
inp = input(24, listify=True, sample=False)
inp = [tuple(map(int, re.findall('[-0-9]+', i))) for i in inp]

lower=200000000000000
upper=400000000000000

q1 = 0
for a, b in combinations(inp, 2):
    x1, y1, _, dx1, dy1, _ = a
    x2, y2, _, dx2, dy2, _ = b
    a1 = (dy1/dx1)
    a2 = (dy2/dx2)
    if a1 == a2:
        continue
    b1 = y1 - (a1*x1)
    b2 = y2 - (a2*x2)
    x = (b2-b1)/(a1-a2)
    y = a1*x + b1
    if (x-x1)/dx1<0 or (x-x2)/dx2<0:
        continue
    if lower<=x<=upper and lower<=y<=upper:
        q1 += 1

xspeeds = defaultdict(list)
yspeeds = defaultdict(list)
zspeeds = defaultdict(list)
for i in inp:
    x, y, z, dx, dy, dz = i
    xspeeds[dx].append(x)
    yspeeds[dy].append(y)
    zspeeds[dz].append(z)

hail = []
for speed in [xspeeds, yspeeds, zspeeds]:
    speed = {key:sorted(value) for key, value in speed.items() if len(value)>1}
    speed = {key:(value[-1]-value[0]) for key, value in speed.items()}
    for i in range(-500, 500):
        correct = True
        for key, value in speed.items():
            if i==key or i==-key:
                continue
            if value%(key-i)!=0:
                correct = False
                break
        if correct:
            hail.append(i)

dx, dy, dz = hail
x0, y0, z0, dx0, dx0, dz0 = inp[0]
x1, y1, z1, dx1, dy1, dz1 = inp[1]
a0 = (dx0-dy)/(dx0-dx)
a1 = (dy1-dy)/(dx1-dx)
c0 = y0 - (a0*x0)
c1 = y1 - (a1*x1)
x = int((c1-c0)/(a0-a1))
y = int(a0*x + c0)
t = (x - x0)//(dx0-dx)
z = z0 + (dz0 - dz)*t
q2 = sum([x, y, z])

print(f'question 1:\n{q1}\nquestion 2:\n{q2}')
```

    question 1:
    17244
    question 2:
    1025019997186820


# Day 25


```python
def find_frequent(graph, n_frequent=3, iterations=10_000):
    frequency = defaultdict(int)
    for _ in range(iterations):
        a = random.choice(list(graph.keys()))
        b = random.choice(list(graph.keys()))
        path = dijkstra(graph, a, b)
        edges = [(a, b) for a,b in zip(path[:-1],path[1:])]
        for edge in edges:
            frequency[edge]+=1
    frequency = {value:key for key, value in frequency.items()}
    edges = []
    seen = set()
    while len(edges)<n_frequent:
        for freq in sorted(frequency.keys(), reverse=True):
            edge = frequency[freq]
            a, b = edge
            if a in seen or b in seen:
                continue
            else:
                seen.add(a)
                seen.add(b)
                edges.append(edge)
                break    
    return edges

def count_reachable_nodes(graph, start_node):
    visited = set()
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    dfs(start_node)
    return len(visited)

inp = [re.findall('\w+',i) for i in input(25, listify=True, sample=False)]

# parse graph from input
graph = defaultdict(list)
for line in inp:
    for node in line[1:]:
        graph[line[0]].append(node)
        graph[node].append(line[0])

# find top 3 most used edges when traversing
# graph from random nodes
edges = find_frequent(graph)

# re-create graph without these edges
graph = defaultdict(list)
for line in inp:
    for node in line[1:]:
        if (line[0], node) in edges or (node, line[0]) in edges:
            continue
        else:
            graph[line[0]].append(node)
            graph[node].append(line[0])

# see number of reachable nodes from both sides
# of the first removed edge
nodes_a = count_reachable_nodes(graph, edges[0][0]) 
nodes_b = count_reachable_nodes(graph, edges[0][1])
q1 = nodes_a * nodes_b

print(f'question 1:\n{q1}')
```

    question 1:
    571753

