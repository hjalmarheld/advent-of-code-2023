### Finished days:

[Day 1](#day-1)

[Day 2](#day-2)

[Day 3](#day-3)

[Day 4](#day-4)

[Day 5](#day-5)

[Day 6](#day-6)


```python
import re

def input(
        day: int,
        sample: bool=True,
        listify: bool=False):
    if sample:
        filename='sample'
    else:
        filename='input'
    with open(f'data/{day}/{filename}.txt') as file:
        data = file.read()

    if listify:
        data = data.splitlines()

    return data
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

print('question 1')
print(q1)
print('question 2')
print(q2)
```

    question 1
    4361
    question 2
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

print('question 1')
print(q1)
print('question 2')
print(q2)
```

    question 1
    23941
    question 2
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


print('question 1')
print(q1)
print('question 2')
print(q2)
```

    question 1
    910845529
    question 2
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

print('question 1')
print(q1)
print('question 2')
print(q2)
```

    question 1
    1084752
    question 2
    28228952

