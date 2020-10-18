"""
Random maze generator. Taken from
https://gist.github.com/gmalmquist/2782000bd6b378831858
"""
import random
import sys

EMPTY = ' '
WALL = '#'
AGENT = 'o'
GOAL = 'x'

def adjacent(cell):
  i,j = cell
  for (y,x) in ((1,0), (0,1), (-1, 0), (0,-1)):
    yield (i+y, j+x), (i+2*y, j+2*x)

def generate(width, height, verbose=True):
  '''Generates a maze as a list of strings.

     :param width: the width of the maze, not including border walls.
     :param heihgt: height of the maze, not including border walls.
  '''
  # add 2 for border walls.

  width += 2 
  height += 2
  rows, cols = height, width

  maze = {}

  spaceCells = set()
  connected = set()
  walls = set()

  # Initialize with grid.
  for i in range(rows):
    for j in range(cols):
      if (i%2 == 1) and (j%2 == 1):
        maze[(i,j)] = EMPTY
      else:
        maze[(i,j)] = WALL 

  # Fill in border.
  for i in range(rows):
    maze[(i,0)] = WALL
    maze[(i,cols-1)] = WALL
  for j in range(cols):
    maze[(0,j)] = WALL
    maze[(rows-1,j)] = WALL

  for i in range(rows):
    for j in range(cols):
      if maze[(i,j)] == EMPTY:
        spaceCells.add((i,j))
      if maze[(i,j)] == WALL:
        walls.add((i,j))

  # Prim's algorithm to knock down walls.
  originalSize = len(spaceCells)
  connected.add((1,1))
  while len(connected) < len(spaceCells):
    doA, doB = None, None
    cns = list(connected)
    random.shuffle(cns)
    for (i,j) in cns:
      if doA is not None: break
      for A, B in adjacent((i,j)):
        if A not in walls: 
          continue
        if (B not in spaceCells) or (B in connected):
          continue
        doA, doB = A, B
        break
    A, B = doA, doB
    maze[A] = EMPTY
    walls.remove(A)
    spaceCells.add(A)
    connected.add(A)
    connected.add(B)
    if verbose:
      cs, ss = len(connected), len(spaceCells)
      cs += (originalSize - ss)
      ss += (originalSize - ss)
      if cs % 10 == 1:
        print('%s/%s cells connected ...' % (cs, ss), file=sys.stderr)

  # Insert character and goals.
  TL = (1,1)
  BR = (rows-2, cols-2)
  if rows % 2 == 0:
    BR = (BR[0]-1, BR[1])
  if cols % 2 == 0:
    BR = (BR[0], BR[1]-1)

  maze[TL] = AGENT
  maze[BR] = GOAL

  lines = []
  for i in range(rows):
    lines.append(''.join(maze[(i,j)] for j in range(cols)))

  return lines

if __name__ == '__main__':
  width = 21
  height = 21

  args = sys.argv[1:]
  if len(args) >= 1:
    width = int(args[0])
  if len(args) >= 2:
    height = int(args[1])

  if len(args) < 2:
    print('Use command-line args to specify width and height.', file=sys.stderr)
    print('  Odd numbers are suggested because of the walls.', file=sys.stderr)
  print('Non-maze text is printed to stderr, so you \n  can use > to pipe just the maze to a file.\n', file=sys.stderr)

  print('Generating %sx%s maze (not including border)...\n' % (width, height), file=sys.stderr)

  maze = generate(width, height)

  print('Done.\n', file=sys.stderr)

  print('\n'.join(maze))