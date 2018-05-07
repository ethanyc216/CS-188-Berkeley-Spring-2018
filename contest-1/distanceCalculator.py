"""
This file contains a Distancer object which computes and 
caches the shortest path between any two points in the maze. 

Example:
distancer = Distancer(gameState.data.layout)
distancer.getDistance( (1,1), (10,10) )
"""

import sys, time, random

class Distancer:
  def __init__(self, layout, default = 10000):
    """
    Initialize with Distancer(layout).  Changing default is unnecessary.    
    """
    self._distances = {}
    self.default = default
    self.initialWalls = layout.walls
    self.dc = DistanceCalculator(layout, self, default)

  def getMazeDistances(self):
    self.dc.run()
    
  def getDistance(self, pos1, pos2, walls):
    """
    The getDistance function is the only one you'll need after you create the object.
    """
    if self._distances == {}:
      return manhattanDistance(pos1, pos2)
    if isInt(pos1) and isInt(pos2):
      pos1 = (int(pos1[0]), int(pos1[1]))
      pos2 = (int(pos2[0]), int(pos2[1]))
      return self.getDistanceOnGrid(pos1, pos2, walls)
    pos1Grids = getGrids2D(pos1)
    pos2Grids = getGrids2D(pos2)
    bestDistance = self.default
    for pos1Snap, snap1Distance in pos1Grids:
      for pos2Snap, snap2Distance in pos2Grids:
        gridDistance = self.getDistanceOnGrid(pos1Snap, pos2Snap, walls)
        distance = gridDistance + snap1Distance + snap2Distance
        if bestDistance > distance:
          bestDistance = distance
    return bestDistance

  def getDistanceOnGrid(self, pos1, pos2, walls):
    self.updateWalls(walls)
    key = (pos1, pos2)
    if key in self._distances[walls]:
      return self._distances[walls][key]
    else:
      raise Exception("Positions not in grid: " + str(key))

  def isReadyForMazeDistance(self):
    return self._distances != {}

  def updateWalls(self, newWalls):
    if newWalls in self._distances:
      return

    removedWalls = []
    for x in range(newWalls.width):
      for y in range(newWalls.height):
        if self.initialWalls[x][y] and not newWalls[x][y]:
          removedWalls.append((x, y))

    prevWalls = self.initialWalls.deepCopy()
    for x, y in removedWalls:
      currWalls = prevWalls.deepCopy()
      currWalls[x][y] = False
      self.removeWall((x, y), prevWalls, currWalls)
      prevWalls = currWalls

  def removeWall(self, pos, oldWalls, newWalls):
    if newWalls in self._distances:
      return
    oldDists = self._distances[oldWalls]
    allNodes = newWalls.asList(False)
    newDists = oldDists.copy()
    updateDistancesFromSource(pos, newDists, allNodes, newWalls)
    for pos1, pos2 in oldDists.keys():
      prevDist = oldDists[(pos1, pos2)]
      newDist = newDists[(pos1, pos)] + newDists[(pos, pos2)]
      newDists[(pos1, pos2)] = min(prevDist, newDist)
    self._distances[newWalls] = newDists

  def recomputeDistances(self, walls):
    distances = {}
    allNodes = walls.asList(False)
    for source in allNodes:
      updateDistancesFromSource(source, distances, allNodes, walls)
    self._distances[walls.deepCopy()] = distances

def manhattanDistance(x, y ):
  return abs( x[0] - y[0] ) + abs( x[1] - y[1] )

def isInt(pos):
  x, y = pos
  return x == int(x) and y == int(y)

def getGrids2D(pos):
  grids = []
  for x, xDistance in getGrids1D(pos[0]):
    for y, yDistance in getGrids1D(pos[1]):
      grids.append(((x, y), xDistance + yDistance))
  return grids
  
def getGrids1D(x):
  intX = int(x)
  if x == int(x):
    return [(x, 0)]
  return [(intX, x-intX), (intX+1, intX+1-x)]
  
##########################################
# MACHINERY FOR COMPUTING MAZE DISTANCES #
##########################################

distanceMap = {}

class DistanceCalculator:
  def __init__(self, layout, distancer, default = 10000):
    self.layout = layout
    self.distancer = distancer
    self.default = default
  
  def run(self):
    global distanceMap

    if self.layout.walls not in distanceMap:
      distances = computeDistances(self.layout)
      distanceMap[self.layout.walls] = distances
    else:
      distances = distanceMap[self.layout.walls]

    self.distancer._distances = distanceMap

def computeDistances(layout):
    "Runs UCS to all other positions from each position"
    distances = {}
    allNodes = layout.walls.asList(False)
    for source in allNodes:
        updateDistancesFromSource(source, distances, allNodes, layout.walls)
    return distances

def updateDistancesFromSource(source, distances, allNodes, walls):
    dist = {}
    closed = {}
    for node in allNodes:
        dist[node] = sys.maxint
    import util
    queue = util.PriorityQueue()
    queue.push(source, 0)
    dist[source] = 0
    while not queue.isEmpty():
        node = queue.pop()
        if node in closed:
            continue
        closed[node] = True
        nodeDist = dist[node]
        adjacent = []
        x, y = node
        if not walls[x][y+1]:
            adjacent.append((x,y+1))
        if not walls[x][y-1]:
            adjacent.append((x,y-1) )
        if not walls[x+1][y]:
            adjacent.append((x+1,y) )
        if not walls[x-1][y]:
            adjacent.append((x-1,y))
        for other in adjacent:
            if not other in dist:
                continue
            oldDist = dist[other]
            newDist = nodeDist+1
            if newDist < oldDist:
                dist[other] = newDist
                queue.push(other, newDist)
    for target in allNodes:
        distances[(target, source)] = dist[target]
        distances[(source, target)] = dist[target]
    

def getDistanceOnGrid(distances, pos1, pos2):
    key = (pos1, pos2)
    if key in distances:
      return distances[key]
    return 100000
  
