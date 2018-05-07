# game.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import *
import time, os
import traceback
import sys
from config import *
from capsule import LEGAL_CAPSULES
import contestPowersBayesNet
from copy import deepcopy

#######################
# Parts worth reading #
#######################

# TODO: A lot of this class should be refactored into a table of
# constants, so that it is more generic
class AgentPowers:
    def __init__(self, powerDict={}, powerLimit=0):
        laser = int(powerDict.get('laser', 0))
        assert laser >= 0 and laser <= 2

        speed = int(powerDict.get('speed', 0))
        assert speed >= 0 and speed <= 2
        
        blast = int(powerDict.get('blast', 0))
        assert blast >= 0  # Can have any number of grenades
        
        armour = int(powerDict.get('armour', 0))
        assert armour >= 0 and armour <= 1
        
        juggernaut = int(powerDict.get('juggernaut', 0))
        assert juggernaut >= 0 and juggernaut <= 1
        
        capacity = int(powerDict.get('capacity', 0))
        assert capacity >= 0 and capacity <= 2
        
        invisibility = int(powerDict.get('invisibility', 0))
        assert invisibility >= 0 and invisibility <= 2

        sonar = int(powerDict.get('sonar', 0))
        assert sonar >= 0 and sonar <= 1

        respawn = int(powerDict.get('respawn', 0))
        assert respawn >= 0 and respawn <= 2

        assert (laser+speed+capacity+invisibility+respawn) <= powerLimit

        self.laser = laser
        self.timestepsBetweenMoves = (1.0 / SPEED_FROM_POINTS[speed])
        self.blast = blast
        self.armour = armour
        self.juggernaut = juggernaut
        self.capacity = capacity
        self.invisibility = invisibility
        self.sonar = sonar
        self.respawn = respawn

    def hasNoPowers(self):
        return self.laser == 0 and self.blast == 0 and self.armour == 0 and \
            self.juggernaut == 0 and self.capacity == 0 and \
            self.invisibility == 0 and self.sonar == 0 and \
            self.timestepsBetweenMoves == 1.0 and self.respawn == 0 

    def copy(self):
        result = AgentPowers()
        result.laser = self.laser
        result.timestepsBetweenMoves = self.timestepsBetweenMoves
        result.blast = self.blast
        result.armour = self.armour
        result.juggernaut = self.juggernaut
        result.invisibility = self.invisibility
        result.sonar = self.sonar
        result.capacity = self.capacity
        result.respawn = self.respawn

        return result

class AgentObservedVariables:
    def __init__(self,size=1,backpack=0, evidenceAssignmentDict=None):
        self.size = size
        self.backpack = backpack
        self.evidenceAssignmentDict = evidenceAssignmentDict

    def copy(self):
        return AgentObservedVariables(self.size,self.backpack,self.evidenceAssignmentDict)

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # Inspects the starting state
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()

    def choosePowers(self, state, limit):
        """
        Chooses powers (eg. laser, invisibility, speed)
        """
        return {}

class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'
    LASER = 'Laser'
    BLAST = 'Blast'

    LEFT =       {NORTH: WEST,
                   SOUTH: EAST,
                   EAST:  NORTH,
                   WEST:  SOUTH,
                   LASER: LASER,
                   BLAST: BLAST,
                   STOP:  STOP}

    RIGHT =      dict([(y,x) for x, y in LEFT.items()])

    REVERSE = {NORTH: SOUTH,
               SOUTH: NORTH,
               EAST: WEST,
               WEST: EAST,
               LASER: LASER,
               BLAST: BLAST,
               STOP: STOP}

    asVector = {NORTH: (0, 1),
                   SOUTH: (0, -1),
                   EAST:  (1, 0),
                   WEST:  (-1, 0),
                   LASER:  (0, 0),
                   BLAST: (0,0),
                   STOP:  (0, 0)}

class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def getPosition(self):
        return (self.pos)

    def getDirection(self):
        return self.direction

    def isInteger(self):
        x,y = self.pos
        return x == int(x) and y == int(y)

    def __eq__(self, other):
        if other == None: return False
        return (self.pos == other.pos and self.direction == other.direction)

    def __hash__(self):
        x = hash(self.pos)
        y = hash(self.direction)
        return hash(x + 13 * y)

    def __str__(self):
        return "(x,y)="+str(self.pos)+", "+str(self.direction)

    def generateSuccessor(self, vector):
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        x, y = self.pos
        dx, dy = vector
        direction = Actions.vectorToDirection(vector)
        if direction == Directions.STOP or direction == Directions.LASER or direction == Directions.BLAST:
            direction = self.direction # There is no stop direction
        return Configuration((x + dx, y+dy), direction)

class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, scared, etc).
    """

    def __init__( self, startConfiguration, isPacman):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.isPacman = isPacman
        self.isRespawning = False
        self.scaredTimer = 0
        self.sonarTimer = 0
        self.armourTimer = 0
        self.juggernautTimer = 0
        self.numCarrying = 0
        self.killCount = 0
        self.deathCount = 0
        self.numReturned = 0
        self.deathTimer = -1
        self.corpseExplosion = False
        self.ninjaMode = False
        self.powers = AgentPowers()
        self.observedVars = AgentObservedVariables()

    def __str__( self ):
        if self.isPacman:
            return "Pacman: " + str( self.configuration )
        else:
            return "Ghost: " + str( self.configuration )

    def __eq__( self, other ):
        if other == None:
            return False
        return self.configuration == other.configuration and self.isRespawning == other.isRespawning and self.scaredTimer == other.scaredTimer and self.sonarTimer == other.sonarTimer and self.armourTimer == other.armourTimer and self.juggernautTimer == other.juggernautTimer

    def __hash__(self):
        return hash((self.configuration, self.isRespawning, self.scaredTimer, self.sonarTimer, self.armourTimer, self.juggernautTimer))

    def copy( self ):
        state = AgentState( self.start, self.isPacman )
        state.configuration = self.configuration
        state.isRespawning = self.isRespawning
        state.scaredTimer = self.scaredTimer
        state.sonarTimer = self.sonarTimer
        state.armourTimer = self.armourTimer
        state.juggernautTimer = self.juggernautTimer
        state.numCarrying = self.numCarrying
        state.killCount = self.killCount
        state.deathCount = self.deathCount
        state.numReturned = self.numReturned
        state.deathTimer = self.deathTimer
        state.corpseExplosion = self.corpseExplosion
        state.ninjaMode = self.ninjaMode
        state.powers = self.powers.copy()
        state.observedVars = self.observedVars.copy()

        return state

    def getPosition(self):
        if self.configuration == None: return None
        return self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection()

    def getIsRespawning(self):
        return self.isRespawning

    def getLaserPower(self):
        return self.powers.laser

    def getBlastPower(self):
        return self.powers.blast

    def getArmourPower(self):
        return self.powers.armour

    def getJuggernautPower(self):
        return self.powers.juggernaut

    def getFoodCapacity(self):
        return FOOD_CAPACITY[self.powers.capacity]

    def getInvisibility(self):
        return self.powers.invisibility

    def getSonarPower(self):
        return self.powers.sonar

    def getSpeed(self):
        return 1.0/self.powers.timestepsBetweenMoves

    def getRespawnPower(self):
        return self.powers.respawn

class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a pacman board.
    """
    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]: raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT = 30

        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        # return hash(str(self))
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item =True ):
        return sum([x.count(item) for x in self.data])

    def asList(self, key = True):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key: list.append( (x,y) )
        return list

    def packBits(self):
        """
        Returns an efficient int list representation

        (width, height, bitPackedInts...)
        """
        bits = [self.width, self.height]
        currentInt = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cellIndexToPosition(i)
            if self[x][y]:
                currentInt += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(currentInt)
                currentInt = 0
        bits.append(currentInt)
        return tuple(bits)

    def _cellIndexToPosition(self, index):
        x = index / self.height
        y = index % self.height
        return x, y

    def _unpackBits(self, bits):
        """
        Fills in data from a bit-level representation
        """
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height: break
                x, y = self._cellIndexToPosition(cell)
                self[x][y] = bit
                cell += 1

    def _unpackInt(self, packed, size):
        bools = []
        if packed < 0: raise ValueError, "must be a positive integer"
        for i in range(size):
            n = 2 ** (self.CELLS_PER_INT - i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools

def reconstituteGrid(bitRep):
    if type(bitRep) is not type((1,2)):
        return bitRep
    width, height = bitRep[:2]
    return Grid(width, height, bitRepresentation= bitRep[2:])

####################################
# Parts you shouldn't have to read #
####################################

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions = {Directions.NORTH: (0, 1),
                   Directions.SOUTH: (0, -1),
                   Directions.EAST:  (1, 0),
                   Directions.WEST:  (-1, 0),
                   Directions.LASER:  (0, 0),
                   Directions.BLAST:  (0, 0),
                   Directions.STOP:  (0, 0)}

    _directionsAsList = _directions.items()

    TOLERANCE = .001

    def reverseDirection(action):
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action
    reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed = 1.0):
        dx, dy =  Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        # Note that this might cause an agent that just lost the
        # juggernaut power to run into a wall.  In this case, the wall
        # should be destroyed.
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    def getPossibleJuggernautActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x >= 1 and next_x <= walls.width - 2 and \
               next_y >= 1 and next_y <= walls.height - 2:
                possible.append(dir)

        return possible

    getPossibleJuggernautActions = staticmethod(getPossibleJuggernautActions)

    def getLegalNeighbors(position, walls):
        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)

class GameStateData:
    """

    """
    def __init__( self, prevState = None ):
        """
        Generates a new data packet by copying information from its predecessor.
        """
        if prevState != None:
            self.food = prevState.food.shallowCopy()
            self.walls = prevState.walls.shallowCopy()
            self.timedWalls = prevState.timedWalls.copy()
            self.redCapsules = [c.deepCopy() for c in prevState.redCapsules]
            self.blueCapsules = [c.deepCopy() for c in prevState.blueCapsules]
            self.agentStates = self.copyAgentStates( prevState.agentStates )
            self.layout = prevState.layout
            self.score = prevState.score
            self.time = prevState.time

        self._lose = False
        self._win = False
        self.resetChangeData()

    def resetChangeData( self ):
        self._foodEaten = None
        self._foodAdded = None
        self._capsuleEaten = None
        self._wallsChanged = []
        self._agentMoved = None
        self._timeTillAgentMovesAgain = None
        self._action = None
        self.scoreChange = 0

    def deepCopy( self ):
        state = GameStateData( self )
        state.food = self.food.deepCopy()
        state.walls = self.walls.deepCopy()
        # timedWalls, capsules have already been sufficiently copied
        state.layout = self.layout.deepCopy()
        state._agentMoved = self._agentMoved
        state._timeTillAgentMovesAgain = self._timeTillAgentMovesAgain
        state._action = self._action
        state._foodEaten = self._foodEaten
        state._foodAdded = self._foodAdded
        state._capsuleEaten = self._capsuleEaten
        state._wallsChanged = self._wallsChanged[:]
        return state


    def copyAgentStates( self, agentStates ):
        copiedStates = []
        for agentState in agentStates:
            copiedStates.append( agentState.copy() )
        return copiedStates

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        if other == None or not isinstance(other, GameStateData): return False
        if not self.agentStates == other.agentStates: return False
        if not self.food == other.food: return False
        if not self.walls == other.walls: return False
        if not self.timedWalls == other.timedWalls: return False
        if not self.redCapsules == other.redCapsules: return False
        if not self.blueCapsules == other.blueCapsules: return False
        if not self.score == other.score: return False
        if not self.time == other.time: return False
        return True

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        for i, state in enumerate( self.agentStates ):
            try:
                int(hash(state))
            except TypeError, e:
                print e
                #hash(state)
        return int((hash(tuple(self.agentStates)) + 13*hash(self.food) + 13*hash(self.walls) + 17*hash(tuple(self.timedWalls.items())) + 113* hash(tuple(self.redCapsules + self.blueCapsules)) + 7 * hash(self.score) + 11 * hash(self.time)) % 1048575 )

    def __str__( self ):
        width, height = self.layout.width, self.layout.height
        map = Grid(width, height)
        if type(self.food) == type((1,2)):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agentState in self.agentStates:
            if agentState == None: continue
            if agentState.configuration == None: continue
            x,y = [int( i ) for i in nearestPoint( agentState.configuration.pos )]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr( agent_dir )
            else:
                map[x][y] = self._ghostStr( agent_dir )

        return str(map) + ("\nScore: %d\n" % self.score)

    def _foodWallStr( self, hasFood, hasWall ):
        if hasFood:
            return '.'
        elif hasWall:
            return '%'
        else:
            return ' '

    def _pacStr( self, dir ):
        if dir == Directions.NORTH:
            return 'v'
        if dir == Directions.SOUTH:
            return '^'
        if dir == Directions.WEST:
            return '>'
        return '<'

    def _ghostStr( self, dir ):
        return 'G'
        if dir == Directions.NORTH:
            return 'M'
        if dir == Directions.SOUTH:
            return 'W'
        if dir == Directions.WEST:
            return '3'
        return 'E'

    def initialize( self, layout, numGhostAgents ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.food = layout.food.deepCopy()
        self.walls = layout.walls.deepCopy()
        self.timedWalls = {}
        #jferguson -- remove capsules for project 1
        if PROJECT_FLAG == 1:
            self.redCapsules = []
            self.blueCapsules = []
        else:
            self.redCapsules = halfCapsuleList(layout.capsules, self.food, red = False)
            self.blueCapsules = halfCapsuleList(layout.capsules, self.food, red = True)
        self.layout = layout
        self.score = 0
        self.scoreChange = 0
        self.time = 0

        self.agentStates = []
        numGhosts = 0
        # TODO: Following code is contest-specific, try to get it into capture.py or somehow refactor
        for i in range(len(layout.agentPositions)):
            isPacman, pos = layout.agentPositions[i]
            config = Configuration(pos, Directions.STOP)
            self.agentStates.append( AgentState(config, isPacman) )

def halfCapsuleList(capsules, grid, red):
    halfway = grid.width / 2
    newList = []
    for capsule in capsules:
        x,y = capsule.getPosition()
        if red and x < halfway:
            newList.append(capsule)
        elif not red and x >= halfway:
            newList.append(capsule)
    return newList

try:
    import boinc
    _BOINC_ENABLED = True
except:
    _BOINC_ENABLED = False

class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__( self, redTeam, blueTeam, powerLimit, capsuleLimit, display, initState, length, keyboardAgents, rules, startingIndex=0, muteAgents=False, catchExceptions=False ):
        self.redTeam = redTeam
        self.blueTeam = blueTeam
        self.agentCrashed = False
        self.powerLimit = powerLimit
        self.capsuleLimit = capsuleLimit
        self.display = display
        self.state = initState
        self.length = length
        self.keyboardAgents = keyboardAgents
        self.state.data.timeleft = length
        self.rules = rules
        self.startingIndex = startingIndex
        self.gameOver = False
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.moveHistory = []
        self.positionHistory = []
        self.totalAgentTimes = [0 for i in range(4)]
        self.totalAgentTimeWarnings = [0 for i in range(4)]
        self.agentTimeout = False
        import cStringIO
        self.agentOutput = [cStringIO.StringIO() for i in range(4)]
        self.comeback = [False, False] #Flag for whether each team [blue, red] has been down by >=15 

    def getProgress(self):
        if self.gameOver:
            return 1.0
        else:
            return self.rules.getProgress(self)

    def _agentCrash( self, agentIndex, quiet=False):
        "Helper method for handling agent crashes"
        if not quiet: traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    OLD_STDOUT = None
    OLD_STDERR = None

    def mute(self, agentIndex):
        if not self.muteAgents: return
        global OLD_STDOUT, OLD_STDERR
        import cStringIO
        OLD_STDOUT = sys.stdout
        OLD_STDERR = sys.stderr
        sys.stdout = self.agentOutput[agentIndex]
        sys.stderr = self.agentOutput[agentIndex]

    def unmute(self):
        if not self.muteAgents: return
        global OLD_STDOUT, OLD_STDERR
        # Revert stdout/stderr to originals
        sys.stdout = OLD_STDOUT
        sys.stderr = OLD_STDERR

    def runWithTimeout(self, f, timeout, objectStr, methodName):
        if self.catchExceptions:
            try:
                timedF = TimeoutFunction(f, int(timeout))
                try:
                    start_time = time.time()
                    result = timedF()
                    time_taken = time.time() - start_time
                except TimeoutFunctionException:
                    print >>sys.stderr, "%s ran out of time during %s" % objectStr, methodName
                    return False
            except Exception,data:
                return False
        else:
            result = f()
        return result

    def initializeAgents(self):
        timeout = self.rules.getMaxStartupTime()
        f = lambda: self.redTeam.createAgents(0, 2, True, self.state.deepCopy())
        red = self.runWithTimeout(f, timeout, 'Red team', 'createAgents')
        f = lambda: self.blueTeam.createAgents(1, 3, False, self.state.deepCopy())
        blue = self.runWithTimeout(f, timeout, 'Blue team', 'createAgents')
        if not (red and blue):
            return False

        self.agents = [red[0], blue[0], red[1], blue[1]]

        # Replace agents by keyboard agents if necessary
        firstKeyboardAgent = False
        for i, agent in enumerate(self.keyboardAgents):
            if agent:
                self.agents[i] = agent
        return True

    def initializeAgentObservedVars(self):
        for s in self.state.data.agentStates:
            gSize = 1.2 - 0.2 * (s.getSpeed())
            belt = s.getLaserPower() > 0

            evidenceAssignmentDict = contestPowersBayesNet.computeEvidence(s.powersAssignments)
            s.observedVars = AgentObservedVariables(gSize,belt, \
                            evidenceAssignmentDict=evidenceAssignmentDict)
        return True


    def initializeAgentPowers(self):
        for i, agent in enumerate(self.agents):
            f = lambda: agent.choosePowers(self.state.deepCopy(), self.powerLimit)
            timeout = self.rules.getPowerChoosingTime(i)
            powers = self.runWithTimeout(f, timeout, 'Agent ' + str(i), 'choosePowers')
            if powers == False:
                return False

            assert sum(powers.values()) <= self.powerLimit
            for power in powers.keys():
                assert power in LEGAL_POWERS

            powerData = AgentPowers(powers, self.powerLimit)
            self.state.data.agentStates[i].powers = powerData
            self.state.data.agentStates[i].powersAssignments = powers
        return True

    def initializeCapsules(self):
        redAgents = [self.state.data.agentStates[i] for i in [0, 2]]
        redEvidences = [deepcopy(s.observedVars.evidenceAssignmentDict)
                            for s in redAgents]
        blueAgents = [self.state.data.agentStates[i] for i in [1, 3]]
        blueEvidences = [deepcopy(s.observedVars.evidenceAssignmentDict)
                            for s in blueAgents]

        timeout = self.rules.getCapsuleChoosingTime()
        blueHidden = self.state.deepCopy()
        for i in self.state.getBlueTeamIndices():
            blueHidden.data.agentStates[i].powers = AgentPowers()
            blueHidden.data.agentStates[i].powersAssignments = {}
        f = lambda: self.redTeam.chooseCapsules(blueHidden, self.capsuleLimit, blueEvidences)
        extraRedCapsules = self.runWithTimeout(f, timeout, 'Red team', 'chooseCapsules')
        if extraRedCapsules == False:
            return False
        if not self.checkCapsules(extraRedCapsules, self.state.getRedCapsules(), self.capsuleLimit):
            print >>sys.stderr, "Red team chose bad capsules"
            return False

        redHidden = self.state.deepCopy()
        for i in self.state.getRedTeamIndices():
            redHidden.data.agentStates[i].powers = AgentPowers()
            redHidden.data.agentStates[i].powersAssignments = {}
        f = lambda: self.blueTeam.chooseCapsules(redHidden, self.capsuleLimit, redEvidences)
        extraBlueCapsules = self.runWithTimeout(f, timeout, 'Blue team', 'chooseCapsules')
        if extraBlueCapsules == False:
            return False
        if not self.checkCapsules(extraBlueCapsules, self.state.getBlueCapsules(), self.capsuleLimit):
            print >>sys.stderr, "Blue team chose bad capsules"
            return False

        self.state.data.redCapsules += extraRedCapsules
        self.state.data.blueCapsules += extraBlueCapsules
        return True

    def checkCapsules(self, newCapsules, existingCapsules, capsuleLimit):
        existingPos = [c.getPosition() for c in existingCapsules]
        for cap in newCapsules:
            if cap.__class__ not in LEGAL_CAPSULES:
                print >>sys.stderr, "Not a valid capsule type:", cap
                return False
            x, y = cap.getPosition()
            if self.state.data.walls[x][y]:
                print >>sys.stderr, "Capsules cannot overlap a wall"
                return False
            if self.state.data.food[x][y]:
                print >>sys.stderr, "Capsules cannot overlap food pellets"
                return False
            if (x, y) in existingPos:
                print >>sys.stderr, "Capsules cannot overlap other capsules of the same team"
                return False

        left = halfCapsuleList(newCapsules, self.state.data.food, red = True)
        if (len(left) > capsuleLimit / 2) or ((len(newCapsules) - len(left)) > capsuleLimit / 2):
            print >>sys.stderr, "Too many capsules on one side"
            return False
        return True
            

    def getAgentAction(self, agentIndex):
        # Fetch the next agent
        agent = self.agents[agentIndex]
        move_time = 0
        skip_action = False
        # Generate an observation of the state
        if 'observationFunction' in dir( agent ):
            self.mute(agentIndex)
            if self.catchExceptions:
                try:
                    timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
                    try:
                        start_time = time.time()
                        observation = timed_func(self.state.deepCopy())
                    except TimeoutFunctionException:
                        skip_action = True
                    move_time += time.time() - start_time
                    self.unmute()
                except Exception,data:
                    self._agentCrash(agentIndex, quiet=False)
                    self.unmute()
                    return False
            else:
                observation = agent.observationFunction(self.state.deepCopy())
            self.unmute()
        else:
            observation = self.state.deepCopy()

        # Solicit an action
        action = None
        self.mute(agentIndex)
        if self.catchExceptions:
            try:
                timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                try:
                    start_time = time.time()
                    if skip_action:
                        raise TimeoutFunctionException()
                    action = timed_func( observation )
                except TimeoutFunctionException:
                    print >>sys.stderr, "Agent %d timed out on a single move!" % agentIndex
                    self.agentTimeout = True
                    self._agentCrash(agentIndex, quiet=True)
                    self.unmute()
                    return False

                move_time += time.time() - start_time

                if move_time > self.rules.getMoveWarningTime(agentIndex):
                    self.totalAgentTimeWarnings[agentIndex] += 1
                    print >>sys.stderr, "Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex])
                    if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                        print >>sys.stderr, "Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex])
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return False

                self.totalAgentTimes[agentIndex] += move_time
                #print "Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex])
                if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                    print >>sys.stderr, "Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex])
                    self.agentTimeout = True
                    self._agentCrash(agentIndex, quiet=True)
                    self.unmute()
                    return False
                self.unmute()
            except Exception,data:
                self._agentCrash(agentIndex)
                self.unmute()
                return False
        else:
            action = agent.getAction(observation)
        self.unmute()
        return True, action

    def getAndRunAgentAction(self, agentIndex):
        actionTuple = self.getAgentAction(agentIndex)
        if not actionTuple:
            return False
        _, action = actionTuple

        # Execute the action
        self.moveHistory.append( (agentIndex, action) )
        if self.catchExceptions:
            try:
                self.state = self.state.generateSuccessor( action )
            except Exception,data:
                self.mute(agentIndex)
                self._agentCrash(agentIndex)
                self.unmute()
                return False
        else:
            self.state = self.state.generateSuccessor( action )
        return True

    def informLearningAgents(self):
        # inform a learning agent of the game result
        for agentIndex, agent in enumerate(self.agents):
            if "final" in dir( agent ) :
                try:
                    self.mute(agentIndex)
                    agent.final( self.state )
                    self.unmute()
                except Exception,data:
                    if not self.catchExceptions: raise
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return False
        return True

    def setup( self ):
        """
        Runs the first two phases, where powers and capsules are chosen.
        """
        return self.initializeAgents() and \
            self.initializeAgentPowers() and \
            self.initializeAgentObservedVars() and \
            self.initializeCapsules()

    def run( self ):
        """
        Main control loop for game play.
        """
        self.display.initialize(self.state.data)
        self.numMoves = 0

        agentIndex = self.state.getNextAgentIndex()
        numAgents = len( self.agents )

        while not self.gameOver:
            event = self.state.getNextEvent()
            if event.isAgentMove():
                agentIndex = event.getAgentIndex()
                if not self.getAndRunAgentAction(agentIndex):
                    return
                self.numMoves += 1
            else:
                self.runEvent()

            # Change the display
            self.display.update(self.state.data)

            ###idx = agentIndex - agentIndex % 2 + 1
            ###self.display.update( self.state.makeObservation(idx).data )

            self.positionHistory.append( (agentIndex, self.state.getAgentPosition(agentIndex)) )
            if self.state.data.score >= 15:
                self.comeback[0] = True
            if self.state.data.score <= -15:
                self.comeback[1] = True

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.getProgress())

        if not self.informLearningAgents():
            return
        self.display.finish()
