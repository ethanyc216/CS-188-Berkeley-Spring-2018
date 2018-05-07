# pacman.py
# ---------
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

"""
Capture.py holds the logic for Pacman capture the flag.

  (i)  Your interface to the pacman world:
          Pacman is a complex environment.  You probably don't want to
          read through all of the code we wrote to make the game runs
          correctly.  This section contains the parts of the code
          that you will need to understand in order to complete the
          project.  There is also some code in game.py that you should
          understand.

  (ii)  The hidden secrets of pacman:
          This section contains all of the logic code that the pacman
          environment uses to decide who can move where, who dies when
          things collide, etc.  You shouldn't need to read this section
          of code, but you can if you want.

  (iii) Framework to start a game:
          The final section contains the code for reading the command
          you use to set up the game, then starting up a new game, along with
          linking in all the external parts (agent functions, graphics).
          Check this section out to see all the options available to you.

To play your first game, type 'python capture.py' from the command line.
The keys are
  P1: 'a', 's', 'd', and 'w' to move
  P2: 'l', ';', ',' and 'p' to move
"""
from capsule import TimerDecrementEvent
from events import EventQueue
from events import Event
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
from game import Grid
from game import Configuration
from game import Agent
from game import AgentPowers
from game import reconstituteGrid
import sys, util, types, time, random, imp, os
from keyboardAgents import KeyboardAgent, KeyboardAgent2
from config import *

###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def getLegalActions( self, agentIndex=0 ):
        """
        Returns the legal actions for the agent specified.
        """
        return AgentRules.getLegalActions( self, agentIndex )

    def generateSuccessor( self, action ):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isOver() or self.eventQueue.isEmpty():
            raise Exception('Can\'t generate a successor of a terminal state.')

        time, event = self.eventQueue.peek()
        assert event.isAgentMove(), 'Can only generate successors of a state where an agent is about to move'
        state = self.makeAgentMove(action)
        state.resolveEventsUntilAgentEvent()

        return state

    def makeAgentMove( self, action ):
        # Copy current state
        state = GameState(self)

        time, event = state.eventQueue.pop()
        agentIndex = event.getAgentIndex()
        agentState = state.data.agentStates[agentIndex]
        agentState.isRespawning = False
        state.data.time = time
        delay = agentState.powers.timestepsBetweenMoves
        state.registerEventWithDelay(event, delay)

        # Find appropriate rules for the agent
        AgentRules.applyAction( state, action, agentIndex )

        #jferguson -> Modify this for project 1 to prevent dying    
        if PROJECT_FLAG > 1:
            AgentRules.checkDeath(state, agentIndex)
            if action == Directions.LASER:
                AgentRules.checkLaserShot(state,agentIndex)
            if action == Directions.BLAST:
                AgentRules.checkBlast(state,agentIndex)

        if PROJECT_FLAG > 1:
            if action == Directions.BLAST:
                agentState.powers.blast -= 1

        for agentState in state.data.agentStates:
            agentState.deathTimer -= 1 

        # Book keeping
        state.data._agentMoved = agentIndex
        # Note:  It is important that the following value accurately
        # reflects when Pacman will make the next move, even if the
        # speed changes (such as a speed-up power pellet).  Otherwise
        # the graphics will do weird things.
        state.data._timeTillAgentMovesAgain = delay
        state.data._action = action
        state.data.score += state.data.scoreChange
        state.data.timeleft = self.data.timeleft - 1

        return state

    def runEvent( self ):
        # Check that successors exist
        if self.eventQueue.isEmpty():
            raise Exception('Can\'t run an event of a terminal state.')

        time, event = self.eventQueue.pop()
        assert not event.isAgentMove(), 'Can\'t run an AgentMoveEvent'
        self.data.time = time
        event.trigger(self)
        return event

    def getNextEvent( self ):
        _, event = self.eventQueue.peek()
        return event

    def getAgentState(self, index):
        return self.data.agentStates[index]

    def getAgentPosition(self, index):
        """
        Returns a location tuple if the agent with the given index is observable;
        if the agent is unobservable, returns None.
        """
        agentState = self.data.agentStates[index]
        ret = agentState.getPosition()
        if ret:
            return tuple(int(x) for x in ret)
        return ret

    def getNextAgentIndex(self):
        for time, event in self.eventQueue.getSortedTimesAndEvents():
            if event.isAgentMove():
                return event.getAgentIndex()
        assert False, "No more moves can be made"

    def getAgentMoveTime(self, agentIndex):
        for time, event in self.eventQueue.getSortedTimesAndEvents():
            if event.isAgentMove():
                if event.getAgentIndex() == agentIndex:
                    return time
        assert False, "No more moves can be made by agent " + str(agentIndex)

    def getNumAgents( self ):
        return len( self.data.agentStates )

    def getScore( self ):
        return float(self.data.score)

    def getRedFood(self):
        """
        Returns a matrix of food that corresponds to the food on the red team's side.
        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
        red (meaning red is protecting it, blue is trying to eat it).
        """
        return halfGrid(self.data.food, red = True)

    def getBlueFood(self):
        """
        Returns a matrix of food that corresponds to the food on the blue team's side.
        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
        blue (meaning blue is protecting it, red is trying to eat it).
        """
        return halfGrid(self.data.food, red = False)

    def getCapsulesOfTeam(self, isRed):
        if isRed:
            return self.getRedCapsules()
        else:
            return self.getBlueCapsules()

    def getRedCapsules(self):
        return self.data.redCapsules

    def getBlueCapsules(self):
        return self.data.blueCapsules

    def getWalls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is a wall at (x,y), just call

        walls = state.getWalls()
        if walls[x][y] == True: ...
        """
        return self.data.walls

    def hasFood(self, x, y):
        """
        Returns true if the location (x,y) has food, regardless of
        whether it's blue team food or red team food.
        """
        return self.data.food[x][y]

    def hasWall(self, x, y):
        """
        Returns true if (x,y) has a wall, false otherwise.
        """
        return self.data.walls[x][y]

    def isOver( self ):
        return self.data._win

    def getRedTeamIndices(self):
        """
        Returns a list of agent index numbers for the agents on the red team.
        """
        return self.redTeamIndices[:]

    def getBlueTeamIndices(self):
        """
        Returns a list of the agent index numbers for the agents on the blue team.
        """
        return self.blueTeamIndices[:]

    def getOpponentTeamIndices(self, agentIndex):
        if self.isOnRedTeam(agentIndex):
            return self.getBlueTeamIndices()
        else:
            return self.getRedTeamIndices()

    def getOpponentAgentPowers(self, opponentAgentIndex):
        """
        Returns the dictionary of powers of opponent agent
        """
        return self.data.agentStates[opponentAgentIndex].powers.copy()
    
    def isOnRedTeam(self, agentIndex):
        """
        Returns true if the agent with the given agentIndex is on the red team.
        """
        return self.teams[agentIndex]

    def getInitialAgentPosition(self, agentIndex):
        "Returns the initial position of an agent."
        return self.data.layout.agentPositions[agentIndex][1]

    def getCapsules(self):
        """
        Returns a list of positions (x,y) of the remaining capsules.
        """
        return self.data.redCapsules + self.data.blueCapsules

    def isValidPosition(self, position, isRed):
        capsules = self.getCapsulesOfTeam(isRed)
        return self.isValidPositionWithExistingCapsules(position, capsules)

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__( self, prevState = None ):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState != None: # Initial state
            self.data = GameStateData(prevState.data)
            self.eventQueue = prevState.eventQueue.deepCopy()
            self.blueTeamIndices = prevState.blueTeamIndices
            self.redTeamIndices = prevState.redTeamIndices
            self.data.timeleft = prevState.data.timeleft

            self.teams = prevState.teams
            self.agentDistances = prevState.agentDistances
        else:
            self.data = GameStateData()
            self.eventQueue = EventQueue()
            self.agentDistances = []

    def resolveEventsUntilAgentEvent(self):
        # Resolve any events until the next agent event
        while not self.eventQueue.isEmpty():
            time, event = self.eventQueue.peek()
            if event.isAgentMove():
                return
            else:
                self.runEvent()

    def registerEventWithDelay(self, event, delay):
        self.eventQueue.registerEventAtTime(event, self.data.time + delay)

    def delayAgent(self, agentIndex, delayTime):
        def f(event):
          return event.isAgentMove() and event.getAgentIndex() == agentIndex

        time, event = self.eventQueue.removeFirstEventSatisfying(f)
        self.eventQueue.registerEventAtTime(event, time + delayTime)

    def deepCopy( self ):
        state = GameState( self )
        state.data = self.data.deepCopy()
        # Event queue has already been copied in the constructor
        state.data.timeleft = self.data.timeleft

        state.blueTeamIndices = self.blueTeamIndices[:]
        state.redTeamIndices = self.redTeamIndices[:]
        state.teams = self.teams[:]
        state.agentDistances = self.agentDistances[:]
        return state

    def makeObservation(self, index):
        state = self.deepCopy()

        # Remove states of distant opponents
        if index in self.blueTeamIndices:
            team = self.blueTeamIndices
            otherTeam = self.redTeamIndices
        else:
            otherTeam = self.blueTeamIndices
            team = self.redTeamIndices

        for enemy in otherTeam:
            seen = False
            enemyPos = state.getAgentPosition(enemy)
            enemyInvisRange = INVISIBILITY_RANGE[state.data.agentStates[enemy].getInvisibility()]
            for teammate in team:
                if  state.data.agentStates[teammate].getSonarPower() or util.manhattanDistance(enemyPos, state.getAgentPosition(teammate)) <= enemyInvisRange:
                    seen = True
            if not seen: state.data.agentStates[enemy].configuration = None
        return state

    def isValidPositionWithExistingCapsules(self, position, capsules):
        x, y = position
        existingPos = [c.getPosition() for c in capsules]
        if position in capsules: return False
        return not (self.data.walls[x][y] or self.data.food[x][y])

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data \
            and hasattr(other, 'eventQueue') and self.eventQueue == other.eventQueue \

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        return hash( (self.data, self.eventQueue) )

    def __str__( self ):

        return str(self.data)

    def initialize( self, layout, numAgents ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, numAgents)
        positions = [a.configuration for a in self.data.agentStates]
        self.blueTeamIndices = [i for i,p in enumerate(positions) if not self.isRed(p)]
        self.redTeamIndices = [i for i,p in enumerate(positions) if self.isRed(p)]
        self.teams = [self.isRed(p) for p in positions]

        numAgents = self.getNumAgents()
        for i in range(numAgents):
            self.registerEventWithDelay(AgentMoveEvent(i), i)
        self.registerEventWithDelay(WallTimerEvent(), 1)
        self.registerEventWithDelay(TimerDecrementEvent(), 1)

    def isRed(self, configOrPos):
        width = self.data.layout.width
        if type(configOrPos) == type( (0,0) ):
            return configOrPos[0] < width / 2
        else:
            return configOrPos.pos[0] < width / 2

def halfGrid(grid, red):
    halfway = grid.width / 2
    halfgrid = Grid(grid.width, grid.height, False)
    if red:
        xrange = range(halfway)
    else:
        xrange = range(halfway, grid.width)

    for y in range(grid.height):
        for x in xrange:
            if grid[x][y]: halfgrid[x][y] = True

    return halfgrid

class AgentMoveEvent(Event):
    """
    The GameStates could be generated in one of two modes - either by
    the main game logic, where actions are solicited from actual
    agents, or in planning mode, where one agent thinks about how
    another agent might respond.  As a result, we don't know what to
    do in trigger, so we do not implement it - instead, the logic in
    GameState should make sure to treat AgentMoveEvents specially.
    """
    def __init__(self, agentIndex, prevId=None):
        Event.__init__(self, prevId)
        self.index = agentIndex

    def getAgentIndex(self):
        return self.index

    def isAgentMove(self):
        return True

    def deepCopy(self):
        return AgentMoveEvent(self.index, self.eventId)

    def __eq__( self, other ):
        return isinstance(other, AgentMoveEvent) and \
            self.index == other.index and \
            self.eventId == other.eventId

    def __hash__( self ):
        return hash((self.index, self.eventId))

    def __str__(self):
        return "Agent " + str(self.getAgentIndex()) + " move"

class WallTimerEvent(Event):
    """
    The GameStates could be generated in one of two modes - either by
    the main game logic, where actions are solicited from actual
    agents, or in planning mode, where one agent thinks about how
    another agent might respond.  As a result, we don't know what to
    do in trigger, so we do not implement it - instead, the logic in
    GameState should make sure to treat AgentMoveEvents specially.
    """
    def trigger(self, state):
        timedWalls = state.data.timedWalls
        for pos in timedWalls.keys():
            if timedWalls[pos] == 1:
                del timedWalls[pos]
                x, y = pos
                state.data.walls[x][y] = False
                state.data._wallsChanged.append(pos)
            else:
                timedWalls[pos] -= 1
        # Repeat this work after 1 more timestep
        state.registerEventWithDelay(self, 1)

    def deepCopy(self):
        return WallTimerEvent(self.eventId)

    def __eq__( self, other ):
        return isinstance(other, WallTimerEvent) and \
            self.eventId == other.eventId

    def __hash__( self ):
        return hash(self.eventId)

############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

class CaptureRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, quiet = False):
        self.quiet = quiet

    def newGame( self, redTeam, blueTeam, layout, powerLimit, capsuleLimit, display, length, keyboardAgents, muteAgents, catchExceptions ):
        initState = GameState()
        initState.initialize( layout, 4 )
        starter = random.randint(0,1)
        print('%s team starts' % ['Red', 'Blue'][starter])
        game = Game(redTeam, blueTeam, powerLimit, capsuleLimit, display, initState, length, keyboardAgents, self, startingIndex=starter, muteAgents=muteAgents, catchExceptions=catchExceptions)
        if 'drawCenterLine' in dir(display):
            display.drawCenterLine()
        self._initBlueFood = initState.getBlueFood().count()
        self._initRedFood = initState.getRedFood().count()
        return game

    # TODO:  We should really have a variable called _over, not _win
    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if 'moveHistory' in dir(game):
            if len(game.moveHistory) == game.length:
                state.data._win = True

        if state.isOver():
            game.gameOver = True
            if not game.rules.quiet:
                redCount = 0
                blueCount = 0
                foodToWin = (TOTAL_FOOD/2) - MIN_FOOD
                for index in range(state.getNumAgents()):
                    agentState = state.data.agentStates[index]
                    if index in state.getRedTeamIndices():
                        redCount += agentState.numReturned
                    else:
                        blueCount += agentState.numReturned

                if blueCount >= foodToWin:#state.getRedFood().count() == MIN_FOOD:
                    print 'The Blue team has returned at least %d of the opponents\' dots.' % foodToWin
                elif redCount >= foodToWin:#state.getBlueFood().count() == MIN_FOOD:
                    print 'The Red team has returned at least %d of the opponents\' dots.' % foodToWin
                else:#if state.getBlueFood().count() > MIN_FOOD and state.getRedFood().count() > MIN_FOOD:
                    print 'Time is up.'
                    if state.data.score == 0: print 'Tie game!'
                    else:
                        winner = 'Red'
                        if state.data.score < 0: winner = 'Blue'
                        print 'The %s team wins by %d points.' % (winner, abs(state.data.score))

    def getProgress(self, game):
        blue = 1.0 - (game.state.getBlueFood().count() / float(self._initBlueFood))
        red = 1.0 - (game.state.getRedFood().count() / float(self._initRedFood))
        moves = len(self.moveHistory) / float(game.length)

        # return the most likely progress indicator, clamped to [0, 1]
        return min(max(0.75 * max(red, blue) + 0.25 * moves, 0.0), 1.0)

    def agentCrash(self, game, agentIndex):
        if agentIndex % 2 == 0:
            print >>sys.stderr, "Red agent crashed"
            game.state.data.score = -1
        else:
            print >>sys.stderr, "Blue agent crashed"
            game.state.data.score = 1

    def getMaxTotalTime(self, agentIndex):
        return 900  # Move limits should prevent this from ever happening

    def getMaxStartupTime(self):
        return 30 # 30 seconds for createTeam

    def getPowerChoosingTime(self, agentIndex):
        return 1 # One second to choose powers

    def getCapsuleChoosingTime(self):
        return 30 # 30 seconds to choose capsules

    def getMoveWarningTime(self, agentIndex):
        return 1  # One second per move

    def getMoveTimeout(self, agentIndex):
        return 3  # Three seconds results in instant forfeit

    def getMaxTimeWarnings(self, agentIndex):
        return 2  # Third violation loses the game

class AgentRules:
    """
    These functions govern how each agent interacts with her environment.
    """

    def getLegalActions( state, agentIndex ):
        """
        Returns a list of legal actions (which are both possible & allowed)
        """
        agentState = state.getAgentState(agentIndex)
        conf = agentState.configuration
        assert conf is not None
        if agentState.getJuggernautPower():
            possibleActions = Actions.getPossibleJuggernautActions( conf, state.data.walls )
        else:
            possibleActions = Actions.getPossibleActions( conf, state.data.walls )
        return AgentRules.filterForAllowedActions( agentState, possibleActions)
    getLegalActions = staticmethod( getLegalActions )

    def filterForAllowedActions(agentState, possibleActions):
        if Directions.LASER in possibleActions:
            if (not agentState.getLaserPower()) or agentState.scaredTimer > 0:
                possibleActions.remove( Directions.LASER )

        if Directions.BLAST in possibleActions:
            if (not agentState.getBlastPower()) or agentState.scaredTimer > 0:
                possibleActions.remove(Directions.BLAST)

        return possibleActions
    filterForAllowedActions = staticmethod( filterForAllowedActions )

    def applyAction( state, action, agentIndex ):
        """
        Edits the state to reflect the results of the action.
        """
        legal = AgentRules.getLegalActions( state, agentIndex )
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        # Update Configuration
        agentState = state.data.agentStates[agentIndex]
        speed = 1.0

        vector = Actions.directionToVector( action, speed )
        oldConfig = agentState.configuration
        assert oldConfig is not None
        agentState.configuration = oldConfig.generateSuccessor( vector )

        # Eat
        next = agentState.configuration.getPosition()
        nearest = nearestPoint( next )

        x, y = nearest
        if state.data.walls[x][y]:
            # Only possible for an agent with the Juggernaut power,
            # otherwise the move would not be legal.  However, the
            # agent might not have the power now (a scared Juggernaut
            # could have lost the power last turn).
            # Destroy the wall.
            state.data.walls[x][y] = False
            state.data._wallsChanged.append((x, y))

        if next == nearest:
            # Change agent type
            isRed = state.isOnRedTeam(agentIndex)
            agentState.isPacman = [isRed, state.isRed(agentState.configuration)].count(True) == 1
            # if he's no longer pacman, he's on his own side, so reset the num carrying timer
            if agentState.numCarrying > 0 and not agentState.isPacman:
                #jferguson -- Change the score here now
                score = agentState.numCarrying if isRed else -1*agentState.numCarrying
                state.data.scoreChange += score
                agentState.numReturned += agentState.numCarrying
                foodToWin = (TOTAL_FOOD/2) - MIN_FOOD
                if agentState.numCarrying >= foodToWin:
                    agentState.ninjaMode = True
                agentState.numCarrying = 0
                #jferguson -- Check the win condition here
                redCount = 0
                blueCount = 0
                for index in range(state.getNumAgents()):
                    agentState = state.data.agentStates[index]
                    if index in state.getRedTeamIndices():
                        redCount += agentState.numReturned
                    else:
                        blueCount += agentState.numReturned
                if redCount >= foodToWin or blueCount >= foodToWin:
                    state.data._win = True

        if manhattanDistance( nearest, next ) <= 0.9 :
            AgentRules.consume( agentIndex, nearest, state, state.isOnRedTeam(agentIndex) )

    applyAction = staticmethod( applyAction )

    def consume( agentIndex, position, state, isRed ):
        agentState = state.data.agentStates[agentIndex]
        x,y = position
        capacity = agentState.getFoodCapacity()
        # Eat food
        if agentState.isPacman and state.data.food[x][y] and (agentState.numCarrying <= capacity):
            agentState.numCarrying += 1

            # do all the food grid maintainenace 
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position

        # Eat capsule
        if isRed: myCapsules = state.getRedCapsules()
        else: myCapsules = state.getBlueCapsules()
        for capsule in myCapsules:
            if position == capsule.getPosition():
                if isRed:
                    state.data.redCapsules.remove( capsule )
                else:
                    state.data.blueCapsules.remove( capsule )
                state.data._capsuleEaten = capsule
                capsule.performAction(agentIndex, state)

    consume = staticmethod( consume )

    def dumpFoodFromDeath(state, agentState):
        if not (DUMP_FOOD_ON_DEATH):
            # this feature is not turned on
            return

        if not agentState.isPacman:
            raise Exception('something is seriously wrong, this agent isnt a pacman!')

        # ok so agentState is this:
        if (agentState.numCarrying == 0):
            return
        
        # first, score changes!
        # we HACK pack that ugly bug by just determining if its red based on the first position
        # to die...
        dummyConfig = Configuration(agentState.getPosition(), 'North')
        isRed = state.isRed(dummyConfig)

        # the score increases if red eats dots, so if we are refunding points,
        # the direction should be -1 if the red agent died, which means he dies
        # on the blue side
        scoreDirection = (-1)**(int(isRed) + 1)
        #state.data.scoreChange += scoreDirection * agentState.numCarrying

        def onRightSide(state, x, y):
            dummyConfig = Configuration((x, y), 'North')
            return state.isRed(dummyConfig) == isRed

        # we have food to dump
        # -- expand out in BFS. Check:
        #   - that it's within the limits
        #   - that it's not a wall
        #   - that no other agents are there
        #   - that no power pellets are there
        #   - that it's on the right side of the grid
        def allGood(state, x, y):
            width, height = state.data.layout.width, state.data.layout.height
            food, walls = state.data.food, state.data.walls

            # bounds check
            if x >= width or y >= height or x <= 0 or y <= 0:
                return False

            if walls[x][y]:
                return False
            if food[x][y]:
                return False

            # dots need to be on the side where this agent will be a pacman :P
            if not onRightSide(state, x, y):
                return False

            if ((x,y) in state.data.redCapsules) or ((x,y) in state.data.blueCapsules):
                return False

            # loop through agents
            agentPoses = [state.getAgentPosition(i) for i in range(state.getNumAgents())]
            if (x,y) in agentPoses:
                return False

            return True

        numToDump = agentState.numCarrying
        if numToDump >= (TOTAL_FOOD/2)-MIN_FOOD:
            agentState.corpseExplosion = True
        state.data.food = state.data.food.copy()
        foodAdded = []

        def genSuccessors(x, y):
            DX = [-1, 0, 1]
            DY = [-1, 0, 1]
            return [(x + dx, y + dy) for dx in DX for dy in DY]

        # BFS graph search
        positionQueue = [agentState.getPosition()]
        seen = set()
        while numToDump > 0:
            if not len(positionQueue):
                raise Exception('Exhausted BFS! uh oh')
            # pop one off, graph check
            popped = positionQueue.pop(0)
            if popped in seen:
                continue
            seen.add(popped)

            x, y = popped[0], popped[1]
            x = int(x)
            y = int(y)
            if (allGood(state, x, y)):
                state.data.food[x][y] = True
                foodAdded.append((x, y))
                numToDump -= 1

            # generate successors
            positionQueue = positionQueue + genSuccessors(x, y)

        state.data._foodAdded = foodAdded
        # now our agentState is no longer carrying food
        agentState.numCarrying = 0
        pass

    dumpFoodFromDeath = staticmethod(dumpFoodFromDeath)

    def checkDeath( state, agentIndex):
        agentState = state.data.agentStates[agentIndex]
        otherTeam = state.getOpponentTeamIndices(agentIndex)

        for otherAgentIndex in otherTeam:
            otherAgentState = state.data.agentStates[otherAgentIndex]
            otherPosition = otherAgentState.getPosition()
            if otherPosition == None: continue
            if manhattanDistance( otherPosition, agentState.getPosition() ) <= COLLISION_TOLERANCE:
                AgentRules.collide(state, agentIndex, otherAgentIndex)
    checkDeath = staticmethod( checkDeath )

    def collide(gameState, index1, index2):
        agentState1 = gameState.data.agentStates[index1]
        agentState2 = gameState.data.agentStates[index2]
        if agentState1.isPacman and not agentState2.isPacman:
            if agentState2.scaredTimer <= 0:
                AgentRules.attemptToKillAgent(gameState, index1, index2)
            else:
                AgentRules.attemptToKillAgent(gameState, index2, index1)
        elif agentState2.isPacman and not agentState1.isPacman:
            if agentState1.scaredTimer <= 0:
                AgentRules.attemptToKillAgent(gameState, index2, index1)
            else:
                AgentRules.attemptToKillAgent(gameState, index1, index2)
    collide = staticmethod( collide )

    def attemptToKillAgent(state, deadAgentIndex, killerAgentIndex):
        deadAgentState = state.data.agentStates[deadAgentIndex]
        if deadAgentState.isRespawning:
            return
        killerAgentState = state.data.agentStates[killerAgentIndex]
        if deadAgentState.isPacman:
            AgentRules.dumpFoodFromDeath(state, deadAgentState)
        killerAgentState.killCount += 1
        deadAgentState.deathCount += 1
        deadAgentState.deathTimer = 5
        score = KILL_POINTS
        if state.isOnRedTeam(deadAgentIndex):
            score = -score
        state.data.scoreChange += score
        deadAgentState.isPacman = False
        deadAgentState.configuration = deadAgentState.start
        deadAgentState.scaredTimer = 0
        deadAgentState.isRespawning = True
        respawnTime = RESPAWN_TIMES[deadAgentState.powers.respawn]
        state.delayAgent(deadAgentIndex, respawnTime)
    attemptToKillAgent = staticmethod( attemptToKillAgent )

    def checkLaserShot(state,shooterIndex):
        """
        Checks for death due to a laser shot.  Assumes that agentIndex
        used a Laser action, and that this action was valid.
        """
        shooterState = state.data.agentStates[shooterIndex]
        assert shooterState.configuration is not None
        shooterPosition = shooterState.configuration.getPosition()
        shooterDirection = shooterState.configuration.getDirection()
        otherTeam = state.getOpponentTeamIndices(shooterIndex)

        for targetIndex in otherTeam:
            targetState = state.data.agentStates[targetIndex]
            hasArmour = targetState.getArmourPower()
            isInvisible = (targetState.configuration is None)
            if not (hasArmour or isInvisible):
                targetPosition = targetState.configuration.getPosition()
                dist = manhattanDistance(shooterPosition,targetPosition)
                if(dist <= LASER_RANGE or shooterState.getLaserPower() > 1):
                    if AgentRules.canShootLaser(shooterPosition, targetPosition, shooterDirection, state.getWalls()):
                        AgentRules.attemptToKillAgent(state, targetIndex, shooterIndex)

    checkLaserShot = staticmethod(checkLaserShot)

    def canShootLaser(shooterPosition, targetPosition, shooterDirection, walls):
        if(util.manhattanDistance(shooterPosition,targetPosition) <= COLLISION_TOLERANCE):
            return False
        (px,py) = shooterPosition
        (gx,gy) = targetPosition
        pxr = int(round(px))
        pyr = int(round(py))
        gxr = int(round(gx))
        gyr = int(round(gy))

        if(abs(px-gx) <= COLLISION_TOLERANCE/2 and py < gy and shooterDirection == Directions.NORTH and not(any([ walls[pxr][y] for y in range(pyr,gyr)]))):
            return True
        if(abs(px-gx) <= COLLISION_TOLERANCE/2 and py > gy and shooterDirection == Directions.SOUTH and not(any(walls[pxr][y] for y in range(gyr,pyr)))):
            return True
        if(px < gx and abs(py - gy) <= COLLISION_TOLERANCE/2 and shooterDirection == Directions.EAST and not(any( walls[x][pyr] for x in range(pxr,gxr)))):
            return True
        if(px > gx and abs(py - gy) <= COLLISION_TOLERANCE/2 and shooterDirection == Directions.WEST and not(any(walls[x][pyr] for x in range(gxr,pxr)))):
            return True
        return False

    canShootLaser = staticmethod(canShootLaser)

    def checkBlast(state,shooterIndex):
        """
        Checks for death due to an explosion.  Assumes that agentIndex
        used a Blast action, and that this action was valid.
        """
        shooterState = state.data.agentStates[shooterIndex]
        assert shooterState.configuration is not None
        shooterPosition = shooterState.configuration.getPosition()
        shooterDirection = shooterState.configuration.getDirection()
        radius = BLAST_RADIUS
        otherTeam = state.getOpponentTeamIndices(shooterIndex)

        for targetIndex in otherTeam:
            targetState = state.data.agentStates[targetIndex]
            hasArmour = targetState.getArmourPower()
            isInvisible = (targetState.configuration is None)

            if not (hasArmour or isInvisible):
                targetPosition = targetState.configuration.getPosition()
                dist = manhattanDistance(shooterPosition,targetPosition)
                if(dist <= radius and dist > COLLISION_TOLERANCE):
                    AgentRules.attemptToKillAgent(state, targetIndex, shooterIndex)

    checkBlast = staticmethod(checkBlast)

#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
    return str + ' [Default: %default]'

def parseAgentArgs(str):
    if str == None or str == '': return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key,val = p, 1
        opts[key] = val
    return opts

def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python capture.py
                    - starts a game with two baseline agents
                (2) python capture.py --keys0
                    - starts a two-player interactive game where the arrow keys control agent 0, and all other agents are baseline agents
                (3) python capture.py -r baselineTeam -b myTeam
                    - starts a fully automated game where the red team is a baseline team and blue team is myTeam
    """
    parser = OptionParser(usageStr)

    parser.add_option('-r', '--red', help=default('Red team'),
                      default='baselineTeam')
    parser.add_option('-b', '--blue', help=default('Blue team'),
                      default='baselineTeam')
    parser.add_option('--red-name', help=default('Red team name'),
                      default='Red')
    parser.add_option('--blue-name', help=default('Blue team name'),
                      default='Blue')
    parser.add_option('--redOpts', help=default('Options for red team (e.g. first=keys)'),
                      default='')
    parser.add_option('--blueOpts', help=default('Options for blue team (e.g. first=keys)'),
                      default='')
    parser.add_option('--keys0', help='Make agent 0 (first red player) a keyboard agent', action='store_true',default=False)
    parser.add_option('--keys1', help='Make agent 1 (second red player) a keyboard agent', action='store_true',default=False)
    parser.add_option('--keys2', help='Make agent 2 (first blue player) a keyboard agent', action='store_true',default=False)
    parser.add_option('--keys3', help='Make agent 3 (second blue player) a keyboard agent', action='store_true',default=False)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout; use RANDOM for a random maze; use RANDOM<seed> to use a specified random seed, e.g., RANDOM23'),
                      metavar='LAYOUT_FILE', default='defaultCapture')
    parser.add_option('-t', '--textgraphics', action='store_true', dest='textgraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-m', '--powerLimit', type='int', dest='powerLimit',
                      help=default('Power limit for agents'), default=0)
    parser.add_option('--capsuleLimit', type='int', dest='capsuleLimit',
                      help=default('Capsule limit for teams'), default=0)
    
    parser.add_option('-q', '--quiet', action='store_true',
                      help='Display minimal output and no graphics', default=False)
    
    parser.add_option('-Q', '--super-quiet', action='store_true', dest="super_quiet",
                      help='Same as -q but agent output is also suppressed', default=False)
    
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom in the graphics'), default=1)
    parser.add_option('-i', '--time', type='int', dest='time',
                      help=default('TIME limit of a game in moves'), default=1200, metavar='TIME')
    parser.add_option('-n', '--numGames', type='int',
                      help=default('Number of games to play'), default=1)
    parser.add_option('-f', '--fixRandomSeed', action='store_true',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0)
    parser.add_option('--record', action='store_true',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', default=None,
                      help='Replays a recorded game file.')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('-c', '--catchExceptions', action='store_true', default=False,
                      help='Catch exceptions and enforce time limits')

    options, otherjunk = parser.parse_args(argv)
    assert len(otherjunk) == 0, "Unrecognized options: " + str(otherjunk)
    args = dict()

    # Choose a display format
    #if options.pygame:
    #    import pygameDisplay
    #    args['display'] = pygameDisplay.PacmanGraphics()
    if options.textgraphics:
        import textDisplay
        args['display'] = textDisplay.PacmanGraphics()
    elif options.quiet:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.super_quiet:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
        args['muteAgents'] = True
    else:
        import captureGraphicsDisplay
        # Hack for agents writing to the display
        captureGraphicsDisplay.FRAME_TIME = 0
        args['display'] = captureGraphicsDisplay.PacmanGraphics(options.red, options.blue, options.zoom, frameTime = options.frameTime, capture=True)
        import __main__
        __main__.__dict__['_display'] = args['display']

    if options.fixRandomSeed: random.seed('cs188')

    # Special case: recorded games don't use the runGames method or args structure
    if options.replay != None:
        print 'Replaying recorded game %s.' % options.replay
        import cPickle
        recorded = cPickle.load(open(options.replay))
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    # Choose a pacman agent
    redArgs, blueArgs = parseAgentArgs(options.redOpts), parseAgentArgs(options.blueOpts)
    if options.numTraining > 0:
        redArgs['numTraining'] = options.numTraining
        blueArgs['numTraining'] = options.numTraining
    nokeyboard = options.textgraphics or options.quiet or options.numTraining > 0
    print '\nRed team %s with %s:' % (options.red, redArgs)
    redTeam = loadTeam(True, options.red, nokeyboard, redArgs)
    print '\nBlue team %s with %s:' % (options.blue, blueArgs)
    blueTeam = loadTeam(False, options.blue, nokeyboard, blueArgs)

    redTeam.name = options.red_name
    args['redTeam'] = redTeam
    blueTeam.name = options.blue_name
    args['blueTeam'] = blueTeam

    args['keys'] = [None for i in range(4)]
    keyboardBools = [options.keys0, options.keys1, options.keys2, options.keys3]
    if keyboardBools.count(True) > 2:
        raise Exception('Max of two keyboard agents supported')
    firstKeyboardAgent = False
    for i, val in enumerate(keyboardBools):
        if val and not firstKeyboardAgent:
            args['keys'][i] = KeyboardAgent(i)
            firstKeyboardAgent = True
        elif val:
            args['keys'][i] = KeyboardAgent2(i)

    # Choose a layout
    import layout
    if options.layout == 'RANDOM':
        args['layout'] = layout.Layout(randomLayout().split('\n'))
    elif options.layout.startswith('RANDOM'):
        args['layout'] = layout.Layout(randomLayout(int(options.layout[6:])).split('\n'))
    elif options.layout.lower().find('capture') == -1:
        raise Exception( 'You must use a capture layout with capture.py')
    else:
        args['layout'] = layout.getLayout( options.layout )

    if args['layout'] == None: raise Exception("The layout " + options.layout + " cannot be found")
    args['length'] = options.time
    args['numGames'] = options.numGames
    args['numTraining'] = options.numTraining
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['powerLimit'] = options.powerLimit
    if options.capsuleLimit % 2 != 0:
      raise Exception('Capsule limit of %d is not even', options.capsuleLimit)
    args['capsuleLimit'] = options.capsuleLimit
    return args

def randomLayout(seed = None):
    if not seed:
        seed = random.randint(0,99999999)
    # layout = 'layouts/random%08dCapture.lay' % seed
    # print 'Generating random layout in %s' % layout
    import mazeGenerator
    return mazeGenerator.generateMaze(seed)

import traceback
def loadTeam(isRed, factory, textgraphics, cmdLineArgs):
    "Calls team factories and returns the team"
    try:
        if not factory.endswith(".py"):
            factory += ".py"

        module = imp.load_source('player' + str(int(isRed)), factory)
    except (NameError, ImportError):
        print >>sys.stderr, 'Error: The team "' + factory + '" could not be loaded! '
        traceback.print_exc()
        return None

    args = dict()
    args.update(cmdLineArgs)  # Add command line args with priority

    print "Loading Team:", factory
    print "Arguments:", args

    # if textgraphics and factoryClassName.startswith('Keyboard'):
    #     raise Exception('Using the keyboard requires graphics (no text display, quiet or training games)')

    try:
        teamClass = getattr(module, 'Team')
    except AttributeError:
        print >>sys.stderr, 'Error: The team "' + factory + '" could not be loaded! '
        traceback.print_exc()
        return None

    team = teamClass()
    # TODO: This is very hacky, and should be improved
    team.fileLocation = os.path.dirname(factory)

    # if isRed:
    #   agents = team.createAgents(0, 2, isRed, **args)
    # else:
    #   agents = team.createAgents(1, 3, isRed, **args)

    # # TODO: This is very hacky, and should be improved
    # agents[0].stateFileLocation= os.path.join(os.path.dirname(factory), "state")
    # agents[1].stateFileLocation= os.path.join(os.path.dirname(factory), "state")
    return team

def replayGame( layout, powers, redCapsules, blueCapsules, actions, display, length, redTeamName, blueTeamName ):
    rules = CaptureRules()
    game = rules.newGame( None, None, layout, float("inf"), float("inf"), display, length, [None for i in range(4)], False, False )
    state = game.state

    for i, powerAssignment in enumerate(powers):
        powerData = AgentPowers(powerAssignment, float("inf"))
        state.data.agentStates[i].powers = powerData
        state.data.agentStates[i].powersAssignments = powerAssignment

    game.initializeAgentObservedVars()
    state.data.redCapsules = redCapsules
    state.data.blueCapsules = blueCapsules

    display.redTeam = redTeamName
    display.blueTeam = blueTeamName
    display.initialize(state.data)

    for agentIndex, action in actions:
        assert agentIndex == state.getNextAgentIndex()
        # Execute the action
        state = state.generateSuccessor( action )
        # Change the display
        display.update( state.data )
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()

def runGames( redTeam, blueTeam, layout, powerLimit, capsuleLimit, display, length, keys, numGames, record, numTraining, muteAgents=False, catchExceptions=False ):

    rules = CaptureRules()
    games = []

    if numTraining > 0:
        print 'Playing %d training games' % numTraining

    for i in range( numGames ):
        beQuiet = i < numTraining
        if beQuiet:
            # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        g = rules.newGame( redTeam, blueTeam, layout, powerLimit, capsuleLimit, gameDisplay, length, keys, muteAgents, catchExceptions )
        g.setup()
        powers = [dict(s.powersAssignments.items()) for s in g.state.data.agentStates]
        redCapsules = [c.deepCopy() for c in g.state.getRedCapsules()]
        blueCapsules = [c.deepCopy() for c in g.state.getBlueCapsules()]
        
        g.run()
        if not beQuiet: games.append(g)

        g.record = None
        if record:
            import time, cPickle, game
            #fname = ('recorded-game-%d' % (i + 1)) +  '-'.join([str(t) for t in time.localtime()[1:6]])
            #f = file(fname, 'w')
            components = {'layout': layout, 'actions': g.moveHistory, 'length': length, 'redTeamName': redTeam.name, 'blueTeamName':blueTeam.name, 'powers' : powers, 'redCapsules' : redCapsules, 'blueCapsules' : blueCapsules}
            #f.close()
            print "recorded"
            g.record = cPickle.dumps(components)
            with open('replay' + str(i),'wb') as f:
                f.write(g.record)

    if numGames > 1:
        scores = [game.state.data.score for game in games]
        redWinRate = [s > 0 for s in scores].count(True)/ float(len(scores))
        blueWinRate = [s < 0 for s in scores].count(True)/ float(len(scores))
        print 'Average Score:', sum(scores) / float(len(scores))
        print 'Scores:       ', ', '.join([str(score) for score in scores])
        print 'Red Win Rate:  %d/%d (%.2f)' % ([s > 0 for s in scores].count(True), len(scores), redWinRate)
        print 'Blue Win Rate: %d/%d (%.2f)' % ([s < 0 for s in scores].count(True), len(scores), blueWinRate)
        print 'Record:       ', ', '.join([('Blue', 'Tie', 'Red')[max(0, min(2, 1 + s))] for s in scores])
    return games

def save_score(game):
    with open('score', 'w') as f:
        print >>f, game.state.data.score

def save_achievements(game):
    achievements = set([])
    moveHistory = game.moveHistory
    RED = 1
    BLUE = 0
    redWon = game.state.data.score > 0
    blueWon = game.state.data.score < 0

    redTeam = game.state.isOnRedTeam

    lastMove = [0, 0]
    for time_step, (agent, action) in enumerate(moveHistory):
        if action != Directions.STOP:
            lastMove[int(redTeam(agent))] = time_step
        if time_step - lastMove[BLUE] >= 600 and blueWon:
            achievements.add((BLUE, 'Stalemates are the best'))
        if time_step - lastMove[RED] >= 600 and redWon:
            achievements.add((RED, 'Stalemates are the best'))

    deathCounts = [0, 0]
    for agent, agentState in enumerate(game.state.data.agentStates):
        deathCounts[int(redTeam(agent))] += agentState.deathCount
        if agentState.deathTimer >= 0:
            achievements.add((int(redTeam(agent)), 'No trace left behind'))
        if agentState.corpseExplosion:
            achievements.add((int(redTeam(agent)), 'Corpse explosion'))
        if agentState.ninjaMode:
            achievements.add((int(redTeam(agent)), 'Ninja mode'))
            
    if deathCounts[BLUE] == 0 and blueWon:
        achievements.add((BLUE, 'Survivor'))
    if deathCounts[RED] == 0 and redWon:
        achievements.add((RED, 'Survivor'))
    if deathCounts[BLUE] >= 3:
        achievements.add((BLUE, 'Behavioral cloning gone wrong'))
        achievements.add((RED, "Grandpac's apprentice"))
    if deathCounts[RED] >= 3:
        achievements.add((RED, 'Behavioral cloning gone wrong'))
        achievements.add((BLUE, "Grandpac's apprentice"))
    
    if redWon and len(game.state.getRedCapsules()) >= 2:
        achievements.add((RED, 'Powerless'))
    if blueWon and len(game.state.getBlueCapsules()) >= 2:
        achievements.add((BLUE, 'Powerless'))
        
    totalPositions = game.state.data.layout.width * game.state.data.layout.height
    totalPositions -= game.state.data.walls.count()

    visitedPositions = [set([]), set([])]
    agentInCorridor = [True]*game.state.getNumAgents()
    for agent, position in game.positionHistory:
        visitedPositions[int(redTeam(agent))].add(position)
        if position[0] != game.state.getInitialAgentPosition(agent)[0]:
            agentInCorridor[agent] = False

    if len(visitedPositions[RED]) / float(totalPositions) >= .90:
        achievements.add((RED, 'Explorer'))
    if len(visitedPositions[BLUE]) / float(totalPositions) >= .90:
        achievements.add((BLUE, 'Explorer'))


    killCounts = []
    returnedFoodCounts = []
    if redWon:
        for index in game.state.getRedTeamIndices():
            if agentInCorridor[index]:
                achievements.add((RED, 'One hand tied'))

            agentState = game.state.data.agentStates[index]
            killCounts.append(agentState.killCount)
            returnedFoodCounts.append(agentState.numReturned)

        if max(killCounts) == 0:
            achievements.add((RED, 'The best defence is no defence'))

        if min(killCounts) >= 1:
            achievements.add((RED, 'Defensive master'))
        if min(returnedFoodCounts) >= 7:
            achievements.add((RED, 'Offensive master'))
        if game.comeback[RED]:
            achievements.add((RED, 'The Pac is bac'))
    elif blueWon:
        for index in game.state.getBlueTeamIndices():
            if agentInCorridor[index]:
                achievements.add((BLUE, 'One hand tied'))

            agentState = game.state.data.agentStates[index]
            killCounts.append(agentState.killCount)
            returnedFoodCounts.append(agentState.numReturned)

        if max(killCounts) == 0:
            achievements.add((BLUE, 'The best defence is no defence'))

        if min(killCounts) >= 1:
            achievements.add((BLUE, 'Defensive master'))
        if min(returnedFoodCounts) >= 7:
            achievements.add((BLUE, 'Offensive master'))

        if game.comeback[BLUE]:
            achievements.add((BLUE, 'The Pac is bac'))
            
    with open('achievements', 'w') as f:
        for team, achievement in achievements:
            print >>f, "%s,%s" %(team, achievement)

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python capture.py

    See the usage string for more details.

    > python capture.py --help
    """
    options = readCommand( sys.argv[1:] ) # Get game components based on input
    games = runGames(**options)
    
    save_score(games[0])
    save_achievements(games[0])
    # import cProfile
    # cProfile.run('runGames( **options )', 'profile')
