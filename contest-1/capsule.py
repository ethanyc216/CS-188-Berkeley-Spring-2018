from events import Event
from config import SCARED_TIME, SONAR_TIME, ARMOUR_TIME, JUGGERNAUT_TIME
import util

class Capsule:
    """
    An abstract class for a power pellet.
    NOTE:  Any instance information can only be added in the __init__
    method and can never be mutated afterwards.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getPosition(self):
        return self.x, self.y

    def deepCopy(self):
        util.raiseNotDefined()

    def __eq__(self, other):
        util.raiseNotDefined()

    def __hash__(self):
        util.raiseNotDefined()


    def performAction(self, agentIndex, state):
        """
        Called when the agentIndex'th agent eats the capsule.  The
        capsule has already been removed from the state.
        """
        util.raiseNotDefined()

class PositionCapsule(Capsule):
    """
    A capsule whose state is only its location.  Subclasses should
    override only the performAction method and nothing else.
    """
    def deepCopy(self):
        return self.__class__(self.x, self.y)

    def __hash__(self):
        return hash( (self.x, self.y) )

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.x == other.x and self.y == other.y

class ScareCapsule(PositionCapsule):
    def performAction(self, agentIndex, state):
        if state.isOnRedTeam(agentIndex):
            otherTeam = state.getBlueTeamIndices()
        else:
            otherTeam = state.getRedTeamIndices()

        for otherAgentIndex in otherTeam:
            otherAgentState = state.data.agentStates[otherAgentIndex]
            otherAgentState.scaredTimer = SCARED_TIME

class GrenadeCapsule(PositionCapsule):
    def performAction(self, agentIndex, state):
        agentState = state.data.agentStates[agentIndex]
        agentState.powers.blast += 1

class SonarCapsule(PositionCapsule):
    def performAction(self, agentIndex, state):
        agentState = state.data.agentStates[agentIndex]
        agentState.powers.sonar = 1
        agentState.sonarTimer = SONAR_TIME

class ArmourCapsule(PositionCapsule):
    def performAction(self, agentIndex, state):
        agentState = state.data.agentStates[agentIndex]
        agentState.powers.armour = 1
        agentState.armourTimer = ARMOUR_TIME

class JuggernautCapsule(PositionCapsule):
    def performAction(self, agentIndex, state):
        agentState = state.data.agentStates[agentIndex]
        agentState.powers.juggernaut = 1
        agentState.juggernautTimer = JUGGERNAUT_TIME

class WallCapsule(Capsule):
    def __init__(self, x, y, wallPositions, time):
        Capsule.__init__(self, x, y)
        self.wallPositions = tuple(wallPositions)
        self.timeForWalls = time

    def performAction(self, agentIndex, state):
        agentPositions = [s.getPosition() for s in state.data.agentStates]
        for x, y in self.wallPositions:
            if (x, y) not in agentPositions:
                state.data.walls[x][y] = True
                state.data.timedWalls[(x, y)] = self.timeForWalls
                state.data._wallsChanged.append((x, y))

    def deepCopy(self):
        return WallCapsule(self.x, self.y, self.wallPositions, self.timeForWalls)

    def __hash__(self):
        return hash( (self.x, self.y, self.wallPositions, self.timeForWalls) )

    def __eq__(self, other):
        return isinstance(other, WallCapsule) and \
            self.x == other.x and self.y == other.y and \
            self.wallPositions == other.wallPositions and \
            self.timeForWalls == other.timeForWalls


class TimerDecrementEvent(Event):
    def trigger(self, state):
        for agentState in state.data.agentStates:
            scaredTimer = agentState.scaredTimer
            if scaredTimer == 1 and agentState.configuration is not None:
                agentState.configuration.pos = util.nearestPoint( agentState.configuration.pos )
            agentState.scaredTimer = max( 0, scaredTimer - 1 )

            sonarTimer = agentState.sonarTimer
            if sonarTimer == 1:
                agentState.powers.sonar = 0
            agentState.sonarTimer = max( 0, sonarTimer - 1 )

            armourTimer = agentState.armourTimer
            if armourTimer == 1:
                agentState.powers.armour = 0
            agentState.armourTimer = max( 0, armourTimer - 1 )

            juggernautTimer = agentState.juggernautTimer
            if juggernautTimer == 1:
                agentState.powers.juggernaut = 0
            agentState.juggernautTimer = max( 0, juggernautTimer - 1 )

        # Repeat this work after 1 more timestep
        state.registerEventWithDelay(self, 1)

    def deepCopy(self):
        return TimerDecrementEvent(self.eventId)

    def __eq__( self, other ):
        return isinstance(other, TimerDecrementEvent) and \
            self.eventId == other.eventId

    def __hash__( self ):
        return hash(self.eventId)

DefaultCapsule = ScareCapsule
LEGAL_CAPSULES = [ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule]
