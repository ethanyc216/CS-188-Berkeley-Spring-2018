# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        for ghostState in newGhostStates:
            if ghostState.scaredTimer > 0:
                pass
            elif manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return float('-inf')

        disFood = [manhattanDistance(food, newPos) for food in newFood.asList()]

        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            return float('inf')
        else:
            return - 10 * min(disFood) 

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        #get pacman legal actions
        ############
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        successorStates = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        scores = [self.minimizer(state, 0, 1) for state in successorStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def maximizer(self, gameState, currentDepth):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        successorStates = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        scores = [self.minimizer(state, currentDepth, 1) for state in successorStates]
        return max(scores)

    def minimizer(self, gameState, currentDepth, agentIndex):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        successorStates = [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        if agentIndex == gameState.getNumAgents() - 1:
            scores = [self.maximizer(state, currentDepth + 1) for state in successorStates]
        else:
            scores = [self.minimizer(state, currentDepth, agentIndex + 1) for state in successorStates]
        return min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-inf')
        beta = float('inf')
        score = float('-inf')

        # Choose one of the best actions
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            minScore = self.minimizer(successorState, 0, 1, alpha, beta)
            if minScore > score:
                score = minScore
                move = action
            alpha = max(alpha, minScore)
        return move

    def maximizer(self, gameState, currentDepth, alpha, beta):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        maxScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            maxScore = max(maxScore, self.minimizer(successorState, currentDepth, 1, alpha, beta))
            if maxScore > beta:
                return maxScore
            alpha = max(alpha, maxScore)
        return maxScore

    def minimizer(self, gameState, currentDepth, agentIndex, alpha, beta):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        minScore = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                minScore = min(minScore, self.maximizer(successorState, currentDepth + 1, alpha, beta))
            else:
                minScore = min(minScore, self.minimizer(successorState, currentDepth, agentIndex +1, alpha, beta))
            if minScore < alpha:
                return minScore
            beta = min(minScore, beta)
        return minScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        successorStates = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        scores = [self.avgValue(state, 0, 1) for state in successorStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def maximizer(self, gameState, currentDepth):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        successorStates = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        scores = [self.avgValue(state, currentDepth, 1) for state in successorStates]
        return max(scores)

    def avgValue(self, gameState, currentDepth, agentIndex):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        successorStates = [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        if agentIndex == gameState.getNumAgents() - 1:
            scores = [self.maximizer(state, currentDepth + 1) for state in successorStates]
        else:
            scores = [self.avgValue(state, currentDepth, agentIndex + 1) for state in successorStates]
        return sum(scores)/len(scores)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      The score consists of 3 parts: current game score, distance to the nearest food and the ghost state.
      The Pacman has piority to eat the nearest food. 
      And when the ghost is scared, the Pacman the situation would be good, as well as when ghost is far enough.
      When the ghost is really close, the situation is bad and will have -inf score.
    """
    currentScore = scoreEvaluationFunction(currentGameState)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghostEval = 0
    for ghostState in newGhostStates:
        if ghostState.scaredTimer > 0:
            ghostEval = 200
        else:
            if manhattanDistance(ghostState.getPosition(), newPos) <2:
                ghostEval += float('-inf')
            else:
                ghostEval += manhattanDistance(ghostState.getPosition(), newPos) + 40

    disFood = [manhattanDistance(food, newPos) for food in newFood.asList()]
    disFood.append(0)
    return currentScore - 10 * min(disFood) + ghostEval
    
# Abbreviation
better = betterEvaluationFunction

