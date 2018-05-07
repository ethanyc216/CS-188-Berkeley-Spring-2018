# search.py
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class Node(object):
    def __init__(self, state, state_list, path, cost):
        self.state = state
        self.state_list = state_list
        self.path = path 
        self.cost = cost

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
        Start: (5, 5)
        Is the start a goal? False
        Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    """

    #closed_Set = set()
    stack = util.Stack()

    start_Node =Node(problem.getStartState(), [problem.getStartState()], [], 0) #state, state_list, path, cost
    stack.push(start_Node)

    while not stack.isEmpty():
        current_Node = stack.pop()
        current_state = current_Node.state
        current_state_list = current_Node.state_list
        if problem.isGoalState(current_state):
            return current_Node.path
        else:
            next_Nodes = problem.getSuccessors(current_state)
            for next_node in next_Nodes:
                next_state = next_node[0]
                if next_state not in current_state_list:
                    next_state_list = current_state_list[:]
                    next_state_list.append(next_node[0])
                    next_path = current_Node.path[:]
                    next_path.append(next_node[1])
                    next_cost = current_Node.cost + next_node[2]
                    next_Node = Node(next_state, next_state_list, next_path, next_cost)
                    stack.push(next_Node)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    closed_Set = set()
    queue = util.Queue()

    start_Node =Node(problem.getStartState(), [problem.getStartState()], [], 0) #state, state_list #empty here, path, cost
    queue.push(start_Node)

    while not queue.isEmpty():
        current_Node = queue.pop()
        closed_Set.add(current_Node.state)
        current_state = current_Node.state
        current_state_list = current_Node.state_list
        if problem.isGoalState(current_state):
            return current_Node.path
        else:
            next_Nodes = problem.getSuccessors(current_state)
            for next_node in next_Nodes:
                next_state = next_node[0]
                if (next_state not in closed_Set) and (next_state not in current_state_list):
                    closed_Set.add(next_state)
                    next_state_list = current_state_list[:]
                    next_state_list.append(next_node[0])
                    next_path = current_Node.path[:]
                    next_path.append(next_node[1])
                    next_cost = current_Node.cost + next_node[2]
                    next_Node = Node(next_state, next_state_list, next_path, next_cost)
                    queue.push(next_Node)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    closed_Set = set()
    queue = util.PriorityQueue()

    start_Node =Node(problem.getStartState(), [problem.getStartState()], [], 0) #state, state_list #empty here, path, cost
    queue.push(start_Node, 0)

    while not queue.isEmpty():
        current_Node = queue.pop()
        closed_Set.add(current_Node.state)
        current_state = current_Node.state
        current_state_list = current_Node.state_list
        if problem.isGoalState(current_state):
            return current_Node.path
        else:
            next_Nodes = problem.getSuccessors(current_state)
            for next_node in next_Nodes:
                next_state = next_node[0]
                if (next_state not in closed_Set) and (next_state not in current_state_list):
                    if not problem.isGoalState(next_state):
                        closed_Set.add(next_state)
                    next_state_list = current_state_list[:]
                    next_state_list.append(next_node[0])
                    next_path = current_Node.path[:]
                    next_path.append(next_node[1])
                    next_cost = current_Node.cost + next_node[2]
                    next_Node = Node(next_state, next_state_list, next_path, next_cost)
                    queue.push(next_Node, next_cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    closed_Set = set()
    queue = util.PriorityQueue()

    start_Node =Node(problem.getStartState(), [problem.getStartState()], [], 0) #state, state_list #empty here, path, cost
    queue.push(start_Node, 0)

    while not queue.isEmpty():
        current_Node = queue.pop()
        closed_Set.add(current_Node.state)
        current_state = current_Node.state
        current_state_list = current_Node.state_list
        if problem.isGoalState(current_state):
            return current_Node.path
        else:
            next_Nodes = problem.getSuccessors(current_state)
            for next_node in next_Nodes:
                next_state = next_node[0]
                if (next_state not in closed_Set) and (next_state not in current_state_list):
                    if not problem.isGoalState(next_state):
                        closed_Set.add(next_state)
                    next_state_list = current_state_list[:]
                    next_state_list.append(next_node[0])
                    next_path = current_Node.path[:]
                    next_path.append(next_node[1])
                    next_cost = current_Node.cost + next_node[2]
                    next_Node = Node(next_state, next_state_list, next_path, next_cost)
                    next_heuristic_cost = heuristic(next_state, problem)
                    queue.push(next_Node, next_cost + next_heuristic_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
