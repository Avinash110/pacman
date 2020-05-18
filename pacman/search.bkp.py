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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    stack = util.Stack()
    startState = problem.getStartState()
    stack.push((startState, []))
    statesExpanded = []
    steps = []
    while len(stack.list) != 0:
        currentCoordinates, currentAction = stack.pop()
        statesExpanded.append(currentCoordinates)
        if(problem.isGoalState(currentCoordinates)):
            steps = currentAction
            break
        nextMoves = list(filter(lambda move: move[0] not in statesExpanded, problem.getSuccessors(currentCoordinates)))
        for move in nextMoves:
            coordinates, action, cost = move
            stack.push((coordinates, currentAction + [action]))
    return steps

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    startState = problem.getStartState()
    queue.push((startState, []))
    statesExpanded = [startState]
    steps = []
    while len(queue.list) != 0:
        currentCoordinates, currentAction = queue.pop()
        if(problem.isGoalState(currentCoordinates)):
            steps = currentAction
            break
        nextMoves = list(filter(lambda move: move[0] not in statesExpanded, problem.getSuccessors(currentCoordinates)))
        for move in nextMoves:
            coordinates, action, cost = move
            statesExpanded.append(coordinates)
            queue.push((coordinates, currentAction + [action]))
    return steps
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    import heapq
    queue = util.PriorityQueue()
    startState = problem.getStartState()
    statesExpanded = [startState]

    queue.push((startState, []), 0)
    steps = []
    while len(queue.heap) != 0:
        currentCoordinates, currentAction = queue.pop()
        statesExpanded.append(currentCoordinates)
        if(problem.isGoalState(currentCoordinates)):
            steps = currentAction
            break
        nextMoves = problem.getSuccessors(currentCoordinates)
        for move in nextMoves:
            coordinates, action, cost = move
            tempAction = currentAction + [action]
            costVal = problem.getCostOfActions(tempAction)

            nodesInOpenList = list(map(lambda x:x[2][0], queue.heap))
            if (coordinates not in statesExpanded and coordinates not in nodesInOpenList):
                queue.push((coordinates, tempAction), costVal)
            else:
                for openNode in queue.heap:
                    if(openNode[2][0] == coordinates and costVal < openNode[0]):
                        queue.heap.remove(openNode)
                        queue.push((coordinates, tempAction), costVal)
                        heapq.heapify(queue.heap)
                        
    return steps
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    import heapq
    queue = util.PriorityQueue()
    startState = problem.getStartState()
    statesExpanded = []
    queue.push((startState, []), 0)
    steps = []
    while (len(queue.heap) != 0):
        currentCoordinates, currentAction = queue.pop()
        statesExpanded.append(currentCoordinates)
        if(problem.isGoalState(currentCoordinates)):
            steps = currentAction
            break
        nextMoves = problem.getSuccessors(currentCoordinates)
        for move in nextMoves:
            nodesInOpenList = list(map(lambda x:x[2][0], queue.heap))
            coordinates, action, cost = move
            tempAction = currentAction + [action]
            tempF = problem.getCostOfActions(tempAction) + heuristic(coordinates, problem)
            if (coordinates not in statesExpanded and coordinates not in nodesInOpenList):
                queue.push((coordinates, tempAction), tempF)
            else:
                for openNode in queue.heap:
                    if(openNode[2][0] == coordinates and tempF < openNode[0]):
                        queue.heap.remove(openNode)
                        queue.push((coordinates, tempAction), tempF)
                        heapq.heapify(queue.heap)
    return steps

    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
