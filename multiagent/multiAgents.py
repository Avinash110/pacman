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
import math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newGhostStatesPosition = [ghostState.getPosition() for ghostState in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        newFoodList = newFood.asList()

        if (newPos in newGhostStatesPosition):
            return -1
        else:
            minDistance = math.inf
            minDistanceGhostAway = min(util.manhattanDistance(ghostP, newPos) for ghostP in newGhostStatesPosition)
            if(action == "Stop"):
                return successorGameState.getScore() - minDistanceGhostAway
            else:
                for food in newFoodList:
                    minDistance = min(minDistance, util.manhattanDistance(food, newPos))
                return successorGameState.getScore() + (2/minDistance) - (0.1/minDistanceGhostAway)

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
        "*** YOUR CODE HERE ***"
        isAgentPacman = self.index == 0
        bestAction = self.recursiveFunction(gameState, None, 0)
        return bestAction[0]
        # util.raiseNotDefined()

    def recursiveFunction(self, gameState, action, recursiveIndex):

        currentAgentIndex = recursiveIndex % gameState.getNumAgents()
        isAgentPacman = (currentAgentIndex) == 0
        if(recursiveIndex == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
            return (action, self.evaluationFunction(gameState))

        legalActions = gameState.getLegalActions(currentAgentIndex)
        bestStateAndAction = (None, -math.inf) if isAgentPacman else (None, math.inf)
        for action in legalActions:
            tempState = gameState.generateSuccessor(currentAgentIndex, action)
            bestAction, rec = self.recursiveFunction(tempState, action, recursiveIndex+1)
            if isAgentPacman:
                if(rec > bestStateAndAction[1]):
                    bestStateAndAction = (action, rec)
            else:
                if(rec < bestStateAndAction[1]):
                    bestStateAndAction = (action, rec)
        return bestStateAndAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestAction = self.recursiveFunction(gameState, -math.inf, math.inf, None, 0)
        return bestAction[0]
        util.raiseNotDefined()
    def recursiveFunction(self, gameState, alpha, beta, action, recursiveIndex):

        currentAgentIndex = recursiveIndex % gameState.getNumAgents()
        isAgentPacman = (currentAgentIndex) == 0
        if(recursiveIndex == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
            return (action, self.evaluationFunction(gameState))

        legalActions = gameState.getLegalActions(currentAgentIndex)
        bestStateAndAction = (None, -math.inf) if isAgentPacman else (None, math.inf)
        for action in legalActions:
            tempState = gameState.generateSuccessor(currentAgentIndex, action)
            bestAction, rec = self.recursiveFunction(tempState, alpha, beta, action, recursiveIndex+1)
            if isAgentPacman:
                if(rec > beta):
                    return (action, rec)
                alpha = max(alpha, rec)
                if(rec > bestStateAndAction[1]):
                    bestStateAndAction = (action, rec)
            else:
                if(rec < alpha):
                    return (action, rec)
                beta = min(beta, rec)
                if(rec < bestStateAndAction[1]):
                    bestStateAndAction = (action, rec)
        # print("index: ",recursiveIndex, " alpha : ",alpha, " beta : ",beta)
        return bestStateAndAction

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
        "*** YOUR CODE HERE ***"
        return 'Left'
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return 1/len(currentGameState.getFood())
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction