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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhosts = successorGameState.getGhostPositions()
        pelletsRemaining = len(newFood) - 1

        # calculate distance to closest food
        if len(newFood) > 0:
            closestFood = manhattanDistance(newPos, newFood[0])
            for (food) in newFood:
                dist = manhattanDistance(newPos, food);
                if dist < closestFood:
                    closestFood = dist
        else:
            closestFood = 0.00001

        # calculate distance to closest ghost
        minGhostDist = 10000
        for (ghost) in newGhosts:
            dist = manhattanDistance(newPos, ghost);
            if dist < minGhostDist:
                minGhostDist = dist

        score = 1.0/closestFood - pelletsRemaining  # go toward closest food

        if minGhostDist < 2:
            score = score - 100000  # run away if a ghost is too close

        return score


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

    # pacman
    def max_val(self, gameState, agentIndex, numAgents, atDepth):
        legalActions = gameState.getLegalActions(agentIndex)

        # base case, return score and dummy action
        if atDepth == self.depth or not legalActions:
            return self.evaluationFunction(gameState), ""

        # calculate max of all real successor values
        else:
            max = -10000000000.0
            maxAction = ""
            for (action) in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                score, dummy_action = self.min_val(successorState, agentIndex + 1, numAgents, atDepth)
                if score > max:
                    max = score
                    maxAction = action
            return max, maxAction

    # ghost
    def min_val(self, gameState, agentIndex, numAgents, atDepth):
        legalActions = gameState.getLegalActions(agentIndex)

        # base case, return score and dummy action
        if atDepth == self.depth or not legalActions:
            return self.evaluationFunction(gameState), ""

        # last ghost, next agent index will be 0 and depth will increment
        if agentIndex == numAgents - 1:
            atDepth = atDepth + 1
            agentIndex = -1

        # calculate min of all real successor values
        min = 100000000000.0
        minAction = ""
        for (action) in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == 0:
                # next agent might be pacman
                score, dummy_action = self.max_val(successorState, agentIndex + 1, numAgents, atDepth)
            else:
                # or another ghost
                score, dummy_action = self.min_val(successorState, agentIndex + 1, numAgents, atDepth)
            if score < min:
                min = score
                minAction = action

        return min, minAction

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game s tate after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        numAgents = gameState.getNumAgents()
        agentIndex = 0
        atDepth = 0
        score, action = self.max_val(gameState, agentIndex, numAgents, atDepth)  # minimax value of first pacman
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_val(self, gameState, agentIndex, numAgents, atDepth, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)

        if atDepth == self.depth or not legalActions:
            return self.evaluationFunction(gameState), ""

        else:  # pacman
            max = -10000000000.0
            maxAction = ""
            for (action) in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                score, dummy_action = self.min_val(successorState, agentIndex + 1, numAgents, atDepth, alpha, beta)
                if score > max:
                    max = score
                    maxAction = action
                # prune on min bound
                if max > beta:
                    return max, maxAction
                # update max bound
                if alpha < max:
                    alpha = max

            return max, maxAction

    def min_val(self, gameState, agentIndex, numAgents, atDepth, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)

        if atDepth == self.depth or not legalActions:
            return self.evaluationFunction(gameState), ""

        if agentIndex == numAgents - 1:
            atDepth = atDepth + 1
            agentIndex = -1
        min = 100000000000.0
        minAction = ""
        for (action) in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == 0:
                score, dummy_action = self.max_val(successorState, agentIndex + 1, numAgents, atDepth, alpha, beta)
            else:
                score, dummy_action = self.min_val(successorState, agentIndex + 1, numAgents, atDepth, alpha, beta)
            if score < min:
                min = score
                minAction = action
            # prune on max bound
            if min < alpha:
                return min, minAction
            # update min bound
            if beta > min:
                beta = min

        return min, minAction

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """

        numAgents = gameState.getNumAgents()
        agentIndex = 0
        atDepth = 0
        #print "Depth of this problem is: ", self.depth
        alpha = -10000000000.0
        beta = 10000000000.0
        score, action = self.max_val(gameState, agentIndex, numAgents, atDepth, alpha, beta)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max_val(self, gameState, agentIndex, numAgents, atDepth):
        """legalActions = gameState.getLegalActions(agentIndex)

        if atDepth == self.depth or not legalActions:
            return self.evaluationFunction(gameState), ""

        else:  # pacman
            max = -10000000000.0
            maxAction = ""
            for (action) in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                score, dummy_action = self.exp_val(successorState, agentIndex + 1, numAgents, atDepth)
                if score > max:
                    max = score
                    maxAction = action

            return max, maxAction"""

    def exp_val(self, gameState, agentIndex, numAgents, atDepth):
        """legalActions = gameState.getLegalActions(agentIndex)

        if atDepth == self.depth or not legalActions:
            return self.evaluationFunction(gameState), ""

        if agentIndex == numAgents - 1:
            atDepth = atDepth + 1
            agentIndex = -1
        # equal probability for all actions (1/total successors)
        prob = 1.0/len(legalActions)
        expect = 0.0
        for (action) in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == 0:
                score, dummy_action = self.max_val(successorState, agentIndex + 1, numAgents, atDepth)
            else:
                score, dummy_action = self.exp_val(successorState, agentIndex + 1, numAgents, atDepth)
            # calculate total expected value of node
            expect = expect + score*prob

        return expect, """""

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        """numAgents = gameState.getNumAgents()
        agentIndex = 0
        atDepth = 0
        score, action = self.max_val(gameState, agentIndex, numAgents, atDepth)
        return action"""


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Add to the score a factor of the distance to each ghost.
      The farther away all the ghosts from Pacman, the higher the evaluation.
      If the ghost is within 1 away, the score will be heavily decreased.
      *Except* when the scared timers > 0! In this case, the closer the ghost, the higher the evaluation (by a lot).
      Subtraact a factor of the distance to all ghosts.
      If the scared ghost is within 1 away, the score will be heavily increased.

      Subtract a factor of the distance to the closest capsule.
      The closer a capsule to Pacman, the higher the evaluation.
      If the capsule is within 1 away, the score will be heavily increased.

      Subtract a factor of the number of food and capsules remaining. The less food and capsules in the state, the
      higher its evaluation.

      Subtract a factor of the distance to the closest pellet (if true distance is 0, calculated
      distance will go to 0.00000001 to prevent div/0 errors). The closer the closest pellet, the higher the evaluation.


    """

    # Useful information you can extract from a GameState (pacman.py)
    pacPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsulePositions = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    foodRemaining = currentGameState.getNumFood()
    capsulesRemaining = len(capsulePositions)

    evaluation = 0.0

    totalGhostDistance = 0.00000001
    withinOneGhost = 0.0
    scaredFactor = 1.0
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        if ghostState.scaredTimer > 0:
            scaredFactor = -1.0
        dist = manhattanDistance(pacPos, ghostPos)
        if dist == 0:
            dist = 0.00000001
        if dist < 2.0:
            withinOneGhost = 1000000000000000000.0
        totalGhostDistance += dist
    evaluation += totalGhostDistance * 200.0 * scaredFactor
    evaluation += -1 * withinOneGhost * scaredFactor

    if scaredFactor == -1.0: #ghost-busting
        closestGhost = 10000000000.0
        for ghostState in ghostStates:
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(pacPos, ghostPos)
            if dist == 0:
                dist = 0.00000001
            if dist < closestGhost:
                closestGhost = dist
        #the smaller the closestGhost, the higher the evaluation
        evaluation += -1 * closestGhost * 2000000.0

    totalCapsuleDistance = 0.00000001
    capsuleEaten = 0.0
    for capsulePos in capsulePositions:
        dist = manhattanDistance(pacPos, capsulePos)
        if dist == 0:
            capsuleEaten = 100000000.0
        totalCapsuleDistance += dist
    evaluation += -2000.0*totalCapsuleDistance
    evaluation += capsuleEaten

    evaluation += -100000.0*foodRemaining
    evaluation += -1000.0*capsulesRemaining

    closestDist = 10000000000.0
    if not foodList:
        return evaluation
    for food in foodList:
        dist = manhattanDistance(pacPos, food)
        if dist == 0:
            dist = 0.00000001
        if dist < closestDist:
            closestDist = dist
    evaluation += -1000.0*closestDist

    return evaluation + scoreEvaluationFunction(currentGameState)






















# Abbreviation
better = betterEvaluationFunction

