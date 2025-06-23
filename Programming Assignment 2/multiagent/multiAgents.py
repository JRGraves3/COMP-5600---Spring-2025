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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        foodPos = newFood.asList()
        closeFood = -1
        for food in foodPos:
            distance = util.manhattanDistance(newPos, food)
            if closeFood >= distance or closeFood == -1:
                closeFood = distance

        ghostDist = 1
        ghostProx = 0
        for ghostState in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghostState)
            ghostDist += distance
            if distance <= 1:
                ghostProx += 1
        
        return successorGameState.getScore() + (1 / float(closeFood)) - (1 / float(ghostDist)) - ghostProx

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts)

    def maximize(self, gameState, depth, ghosts):
        """
        maximizing agent in minimax
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            tempVal = self.minimize(succ, depth, 1, ghosts)
            if tempVal > maxVal:
                maxVal = tempVal
                bestAction = action

        # terminal max returns actions whereas intermediate max returns values
        if depth > 1:
            return maxVal
        return bestAction

    def minimize(self, gameState, depth, agentIndex, ghosts):
        """
        minimizing agent in minimax 
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        minVal = float("inf")
        legalActions = gameState.getLegalActions(agentIndex)
        succs = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        if agentIndex == ghosts:
            if depth < self.depth:
                for succ in succs:
                    minVal = min(minVal, self.maximize(succ, depth + 1, ghosts))
            else:
                for succ in succs:
                    minVal = min(minVal, self.evaluationFunction(succ))
        else:
            for succ in succs:
                minVal = min(minVal, self.minimize(succ, depth, agentIndex + 1, ghosts))
        return minVal
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maximize(self, gameState, depth, ghosts, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        maxVal = float("-inf")
        bestAction = Directions.STOP
        
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            temp = self.minimize(succ, depth, 1, ghosts, alpha, beta)
            if temp > maxVal:
                maxVal = temp
                bestAction = action
            
            if maxVal > beta:
                return maxVal
            alpha = max(alpha, maxVal)

        if depth > 1:
            return maxVal
        return bestAction

    def minimize(self, gameState, depth, agentIndex, ghosts, alpha, beta):
        """
        minimizing agent with alpha-beta pruning
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        minVal = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == ghosts:
                if depth < self.depth:
                    temp = self.maximize(succ, depth + 1, ghosts, alpha, beta)
                else:
                    temp = self.evaluationFunction(succ)
            else:
                temp = self.minimize(succ, depth, agentIndex + 1, ghosts, alpha, beta)
            if temp < minVal:
                minVal = temp

            if minVal < alpha:
                return minVal
            beta = min(beta, minVal)
        return minVal

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, ghosts, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, depth, ghosts):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if ghosts == 0:
                return max(expectimax(gameState.generateSuccessor(ghosts, action), depth, 1) for action in gameState.getLegalActions(ghosts))
            else:
                nextA = ghosts + 1
                if gameState.getNumAgents() == nextA:
                    nextA = 0
                if nextA == 0:
                    depth += 1
                return sum(expectimax(gameState.generateSuccessor(ghosts, action), depth, nextA) for action in gameState.getLegalActions(ghosts)) / float(len(gameState.getLegalActions(ghosts)))

        """Performing maximizing task for the root node i.e. pacman"""
        maxVal = float("-inf")
        bestAction = Directions.WEST
        for action in gameState.getLegalActions(0):
            utility = expectimax(gameState.generateSuccessor(0, action), 0, 1)
            if utility > maxVal or maxVal == float("-inf"):
                maxVal = utility
                bestAction = action

        return bestAction




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''Calculatin the distance to the closest food pellet'''
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    minFoodDist = -1
    for food in newFoodList:
        dist = util.manhattanDistance(newPos, food)
        if minFoodDist >= dist or minFoodDist == -1:
            minFoodDist = dist

    '''Calculating the distances from pacman to the ghosts. Checking for the proximity of the ghosts around pacman.'''
    distGhosts = 1
    proxGhosts = 0
    for ghostState in currentGameState.getGhostPositions():
        dist = util.manhattanDistance(newPos, ghostState)
        distGhosts += dist
        if dist <= 1:
            proxGhosts += 1

    '''Obtaining the number of capsules available'''
    newCap = currentGameState.getCapsules()
    numCaps = len(newCap)

    '''Combination of the above calculated metrics.'''
    return currentGameState.getScore() + (1 / float(minFoodDist)) - (1 / float(distGhosts)) - proxGhosts - numCaps

# Abbreviation
better = betterEvaluationFunction
