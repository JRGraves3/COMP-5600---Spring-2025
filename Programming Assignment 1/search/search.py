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
from game import Directions
from typing import List

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

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
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
    "*** YOUR CODE HERE ***"
    # Initialize frontier as Stack
    fringe = util.Stack()
    # Push starting state and [null] path list on to stack
    fringe.push((problem.getStartState(), []))
    # Initialize a set to keep track of the visited nodes
    visited = set()

    # Loop until the frontier is empty
    while not fringe.isEmpty():
        # Pop the current state and path
        state, path = fringe.pop()

        # Check if the agent is at the goal state
        if problem.isGoalState(state):
            return path # If goal, return path taken
        
        # Explore if unvisited
        if state not in visited:
            # Mark as visited
            visited.add(state)

            # Get successors of current state
            for next_state, action, _ in problem.getSuccessors(state):
                # If successor hasn't been visited, add it to the stack
                if next_state not in visited:
                    # Push the successor and path to the stack
                    fringe.push((next_state, path + [action]))

    return [] # no solution
    # util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize frontier as Queue  
    fringe = util.Queue()
    # Push starting state and [null] path list in to queue
    fringe.push((problem.getStartState(), []))
    # Initialize a set to keep track of the visited nodes
    visited = set()

    # Loop until the frontier is empty
    while fringe:
        # Pop the current state and path
        state, path = fringe.pop()

        # Check if the agent is at the goal state
        if problem.isGoalState(state):
            return path # If goal, return path taken
        
        # Explore if unvisited
        if state not in visited:
            # Mark as visited
            visited.add(state)

            # Get successors of current state
            for successor, action, _ in problem.getSuccessors(state):
                # If successor hasn't been visited, add it to the queue
                if successor not in visited:
                    # Push the successor and path to the queue
                    fringe.push((successor, path + [action]))

    return [] # no solution
    # util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Initialize frontier as PriorityQueue
    fringe = util.PriorityQueue()
    # Push starting state and [null] path list in to queue
    fringe.push((problem.getStartState(), []), 0)
    # Initialize a set to keep track of the visited nodes
    visited = set()

    # Loop until the frontier is empty
    while not fringe.isEmpty():
        # Pop the current state and path
        state, path = fringe.pop()

        # Check if the agent is at the goal state
        if problem.isGoalState(state):
            return path # If goal, return path taken
        
        # Explore if unvisited
        if state not in visited:
            # Mark as visited
            visited.add(state)

            # Get successors of current state
            for successor, action, step_cost in problem.getSuccessors(state):
                # If successor hasn't been visited, add it to the queue
                if successor not in visited:
                    # Get the total cost to successor
                    cost = problem.getCostOfActions(path + [action])
                    # Push the successor, path, and total cost into the queue
                    fringe.push((successor, path + [action]), cost)

    return [] # no solution
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize frontier as PriorityQueue
    fringe = util.PriorityQueue()
    # Push starting state and [null] path list in to queue
    fringe.push((problem.getStartState(), []), 0)
    # Initialize a dictionary for the costs
    current_cost = {}
    current_cost[problem.getStartState()] = 0

    # Loop until the frontier is empty
    while not fringe.isEmpty():
        # Pop the current state and path
        state, path = fringe.pop()

        # Check if the agent is at the goal state
        if problem.isGoalState(state):
            return path # If goal, return path taken
        
        # Get successors of current state
        for successor, action, step_cost in problem.getSuccessors(state):
            # Get the cost to reach the successor
            succ_cost = current_cost[state] + step_cost

            # If successor hasn't been visited or the new cost is better than a previous cost, add to the queue
            if successor not in current_cost or succ_cost < current_cost[successor]:
                # Update the cost for successor
                current_cost[successor] = succ_cost
                # Calculate the priority (f(n) = g(n) + h(n))
                priority = succ_cost + heuristic(successor, problem)
                # Push the successor, path, and priority to the queue
                fringe.push((successor, path + [action]), priority)

    return [] # no solution
    # util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
