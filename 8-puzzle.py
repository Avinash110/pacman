#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import math
import heapq


# In[2]:


def getAvailableMoves(row, column):
    moves = []
    if(row > 0):# and lastMove != "down"):
        moves.append('up')
    if(row < 2):# and lastMove != "up"):
        moves.append('down')
    if(column > 0):# and lastMove != "right"):
        moves.append('left')
    if(column < 2):# and lastMove != "left"):
        moves.append('right')
    return moves


# In[3]:


def getManhattanDistance(board):
    goalState = {1: [0,0],2: [0,1], 3: [0,2], 4: [1,2], 5: [2,2], 6: [2,1], 7: [2,0], 8: [1,0]}
#     goalState = {1: [0,0],2: [0,1], 3: [0,2], 4: [1,0], 5: [1,1], 6: [1,2], 7: [2,0], 8: [2,1]}
    currentState = {}
    for i in range(0,3):
        for j in range(0,3):
            currentState[board[i][j]] = [i, j]
    totalManhattanDistance = 0
    for number in goalState:
        totalManhattanDistance += abs(goalState[number][0] - currentState[number][0]) + abs(goalState[number][1] - currentState[number][1])
    return totalManhattanDistance


# In[4]:


def getCurrentPosition(board):
    row =-1
    column = -1
    for i in range(0,3):
        for j in range(0,3):
            if(board[i][j] == ''):
                row = i
                column = j
    return [row, column]


# In[35]:


def expandNode(board):
    currentPosition = getCurrentPosition(board)
    availableMoves = getAvailableMoves(currentPosition[0], currentPosition[1])
    children = []
    for move in availableMoves:
        children.append([doMove(board, move, currentPosition[0], currentPosition[1]), move])
    return children
            


# In[36]:


def doMove(board, move, row, column):
    board = copy.deepcopy(board)
    if(move == 'up'):
        board[row][column] = board[row - 1][column]
        board[row - 1][column] = ''
    if(move == 'down'):
        board[row][column] = board[row + 1][column]
        board[row + 1][column] = ''
    if(move == 'left'):
        board[row][column] = board[row][column - 1]
        board[row][column - 1] = ''
    if(move == 'right'):
        board[row][column] = board[row][column + 1]
        board[row][column + 1] = ''
    return board


# In[46]:


def solveBoard(board):
    queue = []
    heapq.heappush(queue, (0, 0, str(board), ''))
    goalNode = None
    while (1):
        nodeValues = heapq.heappop(queue)
        currNode = eval(nodeValues[2])
        if(getManhattanDistance(currNode) == 0):
            goalNode = nodeValues 
            break
        children = expandNode(currNode)
        for child in children:
            heapq.heappush(queue, (getManhattanDistance(child[0]) + 1 + nodeValues[1], nodeValues[1] + 1, str(child[0]), nodeValues[3]+" "+child[1][0]))
    return goalNode


# In[47]:


# board = [[1,3,4],[8,6,2],['',7,5]]
board = [[3,6,4],['',1,2],[8,7,5]]
# print(checkGoal(board))

# availableMoves(2, 2)

print(solveBoard(board))


# In[ ]:



# a* star
# queue = util.PriorityQueue()
#     startState = problem.getStartState()
#     statesExpanded = []
#     currentF = heuristic(startState, problem)
#     queue.push((startState, []), currentF)
#     steps = []
#     while (1):
#         currentCoordinates, currentAction = queue.pop()
#         if(problem.isGoalState(currentCoordinates)):
#             steps = currentAction
#             break
#         if(currentCoordinates not in statesExpanded):
#             nextMoves = problem.getSuccessors(currentCoordinates)
#             for move in nextMoves:
#                 coordinates, action, cost = move
#                 if(coordinates not in statesExpanded):
#                     tempActions = currentAction + [action]
#                     tempF = problem.getCostOfActions(tempActions) + heuristic(coordinates, problem)
#                     queue.push((coordinates, tempActions), tempF)
#         statesExpanded.append(currentCoordinates)
#     return steps