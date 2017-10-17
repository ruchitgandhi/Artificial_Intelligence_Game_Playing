import numpy
import time
import csv

start_time = time.time()


class State:
    def __init__(self, matrix=None, score=0):
        self.matrix = matrix
        self.score = score

    score = 0


class Successor:
    def __init__(self, rowNum, colNum, score, nextMatrix, emptyColumns):
        self.rowNum = rowNum
        self.colNum = colNum
        self.score = score
        self.nextMatrix = nextMatrix
        self.emptyColumns = emptyColumns


def getKey(successor):
    return successor.score


def readInputFromFile(inputFileName):
    with open(inputFileName) as file:
        n = int(file.readline().rstrip('\n'))
        typesOfFruits = int(file.readline().rstrip('\n'))
        availableTime = float(file.readline().rstrip('\n'))
        inputMatrix = [[] for i in range(n)]
        i = 0
        for line in file.readlines():
            line = line.rstrip('\n')
            inputMatrix[i] = list(map(str, line))
            i += 1
        return n, typesOfFruits, availableTime, inputMatrix


def terminalState(matrix):
    for i in range(n):
        for j in range(n):
            if not matrix[i][j] == -1:
                return False
    return True


def isValidPosition(matrix, row, column, visited, fruitType):
    if row < n and row >= 0 and column < n and column >= 0 and not visited[row][column] and matrix[row][
        column] == fruitType:
        return True
    return False


def shiftCellsDown(matrix, col, upperRow, lowerRow):
    counter = lowerRow - upperRow
    while counter >= 0 and upperRow >= 0:
        matrix[upperRow][col], matrix[lowerRow][col] = matrix[lowerRow][col], matrix[upperRow][col]
        counter = counter - 1
        upperRow = upperRow - 1
        lowerRow = lowerRow - 1

    return matrix


def balanceState(matrix, emptyColumns):
    for col in emptyColumns:
        for row in range(n - 1, 0, -1):
            if matrix[row][col] == -1:
                k = row - 1
                while k >= 0 and matrix[k][col] == -1:
                    k = k - 1
                if matrix[k][col] != -1:
                    matrix = shiftCellsDown(matrix, col, k, row)
    return matrix


def getMaxRegionLength(matrix, row, column, visited):
    global count
    count = 1
    empty_columns = set()
    depthFirstSearch(matrix, row, column, visited, empty_columns)
    return count, matrix, empty_columns


def depthFirstSearch(matrix, row, column, visited, emptyColumns):
    global count
    visited[row][column] = True
    fruit_type = matrix[row][column]
    matrix[row][column] = -1
    emptyColumns.add(column)

    for i in range(4):
        if isValidPosition(matrix, row + rowNeighbours[i], column + colNeighbours[i], visited, fruit_type):
            count = count + 1
            depthFirstSearch(matrix, row + rowNeighbours[i], column + colNeighbours[i], visited, emptyColumns)


def initializeVisited():
    return numpy.full((n, n), False)


def calculateSuccessors(matrix):
    visited = initializeVisited()
    successorList = []
    for i in range(n - 1, -1, -1):
        for j in range(n):
            if not visited[i][j] and not matrix[i][j] == -1:
                # nextMatrix = copy.deepcopy(matrix)
                nextMatrix = list(map(list, matrix))
                regionLength, nextMatrix, emptyColumns = getMaxRegionLength(nextMatrix, i, j, visited)
                successorObj = Successor(i, j, regionLength * regionLength, nextMatrix, emptyColumns)
                successorList.append(successorObj)
    return sorted(successorList, key=getKey, reverse=True)


def max_value(state, alpha, beta, depth):
    global expandedStates
    global return_first_state
    matrix = state.matrix
    v = float('-inf')

    depth = depth + 1
    if depth > depth_limit or terminalState(matrix):
        return state.score, -1, -1, -1, depth

    successors = calculateSuccessors(matrix)
    for successorObj in successors:
        expandedStates = expandedStates + 1
        successorObj.matrix = balanceState(successorObj.nextMatrix, successorObj.emptyColumns)
        successorObj.score = successorObj.score + state.score
        result, totalDepth = min_value(successorObj, alpha, beta, depth)

        if result > v:
            v = result
            row = successorObj.rowNum
            column = successorObj.colNum
            nextStepMatrix = successorObj.matrix
        if v >= beta:
            return v, row, column, nextStepMatrix, totalDepth
        alpha = max(alpha, v)
    return v, row, column, nextStepMatrix, totalDepth


def min_value(state, alpha, beta, depth):
    global expandedStates
    global return_first_state
    matrix = state.matrix
    v = float('inf')

    depth = depth + 1
    if depth > depth_limit or terminalState(matrix):
        return state.score, depth

    successors = calculateSuccessors(matrix)
    for successorObj in successors:
        expandedStates = expandedStates + 1
        successorObj.matrix = balanceState(successorObj.nextMatrix, successorObj.emptyColumns)
        successorObj.score = state.score - successorObj.score
        returnValue = max_value(successorObj, alpha, beta, depth)
        totalDepth = returnValue[4]
        result = returnValue[0]

        if result < v:
            v = result
        if v <= alpha:
            return v, totalDepth
        beta = min(beta, v)
    return v, totalDepth


def readCalibrationFile():
    with open('calibrate.txt') as file:
        statesPerSecond = int(file.readline().rstrip('\n'))
        return statesPerSecond


def convertMinusOneToStar(matrix):
    matrix = [['*' if x == -1 else str(x) for x in row] for row in matrix]
    return matrix


def convertStarToMinusOne(matrix):
    matrix = [[-1 if x == '*' else x for x in row] for row in matrix]
    return matrix


def writeOutputToFile(outputFileName, list):
    with open(outputFileName, 'w+') as outputfile:
        csv_writer = csv.writer(outputfile, delimiter=',')
        csv_writer.writerows([list])


rowNeighbours = [-1, 0, 0, 1]
colNeighbours = [0, -1, 1, 0]

inputMatrix = []
count = 1
expandedStates = 0

inputMatrix.append([[1, 2],
                    [4, 5]])

inputMatrix.append([[1, 2],
                    [4, 5]])

inputMatrix.append([[1, 2, 3, 0],
                    [4, 5, 6, 1],
                    [7, 8, 9, 4],
                    [0, 1, 2, 3]])

inputMatrix.append([[1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 2, 4],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3, 5],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 7],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 6, 8],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 7, 9],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 8, 0],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 3],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 1, 4],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 5],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 2, 4]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
                    ])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                    ])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]])

inputMatrix.append([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                    [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    ])

list_of_times = []
matrixSize = [2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
depth_limit_list = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]

for i in range(len(matrixSize)):
    step_start_time = time.time()
    n = matrixSize[i]
    depth_limit = depth_limit_list[i]
    stateObj = State(convertStarToMinusOne(inputMatrix[i]), 0)
    value, ansRow, ansColumn, nextStepMatrix, totalDepth = max_value(stateObj, float('-inf'), float('inf'), 0)
    nextStepMatrix = convertMinusOneToStar(nextStepMatrix)
    step_time_taken = time.time() - step_start_time
    time_per_node = step_time_taken / expandedStates
    list_of_times.append(time_per_node * 1000)
    expandedStates = 0

writeOutputToFile('calibrate.txt', list_of_times)
print(time.time() - start_time)
