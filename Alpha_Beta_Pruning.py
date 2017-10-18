import numpy
import time
import csv
import math

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
    emptyColumns = set()
    depthFirstSearch(matrix, row, column, visited, emptyColumns)
    return count, matrix, emptyColumns


def depthFirstSearch(matrix, row, column, visited, emptyColumns):
    global count
    visited[row][column] = True
    fruitType = matrix[row][column]
    matrix[row][column] = -1
    emptyColumns.add(column)

    for i in range(4):
        if isValidPosition(matrix, row + rowNeighbours[i], column + colNeighbours[i], visited, fruitType):
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
        if expandedStates == 1 and return_first_state:
            return successorObj.score, successorObj.rowNum, successorObj.colNum, successorObj.matrix, depth
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
    with open('calibration.txt') as file:
        calibration_values = [list(map(float, rec)) for rec in csv.reader(file, delimiter=',')]
        return calibration_values[0]


def calculateDepthLimit(branching_factor, states_to_be_expanded):
    if branching_factor == 1:
        depth = 1
    else:
        depth = math.log(states_to_be_expanded, branching_factor)
    return depth


def convertMinusOneToStar(matrix):
    matrix = [['*' if x == -1 else str(x) for x in row] for row in matrix]
    return matrix


def convertStarToMinusOne(matrix):
    matrix = [[-1 if x == '*' else int(x) for x in row] for row in matrix]
    return matrix


def writeOutputToFile(outputFileName, move, nextStepMatrix):
    with open(outputFileName, 'w+') as outputfile:
        outputfile.write(str(move) + '\n')
        for row in nextStepMatrix:
            for element in row:
                outputfile.write(str(element))
            outputfile.write('\n')


rowNeighbours = [-1, 0, 0, 1]
colNeighbours = [0, -1, 1, 0]
count = 1
expandedStates = 0

n, typesOfFruits, availableTime, inputMatrix = readInputFromFile('input.txt')
return_first_state = False
inputMatrix = convertStarToMinusOne(inputMatrix)

calibration_list = readCalibrationFile()
index = int(n / 2)
states_possible = (availableTime * 1000) / calibration_list[index]

number_of_successors = len(calculateSuccessors(inputMatrix))

print('Number of Successors : ', number_of_successors)

if number_of_successors > 0:
    states_to_be_expanded = 2 * states_possible / number_of_successors

    print('States to be expanded : ', states_to_be_expanded)
    value = math.ceil(min(number_of_successors, calculateDepthLimit(number_of_successors, states_to_be_expanded)))

    print('Value after rounding : ', value)
    depth_limit = value

    if availableTime > 250:
        if n > 23:
            depth_limit = max(2.0, depth_limit)
        elif n < 15:
            depth_limit = max(4.0, depth_limit)
        else:
            depth_limit = max(3.0, depth_limit)
    if availableTime < 10:
        depth_limit = 1

    print('Depth Limit : ', depth_limit)

    if depth_limit == 0 or depth_limit == 1:
        depth_limit = 1
        return_first_state = True

    stateObj = State(inputMatrix, 0)

    value, ansRow, ansColumn, nextStepMatrix, totalDepth = max_value(stateObj, float('-inf'), float('inf'), 0)
    nextStepMatrix = convertMinusOneToStar(nextStepMatrix)
    writeOutputToFile('output.txt', chr(ord('A') + ansColumn) + str(ansRow + 1), nextStepMatrix)
    print('Value ', value)
    print('Expanded States : ', expandedStates)
    print('Max Depth : ', totalDepth)
    availableTime = availableTime - time.time() + start_time
    print('Time Left : ', availableTime)
