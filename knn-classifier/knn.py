import numpy as np
import operator
import csv

def classify(x, dataSet, labels, k):
    dataSetSize = len(dataSet)
    difference = np.tile(x, (dataSetSize, 1)) - dataSet
    square =  difference ** 2
    sum = square.sum(axis = 1)
    distance = sum ** 0.5
    sortedIndexes = distance.argsort()

    votes = {}
    for i in range(k):
        vote = labels[sortedIndexes[i]]
        if vote in votes:
            votes[vote] = votes[vote] + 1
        else:
            votes[vote] = 1
    sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def loadCSV(path):
    columns = None
    lines = 0
    rows = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if lines == 0:
                columns = row
            else:
                rows.append(row)
            lines += 1
    result = np.array(rows)
    return columns, result

def normalize(input):
    minValue = input.min(0)
    maxValue = input.max(0)
    range = maxValue - minValue
    result = np.zeros(input.shape)
    m = input.shape[0]
    result = input - np.tile(minValue, (m, 1))
    result = result / np.tile(range, (m, 1))
    return result

def main():
    columns, rows = loadCSV('iris.csv')
    # The iris data set is sorted in terms of the labels.
    # We need to split the dataset into training set and test set.
    # Therefore, shuffle the rows before splitting the dataset.
    np.random.shuffle(rows)
    columnCount = len(columns)
    dataSet = normalize(rows[:, 0 : columnCount - 1].astype(float))
    labels = rows[:, columnCount - 1 :].flatten()

    limit = int(len(dataSet) * 0.90)
    trainingSet = dataSet[0 : limit, :]
    testSet = dataSet[limit :, :]

    k = 20
    testCaseCount = len(testSet)
    errorCount = 0
    for i in range(0, testCaseCount):
        testCase = testSet[i]
        result = classify(testCase[0 : columnCount - 1], trainingSet, labels, k)
        if result != labels[limit + i]:
            print(f'expected = {labels[limit + i]}, classified = {result} [wrong]')
            errorCount += 1
        else:
            print(f'expected = {labels[limit + i]}, classified = {result}')
    errorRate = errorCount / testCaseCount
    print(f'test case count = {testCaseCount}, errors = {errorCount}, error rate = {errorRate}')

if __name__ == '__main__':
    main()