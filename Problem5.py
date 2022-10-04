import math
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compGsubDelta(w, x, y, delta):
    if y >= np.dot(np.transpose(w), x) + delta:
        return (y - np.dot(np.transpose(w), x) - delta) ** 2
    elif abs(y - np.dot(np.transpose(w), x)) < delta:
        return 0
    elif y <= np.dot(np.transpose(w), x) - delta:
        (y - np.dot(np.transpose(w), x) + delta) ** 2
    return -1

def computeFunction(w, x, y, lam, delta):
    runningSum = 0
    for x_i, y_i in zip(x, y):
        runningSum += compGsubDelta(w, x_i, y_i, delta)
    return (((runningSum)/len(x)) + (lam*(sum(w ** 2))))

def computeFunctionGradient(w, x, y, lam, delta, cond):
    runningSum = 0
    n = len(x)
    if cond == 1:
        for x_i, y_i in zip(x, y):
            runningSum += ((y_i - np.dot(np.transpose(w), x_i) - delta) * x_i)
        return ((runningSum * (n/2)) + (2*lam*sum(w)))
    elif cond == 2:
        return (2*lam*sum(w))
    elif cond == 3:
        for x_i, y_i in zip(x, y):
            runningSum += ((y_i - np.dot(np.transpose(w), x_i) + delta) * x_i)
        return ((runningSum * (n/2)) + (2*lam*sum(w)))
    return -1

def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []
    new_w = w
    iteration = 1

    while iteration < num_iter + 1:

        # history_fw.append(computeFunction(new_w, data, y, lam, delta))
        for x_i, y_i in zip(data, y):
            gradient = 0

            if y_i >= np.dot(np.transpose(new_w), x_i) + delta:
                #pass in all data and y?
                gradient = computeFunctionGradient(new_w, data, y, lam, delta, 1)

            elif abs(y_i - np.dot(np.transpose(new_w), x_i)) < delta:
                gradient = computeFunctionGradient(new_w, data, y, lam, delta, 2)

            elif y_i <= np.dot(np.transpose(new_w), x_i) - delta:
                gradient = computeFunctionGradient(new_w, data, y, lam, delta, 3)
                
            new_w = new_w - (eta * (gradient))
            
        
        iteration += 1


    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):

        return new_w, history_fw



# A = np.array([(1), (4)])
    # B = np.array([(1), (2)])
    # print(np.dot(A,B))