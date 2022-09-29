import math
import random
import numpy as np


def compGsubDelta(w, x, y, delta):
    if y >= np.dot(np.transpose(w), x) + delta:
        return (y - np.dot(np.transpose(w), x) - delta) ** 2
    elif abs(y - np.dot(np.transpose(w), x)) < delta:
        return 0
    elif y <= np.dot(np.transpose(w), x) - delta:
        (y - np.dot(np.transpose(w), x) + delta) ** 2
    return -1

def computeFunction(w, x, y, lam):
    runningSum = 0
    for x_i, y_i in zip(x, y):
        runningSum += compGsubDelta(w, x_i, y_i, delta)
    return (((runningSum)/len(x)) + (lam*(sum(w ** 2))))

def computeFunctionGradient(w, x, y, lam, cond):
    pass



def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []
    iteration = 0
    # w = np.array([(0), (0)])
    newW = w
    while iteration < num_iter:
        history_fw.append(computeFunction())
        for x_i, y_i in zip(data, y):

            newW = newW + eta * 
        
        iteration += 1

   
    
    # return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    # return new_w, history_fw
    pass

bgd_l2(np.load("data.npy"), 0, 0, 0, 0, 0, 0)

# A = np.array([(1), (4)])
    # B = np.array([(1), (2)])
    # print(np.dot(A,B))