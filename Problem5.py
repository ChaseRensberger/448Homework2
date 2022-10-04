import math
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def computeFunction(w, x, y, lam, delta):
    n = len(x)
    f = 0
    for x_i, y_i in zip(x, y):
        
        if y_i >= np.dot(np.transpose(w), x_i) + delta:
            f += (((y_i-np.dot(np.transpose(w), x_i) - delta) ** 2)/n)
        elif y_i <= np.dot(np.transpose(w), x_i) - delta:
            f += (((y_i-np.dot(np.transpose(w), x_i) + delta) ** 2)/n)
    
    return f + (lam * ((w[0] ** 2) + (w[1] ** 2)))


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []
    iteration = 1
    n = len(data)
    while iteration < num_iter + 1:
        w0_grad = 0
        w1_grad = 0

        history_fw.append(computeFunction(w, data, y, lam, delta))
        for x_i, y_i in zip(data, y):
            
            if y_i >= np.dot(np.transpose(w), x_i) + delta:
                w0_grad += -(2/n) * x_i[0] * (y_i - np.dot(np.transpose(w), x_i) - delta)
                w1_grad += -(2/n) * (y_i - np.dot(np.transpose(w), x_i) - delta)

            elif y_i <= np.dot(np.transpose(w), x_i) - delta:
                w0_grad += -(2/n) * x_i[0] * (y_i - np.dot(np.transpose(w), x_i) - delta)
                w1_grad += -(2/n) * (y_i - np.dot(np.transpose(w), x_i) - delta)
                

        w0_grad += 2 * lam * w[0]
        w1_grad += 2 * lam * w[1]
        w[0] -= (eta * w0_grad)
        w[1] -= (eta * w1_grad)
            
        iteration += 1

    new_w = [w[0], w[1]]
    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):

        return new_w, history_fw



# A = np.array([(1), (4)])
    # B = np.array([(1), (2)])
    # print(np.dot(A,B))