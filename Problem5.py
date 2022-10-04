import math
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def computeFunction(w, x, y, lam, delta):
    # Used to compute the value of the objective function after each update to w
    n = len(x)
    objFunc = 0
    for x_i, y_i in zip(x, y):
        
        if y_i >= np.dot(np.transpose(w), x_i) + delta:
            objFunc += (((y_i-np.dot(np.transpose(w), x_i) - delta) ** 2)/n)
        # middle case just adds 0 so can be ignored
        elif y_i <= np.dot(np.transpose(w), x_i) - delta:
            objFunc += (((y_i-np.dot(np.transpose(w), x_i) + delta) ** 2)/n)
    
    return objFunc + (lam * ((w[0] ** 2) + (w[1] ** 2)))


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []
    iteration = 1
    n = len(data)
    while iteration < num_iter + 1:
        #pretty sure the best way to do this is with a variable number of w and you want to represent it as a vector but for the sake of getting this turned in i'm leaving it like this
        w0_grad = 0
        w1_grad = 0

        history_fw.append(computeFunction(w, data, y, lam, delta))
        for x_i, y_i in zip(data, y):
            
            if y_i >= np.dot(np.transpose(w), x_i) + delta:
                w0_grad += -(2/n) * x_i[0] * (y_i - np.dot(np.transpose(w), x_i) - delta)
                w1_grad += -(2/n) * (y_i - np.dot(np.transpose(w), x_i) - delta)
            # middle case just adds 0 so can be ignored
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

    history_fw = []
    iteration = 1
    n = len(data)
    while iteration < num_iter + 1:
        w0_grad = 0
        w1_grad = 0

        history_fw.append(computeFunction(w, data, y, lam, delta))
        if i == -1:
            randPoint = random.randint(0, n)
            x_i = data[randPoint]
            y_i = y[randPoint]
        else:
            x_i = data[i]
            y_i = y[i]
            
        if y_i >= np.dot(np.transpose(w), x_i) + delta:
            w0_grad += -(2/n) * x_i[0] * (y_i - np.dot(np.transpose(w), x_i) - delta)
            w1_grad += -(2/n) * (y_i - np.dot(np.transpose(w), x_i) - delta)
        # middle case just adds 0 so can be ignored
        elif y_i <= np.dot(np.transpose(w), x_i) - delta:
            w0_grad += -(2/n) * x_i[0] * (y_i - np.dot(np.transpose(w), x_i) - delta)
            w1_grad += -(2/n) * (y_i - np.dot(np.transpose(w), x_i) - delta)
                

        w0_grad += 2 * lam * w[0]
        w1_grad += 2 * lam * w[1]
        w[0] -= ((1/math.sqrt(iteration)) * w0_grad)
        w[1] -= ((1/math.sqrt(iteration)) * w1_grad)
        if i != -1:
            break
        iteration += 1

    new_w = [w[0], w[1]]
    return new_w, history_fw
