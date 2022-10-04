import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import bgd_l2, sgd_l2

def plotGD(data, y, w, eta, delta, lam, num_iter):
    new_w, history_fw = bgd_l2(data, y, w, eta, delta, lam, num_iter)
    plt.plot(history_fw)
    plt.xlabel('Iteration')
    plt.ylabel('f(w)')
    plt.title("BGD(η = 0.1,δ = 0.01,λ = 0.001,num iter = 50)")
    plt.show()

def plotSGD(data, y, w, eta, delta, lam, num_iter):
    new_w, history_fw = sgd_l2(data, y, w, eta, delta, lam, num_iter)
    plt.plot(history_fw)
    plt.xlabel('Iteration')
    plt.ylabel('f(w)')
    plt.show()

if __name__ == '__main__':
    data = np.load("data.npy")
    data = np.insert(data, 1, np.ones(len(data)), axis=1) # Add column of ones
    xValues = data[:, :2]
    yValues = data[:, 2]
    plotGD(xValues, yValues, np.array([(0), (0)], dtype=float), 0.05, 0.1, 0.001, 50)
    # plotGD(xValues, yValues, np.array([(0), (0)], dtype=float), 0.1, 0.01, 0.001, 50)
    # plotGD(xValues, yValues, np.array([(0), (0)], dtype=float), 0.1, 0, 0.001, 100)
    # plotGD(xValues, yValues, np.array([(0), (0)], dtype=float), 0.1, 0, 0, 100)

    # plotSGD(xValues, yValues, np.array([(0), (0)], dtype=float), 1, 0.1, 0.5, 800)
    # plotSGD(xValues, yValues, np.array([(0), (0)], dtype=float), 1, 0.01, 0.1, 800)
    # plotSGD(xValues, yValues, np.array([(0), (0)], dtype=float), 1, 0, 0, 40)
    # plotSGD(xValues, yValues, np.array([(0), (0)], dtype=float), 1, 0, 0, 800)

    

    








# year = [2014, 2015, 2016, 2017, 2018, 2019]  
# tutorial_count = [39, 117, 111, 110, 67, 29]    

# plt.plot(year, tutorial_count, color="#6c3376", linewidth=3)  
# plt.xlabel('Year')  
# plt.ylabel('Number of futurestud.io Tutorials') 
# plt.savefig('line_plot.pdf') 
