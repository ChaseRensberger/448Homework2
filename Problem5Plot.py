import random
import numpy as np
import matplotlib.pyplot as plt

from Problem5 import bgd_l2

if __name__ == '__main__':
    data = np.load("data.npy")
    data = np.insert(data, 1, np.ones(len(data)), axis=1) # Add column of ones
    xValues = data[:, :2]
    yValues = data[:, 2]
    newW, historyFW = bgd_l2(xValues, yValues, np.array([(0), (0)]), 0.05, 0.1, 0.001, 50)
    print(historyFW)


    # plt.scatter(xValues, yValues)
    # plt.show()








# year = [2014, 2015, 2016, 2017, 2018, 2019]  
# tutorial_count = [39, 117, 111, 110, 67, 29]    

# plt.plot(year, tutorial_count, color="#6c3376", linewidth=3)  
# plt.xlabel('Year')  
# plt.ylabel('Number of futurestud.io Tutorials') 
# plt.savefig('line_plot.pdf') 
