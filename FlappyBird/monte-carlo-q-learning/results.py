"""
    File name: results.py
    Author: Nikola Zubic
"""
import matplotlib.pyplot as plt
import ast
"""
Successfully tested basic Q agent.

Here will go plot results at the end.
"""

f = open("results_2000.txt", "r")
results_list = ast.literal_eval(f.read())

x = []
y = []

for first, second in results_list:
    x.append(first)
    y.append(second)


plt.plot(x, y)
#plt.scatter([1,2], [3,4], color='red', marker='^')
plt.xlabel('episode')
plt.ylabel('training score')
plt.show()
