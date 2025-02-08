import numpy as np

import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 400) #this command arrange the spacing of graph line.

y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3


plt.plot(x, y1, linestyle='-', color='Red', label='y = 2x + 1')  # Solid Red
plt.plot(x, y2, linestyle='--', color='Green', label='y = 2x + 2')  # Dashed Green
plt.plot(x, y3, linestyle='-.', color='Blue', label='y = 2x + 3')  # Dash-dot Blue

plt.xlabel('x-axis')   #Show X-axis in side space
plt.ylabel('y-axis')   #Show Y-axis in side space
plt.title('Graph of y = 2x + c for c = 1, 2, 3')  #Shows the title of the graph

plt.grid(True) # this command show the grid in graph
plt.legend()   #This command shows the legend (Show chart data)

# Show the plot
plt.show()
