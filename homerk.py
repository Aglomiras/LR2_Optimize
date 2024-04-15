# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
np.random.seed(10)
data = [1,2,2,2,3,4,1,2,3,2,2,3,4,5,5,3,6,7,8,96,0,5,4,3]

fig = plt.figure(figsize=(10, 10))

# Creating plot
plt.boxplot(data)

# show plot
plt.show()