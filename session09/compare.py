import matplotlib.pyplot as plt
import numpy as np

# Sample data
val = np.array([2.44, 2.48, 2.35])  

# Custom x labels
x_labels = ['Naive', 'MLP', 'RNN']

x = range(1,len(x_labels)+1)

bar_width = 0.60  # Width of the bars

r1 = np.arange(len(x))

plt.bar(r1, val, color='green', width=bar_width)
plt.xticks(r1, x_labels)

plt.ylabel('Mean Absolute Error')
plt.title('Jena dataset')
plt.grid(axis='y', alpha=0.7) 

plt.show()