import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.random.randn(16, 4) * 2
y = np.random.randn(16, 4) * 3
print(x)
print(x.shape)
data = np.empty((x.shape[0], x.shape[1] + y.shape[1]))
data[:, ::2] = x
data[:, 1::2] = y
bp = plt.boxplot(data, labels=["roll", "roll_r", "flat", "flat_r", "band", "band_r", "centro", "centro_r"])
# Now fill the boxes with desired colors
box_colors = ['pink', 'royalblue']
num_boxes = (data.shape[1])
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    median_x = []
    median_y = []
    for j in range(2):
        median_x.append(med.get_xdata()[j])
        median_y.append(med.get_ydata()[j])
        ax1.plot(median_x, median_y, 'k')
    medians[i] = median_y[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')
