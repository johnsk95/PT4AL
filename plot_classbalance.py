import numpy as np
import matplotlib.pyplot as plt

class_dist = np.zeros(10)
class_dist1 = np.zeros(10)


with open('./loss/batch_0.txt', 'r') as f:
    samples = f.readlines()

with open('./loss/batch_1.txt', 'r') as f:
    samples1 = f.readlines()

for sample in samples:
    label = int(sample.split('/')[-2])
    class_dist[label] += 1

for sample in samples1:
    label = int(sample.split('/')[-2])
    class_dist1[label] += 1

plt.bar(list(range(10)), class_dist)
plt.bar(list(range(10)), class_dist1)


plt.xlabel('class labels')
plt.title('Sample Class Distribution')

plt.xticks(list(range(10)), list(range(10)))
plt.savefig('class_dist.png')