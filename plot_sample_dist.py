import numpy as np
import matplotlib.pyplot as plt

with open('./classdist_int.txt', 'r') as f:
    dists = f.readlines()


idx = 1

plt.figure(figsize=(18,3))
plt.subplots_adjust(wspace=0.8)
for dist in dists:
    d = dist.replace(' ', '')
    d = d[1:-3].split('.')
    d = list(map(int,d))
    plt.subplot(1,len(dists),idx)
    plt.bar(list(range(10)), d)
    idx += 1
    plt.yticks(np.arange(0,201,50))
    plt.xticks(list(range(10)))

plt.savefig('class_int.png')
