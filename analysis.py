import numpy as np
import random 

with open('./rotation_loss.txt', 'r') as f:
    losses = f.readlines()

loss_1 = []
name_2 = []

for j in losses:
    loss_1.append(j[:-1].split('_')[0])
    name_2.append(j[:-1].split('_')[1])

s = np.array(loss_1)
sort_index = np.argsort(s)

for i in range(10):
    # sample minibatch from unlabeled pool 
    sample5000 = sort_index[i*5000:(i+1)*5000]
    # sample1000 = sample5000[[j*5 for j in range(1000)]]
    b = np.zeros(10)
    for jj in sample5000:
        b[int(name_2[jj].split('/')[-2])] +=1
    print(f'{i} Class Distribution: {b}')
    s = './loss/batch_' + str(i) + '.txt'
    for k in sample5000:
        with open(s, 'a') as f:
            f.write(name_2[k]+'\n')
    
