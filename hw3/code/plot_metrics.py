import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


my_data = pd.read_csv('metrics.csv',sep=',', header=None)

loss = [0]*30
accuracy = [0]*30
epoch = [0]*30

for i in range(30):
    epoch[i] = i
    loss[i] = np.mean(my_data[34*i+1:34*(i+1)-1][0])
    accuracy[i] = np.mean(my_data[34*i+1:34*(i+1)-1][1])



plt.plot(epoch,loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('ResNet Validation loss')
plt.show()





plt.plot(epoch,accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('ResNet Validation accuracy')
plt.show()


#my_data = genfromtxt('shared_metrics.csv', delimiter=',')
my_data = pd.read_csv('shared_metrics.csv', sep=',', header=None)

loss = [0]*30
accuracy = [0]*30
epoch = [0]*30

for i in range(30):
    epoch[i] = i
    loss[i] = np.mean(my_data[34*i+1:34*(i+1)-1][0])
    accuracy[i] = np.mean(my_data[34*i+1:34*(i+1)-1][1])


plt.plot(epoch,loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Orthogonal ResNet Validation loss')
plt.show()




plt.plot(epoch,accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Orthogonal ResNet Validation accuracy')
plt.show()