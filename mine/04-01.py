
# coding: utf-8

#mini batch
import sys, os
sys.path.append(os.pardir)
#import pickle
import numpy as np
from dataset.mnistPy2 import load_mnist

(x_train, t_train), (x_test, t_test) = \
   load_mnist(normalize=True, one_hot_label=True)

print x_train.shape
print t_train.shape

train_size = x_train.shape[0]  #(60000, 784) 중 첫번째 element
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #60000미만 중에서 무작위로 10개
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print x_batch
print t_batch
