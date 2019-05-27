# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:15:41 2019

@author: Charl
"""

#XOR problem is non-linear: patterns seperated by curves/circles

#math library used for neural networks
import tensorflow as tf
#used for arrays and matrices + functions for arrays
import numpy as np
#extension of numpy - used for plots
import matplotlib.pyplot as plt

#forward function
def forward(x, w1, b1, w2, b2, train=True): 
    Z = tf.nn.sigmoid(tf.matmul(x, w1) + b1) #matmul - multiply 2 matrices | tf.nn - wrapper for NN operations
    Z2 = tf.matmul(Z, w2) + b2
    if train:
        return Z2
    return tf.nn.sigmoid(Z2) #sigmoid - activation function [ f(x) = 1 / (1 + e^{-x}) ] - outputs range from (0, 1)
#non-linear activation output

#activation function of a node defines the output of that node, or "neuron," given an input or set of inputs. 
#This output is then used as input for the next node and so on until a desired solution to the original problem is found.


#the init_weights function builds new variables in the given shape and initialises the network's weights with random values
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) outputs random values from 
#a normal distribution.
    

#tf.Variable - Exists outside the context of a single session.run call
#tf.Tensor - Exists within the context of session.run call
    

X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([[1], [1], [0], [0]])

#define placeholders for input X and output y
#tf.placeholder - variable that is assigned at a later time (can build operations and graph without data)
phX = tf.placeholder(tf.float32, [None, 2]) #(type, dimensionality)
phY = tf.placeholder(tf.float32, [None, 1])

#init weights - random
#layer 1 - 5 nodes
w1 = init_weights([2, 5])
b1 = init_weights([5])
#layer 2
w2 = init_weights([5, 1])
b2 = init_weights([1])

y_hat = forward(phX, w1, b1, w2, b2)
pred = forward(phX, w1, b1, w2, b2, False)

#init learning rate
lr = 0.01
#iterations
epochs = 1000

#init cost function
cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=phY)) 
#https://stackoverflow.com/questions/46291253/tensorflow-sigmoid-and-cross-entropy-vs-sigmoid-cross-entropy-with-logits

#init train function with adam optimizer
train = tf.train.AdamOptimizer(lr).minimize(cost)

#save costs for plotting
costs = []

#create session and init variables
init = tf.global_variables_initializer() #init - constructor
sess = tf.Session()
sess.run(init)

#start training
for i in range(epochs):
    sess.run(train, feed_dict={phX: X, phY: y})
    
    #feed_dict - give values at runtime to tf.placeholder declared earlier
    
    c = sess.run(cost, feed_dict={phX: X, phY: y})
    #sess.run - run optimiser (cost) and use feed_dict to pass values to placeholders
    costs.append(c)
    
    if i % 100 == 0:
        print(f"iteration {i}. Cost: {c}.")
        
print("Training complete.")

#make prediction
prediction = sess.run(pred, feed_dict={phX: X})
print("Percentages: ")
print(prediction)
print("Prediction: ")
print(np.round(prediction))

#plot cost
plt.plot(costs)
plt.show()

#description of a network depends on its architecture and parameters (weights and biases)
#parameters are estimated - defined as a variable of the model
#architecture is determined by the configuration of symbolic operations


#1. Bias - prejudice due to erroneous assumptions. Biases are built into the algorithms because they are
#          created by individuals who have concious/unconcious preferences























