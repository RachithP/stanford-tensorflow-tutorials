""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

def huber(label, prediction, delta=14.0):
    residual = tf.abs(label-prediction)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5*tf.square(delta)
    return tf.cond(residual<delta, f1, f2)


DATA_FILE = 'data/birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
# X, Y = None, None
#############################
########## TO DO ############
X = tf.placeholder(tf.float32, shape=None, name="Birth_Rate")
Y = tf.placeholder(tf.float32, shape=None, name="Life_expectancy")

#############################

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
# w, b = None, None
#############################
########## TO DO ############
w = tf.get_variable(name="weight", shape=None, dtype=tf.float32, initializer=tf.constant(0.0))
b = tf.get_variable(name="bias", shape=None, dtype=tf.float32, initializer=tf.constant(0.0))
tf.summary.scalar('weights', w)
tf.summary.scalar('biases', b)
#############################

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
#############################
########## TO DO ############
Y_predicted = w * X + b
#############################

# Step 5: use the square error as the loss function
#############################
########## TO DO ############
# loss = tf.square(Y_predicted-Y)
# mse_loss = tf.losses.mean_squared_error(Y, Y_predicted)
huber_loss = huber(Y, Y_predicted)
tf.summary.scalar('loss', huber_loss)
#############################

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.001).minimize(huber_loss)

start = time.time()

# Create a filewriter to write the model's graph to TensorBoard
#############################
########## TO DO ############
writer = tf.summary.FileWriter('./graphs/linreg', tf.get_default_graph())
merged = tf.summary.merge_all()
#############################


with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    #############################
    ########## TO DO ############
    sess.run(w.initializer)
    sess.run(b.initializer)
    #############################

    # Step 8: train the model for 100 epochs
    for i in range(200):
        total_loss = 0
        for x, y in data:
            # Execute train_op and get the value of loss.
            # Don't forget to feed in data for placeholders
            _, loss_, summary_tr = sess.run([optimizer, huber_loss, merged],
                                feed_dict={X: x, Y: y})
            writer.add_summary(summary_tr, i)
            total_loss += loss_

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
        if i==99:
            w_out = w.eval()
            b_out = b.eval()
    # close the writer when you're done using it
    #############################
    ########## TO DO ############
    #############################
    writer.close()
    
    # Step 9: output the values of w and b
    # w_out = w.eval(sess)
    # b_out = b.eval(sess)
    #############################
    ########## TO DO ############
    print(w_out, b_out)
    #############################

print('Took: %f seconds' %(time.time() - start))

# uncomment the following lines to see the plot 
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
