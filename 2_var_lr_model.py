import numpy as np
import tensorflow as tf

# sample
data_x = np.random.rand(100)
data_y = data_x*3 + 0.3


# Linear Model's
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))

# model expression
y_head = W*data_x + b

# loss
loss = tf.reduce_mean(tf.square(y_head-data_y))

# algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
opt_process = optimizer.minimize(loss)


# train:
sess = tf.Session()

# init var
init = tf.global_variables_initializer()
sess.run(init)

# sgd:
for i in range(501):
    sess.run(opt_process)
    if i % 20 == 0:
        print(i,sess.run(W),sess.run(b))
