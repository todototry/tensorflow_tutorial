import tensorflow as tf

# define a param
x = tf.Variable(0, name='counter')

# define a const
cons = tf.constant(1)

new_var = tf.add(x, cons) # new_var = tf.add(x, 1)
update = tf.assign(x, new_var)

# init
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# tensor run
sess = tf.Session()

# 1. exec init()
sess.run(init)

# 2. calc
print(sess.run(update))
print(sess.run(update))
print(sess.run(update))