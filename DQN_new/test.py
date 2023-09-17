import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
loss = tf.squared_difference(a,b)
print('loss',loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # z = tf.squared_difference(x, y)
    print(sess.run(loss))
