import numpy as np
import tensorflow as tf
# import tensorflow as tf2
# tf = tf2.compat.v1
# tf.disable_v2_behavior()

# sess = tf.InteractiveSession()
# with tf.variable_scope('scope'):
#     with tf.variable_scope('scope2'):
#         v1 = tf.Variable(1,name = 'xx')
#         v2 = tf.Variable('2',name='yy')
# v3 = tf.Variable(1,name = 'xx')
# print(v3.np())
# # v4 = v3*2
# sess.run(v1.initializer)
# sess.run(v1)
# # print(v1.name)
# # print(v2.name)
# print(tf.version)
# v1 = tf.Variable([[123,2],[11,22]])
# print(v1.numpy())
# v2 = tf.Variable(v1.numpy()*2)
# print(v2)
# print(v2.numpy())
mulArrays = [[1,2,3],[4,5,6],[7,8,9]]
print(list(np.array(mulArrays).flatten()))
print(np.array(mulArrays).flatten())


c = np.arange(1,4)
c = c.reshape(1,4)
cc=1