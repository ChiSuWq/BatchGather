"""
filename: some_test.py

(include the following functions like top-k, np.where, tf.gather and so on)

@author:Su_Chi
@date: 2019/5/23 22:25
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.array_ops import batch_gather
a = [[1,2,3,4],[5,3,4,2]]

a = np.array(a)

index = [[0,0],[1,1]]
print(a[0])

#NOTE: here is the top-k function and the use of it to build the top-k ndarray 
#										for some research like hard negative mining

k = 2#top-k elements  k= the constraint to the number
a_top_k = tf.nn.top_k(a, k)
indices = a_top_k[1]

a_reshape = a.reshape(-1, 4)
indices_reshape = tf.reshape(indices,(-1, 2))

gather_top_k = []
for i in range(len(a)):
	gather_top_k.append(tf.gather(a[i], indices_reshape[i]))

gather_top_k = tf.stack(gather_top_k, axis=0)

gather_top_k = tf.reshape(gather_top_k, (2, 2))
#gather_top_k = tf.expand_dims(gather_top_k, -1)

batch_gather_top_k,batch_indices = batch_gather(a, indices)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	a_top_k_arr = sess.run(a_top_k)
	print(a_top_k_arr)
	gather_top_k = sess.run(gather_top_k)
	print('gather_top_k: ',gather_top_k,sep='\n')
	print('a_top_k_arr[0]:',a_top_k_arr[0],sep='\n')
	print('batch_gather: ',sess.run(batch_gather_top_k),sep='\n')
	print('batch_indices: ',sess.run(batch_indices),sep='\n')


#tensor = [[1,2],[3,4],[5,6]]
mask = np.array([[True,False],[True,False],[True,False]])
"""
mask_reshape = np.reshape(mask,[-1])
mask_reshape = np.expand_dims(mask_reshape,1)
print(mask_reshape.shape)
print(np.squeeze(mask_reshape, axis=(1)).shape)
"""

#t_mask = tf.boolean_mask(tensor, mask)





#NOTE: here are the "tf.where" and "tf.squeeze" functions

mask_origin = tf.convert_to_tensor(mask, name='mask')
mask = tf.reshape(mask_origin,[-1])
mask_where = tf.where(mask)
indices = tf.squeeze(mask_where, [1])
with tf.Session() as sess:
	print(sess.run(mask))
	print(sess.run(tf.shape(mask)))
	print(type(mask))
	print(sess.run(mask_where))
	print(sess.run(indices))# 1-D vector
	print(mask_origin.get_shape().num_elements())




