# BatchGather
another idea for the built-in Tensorflow function "tf.batch_gather".

"tf.batch_gather" is a built-in function matching "tf.nn.top_k" in my own opinion.

* In other words, my opinion maybe a mistake.

the origin "tf.batch_gather" convert the local batch_indices into global ones.

![](https://github.com/ChiSuWq/BatchGather/blob/master/Image/indices_from_local_to_global.jpg)

