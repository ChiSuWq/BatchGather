# BatchGather
##Another idea for the built-in Tensorflow function "tf.batch_gather".

"tf.batch_gather" is a built-in function matching "tf.nn.top_k" in my own opinion.

* In other words, my opinion is maybe a mistake.

The origin "tf.batch_gather" convert the local batch_indices into global ones.

* Here is the array example
![](https://github.com/ChiSuWq/BatchGather/blob/master/Image/example_array.jpg)

* The example that the local indices have be converted to global ones.
* (The TopKV2 results from "tf.nn.top_k")

So we can find that 0 to 4 and 2 to 6.
![](https://github.com/ChiSuWq/BatchGather/blob/master/Image/indices_from_local_to_global.jpg)

then it gathers the values corresponding to the indices.

* my batch_gather function

I do not change the batch_indices. In contrast, the data array is processed, and what I do is just a demo to show the function's principle.

please read my code`some_test.py` to find more, and I also provide the `batch_gather.py` which is from tensorflow-github.

Wanna know more about `batch_gather.py`, you can refer https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/array_ops.py







