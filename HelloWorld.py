import tensorflow as tf

input_data = tf.Variable([0], dtype=tf.float32)
output = tf.nn.sigmoid(input_data)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
