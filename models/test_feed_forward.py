import tensorflow as tf
import numpy as np
import models.feed_forward as ff
from tensorflow.examples.tutorials.mnist import input_data
'''
  Using the MNIST data as test data to test the feed forward neural networks
'''
# read MNIST data
mnist = input_data.read_data_sets("../ex_mnist/MNIST_data", one_hot = True)

# generate the feed forward neural network with softmax (classification task)
sess = tf.Session()
with tf.variable_scope("dnn_softmax") as scope:
  dnn = ff.FeedForward(784, 10, 2, [512], tf.nn.relu, output_layer = 'softmax')
  dnn.new_session(sess)
with tf.variable_scope("dnn_linear") as scope:
  dnn_reg = ff.FeedForward(784, 784, 2, [512], tf.nn.relu, output_layer = 'linear')
  dnn_reg.new_session(sess)

sess.run(tf.global_variables_initializer())

# test updata the learning rate
print dnn.assign_lr(0.001)
print "Test dnn.assign_lr successfully!"

#test dnn.partial_fit(X, Y)
for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #train_step.run({x:batch_xs, y_: batch_ys})
  cost = dnn.partial_fit(batch_xs, batch_ys,True)
  mse = dnn_reg.partial_fit(batch_xs, batch_xs, True)

#test dnn eval_accuracy
print "Classification accuracy: ",dnn.eval_accuracy(mnist.test.images, mnist.test.labels)
print "Regression MSE: ",dnn_reg.get_mse(mnist.test.images, mnist.test.images)
print "Test softmax/linear dnn.partial_fit and dnn.eval_accuracy successfully!"

# test dnn.get_output
output = dnn.get_output(batch_xs)
print output[0, :]
print "test dnn.get_output successfully!"

#test save and restore
save_path = './tmp/mnist.ff.model'
w_trained = sess.run(dnn.weights['w_0'])
saver = tf.train.Saver()
saver.save(dnn.sess, save_path)
with tf.Session() as sess:
  dnn.new_session(sess)
  sess.run(tf.global_variables_initializer())
  w_init_again = sess.run(dnn.weights['w_0'])
  # restore the trained model
  saver.restore(dnn.sess, save_path)
  w_restore = sess.run(dnn.weights['w_0'])

  diff = w_trained-w_restore
  if diff[np.nonzero(diff)].shape[0] == 0:
    print "Test save and restore sucess!\n"


