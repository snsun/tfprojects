import tensorflow as tf
import time
class FeedForward(object):
  
  def __init__(self, n_input, n_output, n_hidden, n_units, active_funcs,output_layer = 'soft_max', keep_prob=1):
    '''
    n_input: the dim of input
    n_hidden: the numbers of hidden layers, int
    n_units: lists, the neural units of every layer. If len(n_nuits) == 1, every layer uses the same numbers of units
    active_funs: tf.nn.relu...
    '''
    tf.set_random_seed(int(time.time()))
    self.n_input = n_input
    self.n_output = n_output
    self.n_hidden = n_hidden
    self.n_units = n_units;
    self.active_funcs = active_funcs
    self.output_layer = output_layer
    
    if len(self.n_units) == 1:
      for i in range(1, self.n_hidden):
        self.n_units.append(self.n_units[0])
    
    network_weights = self._initialize_weights()
    self.weights = network_weights;


    self.x = tf.placeholder(tf.float32, [None, self.n_input])
    self.labels = tf.placeholder(tf.float32, [None, self.n_output])
    for i in range(0, n_hidden +1):
      w_name = 'w_' + str(i);
      b_name = 'b_' + str(i);
      if i == 0:
        self.h = self.active_funcs(tf.matmul(self.x, self.weights[w_name]) + self.weights[b_name])  
        if keep_prob != 1:
          self.h = tf.nn.dropout(self.h, keep_prob)

      elif i > 0 and i < n_hidden:
        self.h = self.active_funcs(tf.matmul(self.h, self.weights[w_name]) + self.weights[b_name])
      else:
        self.h = tf.matmul(self.h, self.weights[w_name]) + self.weights[b_name]
    if output_layer == 'softmax':
      self.output = tf.nn.softmax(self.h)
      self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output)
    elif output_layer == 'linear':
      self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.h, self.labels), 2.0))
      self.output = self.h 
    #learning rate related operations
    self.lr = tf.Variable(0.001, trainable = False)
    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    #init = tf.global_variables_initializer()
    #self.sess = tf.Session()
    #self.sess.run(init)
  
  def new_session(self, sess):
    self.sess = sess
    self.sess.run(tf.global_variables_initializer())

  def _initialize_weights(self):
    all_weights = dict()
    for i in range(0, self.n_hidden+1):
      w_name = 'w_' + str(i);
      b_name = 'b_' + str(i);
      if i == 0:
        n_input = self.n_input
        n_output = self.n_units[i];
      elif i > 0 and i < self.n_hidden:
        n_input = self.n_units[i-1]
        n_output = self.n_units[i]
      else: #before the last output layer
        n_input = self.n_units[self.n_hidden - 1]
        n_output = self.n_output;

      all_weights[w_name] = tf.get_variable(w_name,  shape=[n_input, n_output], 
          initializer=tf.contrib.layers.xavier_initializer())
      all_weights[b_name] = tf.Variable(tf.zeros(n_output))

    return all_weights
        
  def partial_fit(self, X, Y, training):
    if training == True:
      cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.labels: Y})
    else:
      cost = self.sess.run(self.cost, feed_dict={self.x: X, self.labels: Y})
    return cost
  def get_output(self, X):
    return self.sess.run(self.output, feed_dict = {self.x: X});
  
  #for linear ouput layer regression task
  def get_mse(self, X, Y):
    return self.sess.run(self.cost, feed_dict={self.x: X, self.labels: Y})

  #for softmax classification task
  def eval_accuracy(self, X, Y):
    correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = self.sess.run(accuracy, feed_dict = {self.x:X, self.labels:Y})
    return acc

  def assign_lr(self, lr_value): #update the learning rate
    lr = self.sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})
    return lr
