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


    self.feature = tf.placeholder(tf.float32, [None, self.n_input])
    self.mixedspeech = tf.placeholder(tf.float32, [None, self.n_output])

    self.labels1 = tf.placeholder(tf.float32, [None, self.n_output])
    self.labels2 = tf.placeholder(tf.float32, [None, self.n_output])

    for i in range(0, n_hidden +1):
      w_name = 'w_' + str(i);
      b_name = 'b_' + str(i);
      if i == 0:
        self.h = self.active_funcs(tf.matmul(self.feature, self.weights[w_name]) + self.weights[b_name])  
        if keep_prob != 1:
          self.h = tf.nn.dropout(self.h, keep_prob)

      elif i > 0 and i < n_hidden:
        self.h = self.active_funcs(tf.matmul(self.h, self.weights[w_name]) + self.weights[b_name])
      else:
        self.h = tf.matmul(self.h, self.weights[w_name]) + self.weights[b_name]
    self.mask = tf.sigmoid(self.h) 
    self.cleaned1 = self.mask * self.mixedspeech
    self.cleaned2 = (1-self.mask) * self.mixedspeech
    cost1 = tf.reduce_sum(tf.pow(self.cleaned1 - self.labels1, 2.0), reduction_indices=1) + tf.reduce_sum(tf.pow(self.cleaned2 - self.labels2, 2.0), reduction_indices=1)
    cost2 = tf.reduce_sum(tf.pow(self.cleaned2 - self.labels1, 2.0), reduction_indices=1) + tf.reduce_sum(tf.pow(self.cleaned1 - self.labels2, 2.0), reduction_indices=1)
    idx = tf.cast(cost1 > cost2, tf.float32)
    self.cost = tf.reduce_sum(idx*cost2 + (1-idx)*cost1)
     
    #self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.h, self.labels), 2.0))
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
        
  def partial_fit(self, feature, mixedspeech,labels1, labels2,  training):
    if training == True:
      cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.feature: feature, 
                                                                          self.mixedspeech: mixedspeech,
                                                                          self.labels1: labels1,
                                                                          self.labels2: labels2})
    else:
      cost = self.sess.run(self.cost, feed_dict={self.feature: feature, 
                                                 self.mixedspeech: mixedspeech,
                                                 self.labels1: labels1,
                                                 self.labels2: labels2})

    return cost
  def get_output(self, feature, mixedspeech):
    #slice1 = tf.slice(self.cleaned1, [0,l*dim], [-1, dim])
    #slice2 = tf.slice(self.cleaned2, [0,l*dim], [-1, dim])
    clean1,clean2= self.sess.run((self.cleaned1, self.cleaned2), feed_dict = {self.feature: feature, self.mixedspeech: mixedspeech});

    return clean1, clean2
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
