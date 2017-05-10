import tensorflow as tf
import inspect

class LSTM(object):
  """The LSTM model."""
  def __init__(self, 
               size, 
               num_layers, 
               max_gradient_norm, 
               batch_size,
               output_dim,
               learning_rate, 
               learning_rate_decay_factor = 0.5,
               keep_prob = 0.1,
               feats_dim = 257,
               last_layer = 'softmax',
               use_gru=False,
               forward_only = False):
    self.batch_size = batch_size;
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable = False);
    self.keep_prob = keep_prob
    self.feats_dim = feats_dim
    self.output_dim = output_dim
    self.last_layer = last_layer
    def single_cell():
      return tf.contrib.rnn.BasicLSTMCell(size)

    if use_gru:
      def single_cell():
        return tf.contrib.rnn.GRUCell(size)
    if keep_prob < 1:
      drop_cell = tf.contrib.rnn.DropoutWrapper(single_cell(), output_keep_prob = keep_prob) 
    else:
      drop_cell = single_cell()
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell for _ in range(num_layers)])
    
    self.data = tf.placeholder(tf.float32, [batch_size, None, self.feats_dim])
    self.target = tf.placeholder(tf.float32, [batch_size, None, self.output_dim])
    self.sequence_length = tf.placeholder(tf.float32, [batch_size])
    output, state = tf.nn.dynamic_rnn(cell, self.data, sequence_length = self.sequence_length, dtype = tf.float32)
    W = tf.get_variable("last_layer_w", [size, self.output_dim], dtype=tf.float32);
    b = tf.get_variable("last_layer_b", [self.output_dim], dtype=tf.float32);
    myshape = output.get_shape()
    affine_output = tf.matmul(tf.reshape(output, [-1, size]), W) + b
    print tf.reshape(output, [-1, size]).get_shape()
    print affine_output.get_shape()
    affine_output = tf.reshape(affine_output, [batch_size, -1, output_dim])
    def cost_func(output, target, loss_type='ce'): 
      if loss_type == 'ce':
        loss = target * tf.log(output)
        loss = -tf.reduce_sum(loss, reduction_indices = 2)
      
      elif loss_type == 'mse':
        loss = tf.square(output - target)
        loss = tf.reduce_sum(loss, reduction_indices = 2)
      mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices = 2))
      loss = loss * mask
      loss = tf.reduce_sum(loss, reduction_indices = 1)
      loss /= tf.reduce_sum(mask, reduction_indices = 1)
      return tf.reduce_mean(loss),output

    if self.last_layer == 'softmax':
      cost,final_output = cost_func(tf.softmax(affine_output), self.target, 'ce')
    elif self.last_layer == 'linear':
      cost, final_output = cost_func(affine_output, self.target, 'mse')
    elif self.last_layer == 'sigmoid':
      cost, final_output = cost_func(tf.sigmoid(affine_output), self.target, 'mse')
    self.cost = cost
    self.final_output = final_output
    if not forward_only:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                        max_gradient_norm)
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars), 
                                                global_step = tf.contrib.framework.get_or_create_global_step())
    
  def assign_lr(self, session, lr_value):
    lr=session.run(self.learning_rate_decay_op, feed_dict={self.learning_rate: lr_value})
    return lr
  
  def train(self, session, X, Y,sequence_length):
    cost, _ = session.run((self.cost, self.train_op), feed_dict = {self.data: X, self.target: Y, self.sequence_length:sequence_length}) 
    return cost
  def get_output(self, session, X, sequence_length):
    output = tf.run(self.final_output, feed_dict = {self.data:X, self.sequence_length:sequence_length})
    return output

  def cost_func(self, output, target, loss_type='ce'): 
    if loss_type == 'ce':
      loss = target * tf.log(output)
      loss = -tf.reduce_sum(loss, reduction_indices = 2)
      
    elif loss_type == 'mse':
      loss = tf.square(output - target)
      loss = tf.reduce_sum(loss, reduction_indices = 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target, reduction_indices = 2)))
    loss = loss * mask
    loss = tf.reduce_sum(loss, reduction_indices = 1)
    loss /= tf.reduce_sum(mask, reduction_indices = 1)
    return tf.reduce_mean(loss),output

  

