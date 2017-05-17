import numpy as np
import tensorflow as tf
import tools.io_funcs.kaldi_io as kio
from tools.io_funcs import feats_trans
import random
import models.feed_forward_pit as ff
import os, sys, argparse, datetime

def process_file_list(file_list):
  fid = open(file_list,'r')
  proc_file_list=[]
  lines = fid.readlines()
  for line in lines:
    proc_file_list.append(line.rstrip('\n'))
  return proc_file_list, len(lines)

def read_and_decode(filename, input_dim, label_dim, num_epochs):
  filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  _,features = tf.parse_single_sequence_example(serialized_example,
    sequence_features={
      'inputs':tf.FixedLenSequenceFeature([input_dim],tf.float32),
      'labels1':tf.FixedLenSequenceFeature([label_dim], tf.float32),
      'labels2':tf.FixedLenSequenceFeature([label_dim], tf.float32)})
  return features['inputs'], features['labels1'], features['labels2']

def splice_feats(feats, l, r):
  sfeats = []
  row = tf.shape(feats)[0]
  for i in range(l, 0, -1):
    f1 = tf.slice(feats, [0, 0], [row-i, -1])
    for j in range(i):
      f1 = tf.pad(f1, [[1,0],[0,0]],mode='SYMMETRIC')
    sfeats.append(f1)

  sfeats.append(feats)
  for i in range(1,r+1):
    f1 = tf.slice(feats, [i, 0], [-1, -1])
    for j in range(i):
      f1 = tf.pad(f1, [[0,1],[0,0]],mode='SYMMETRIC')
    sfeats.append(f1)
  return tf.concat(sfeats, 1)
 

'''
training or cv of nn
  feats_reader/targets_reader: kaldi reader 
  feats_randomizer/targets_randomizer: random select data
  l/r: left and right windows length, integer.
  training: True/Fasle, if true, traning, else cv
'''
def train(sess,coord,dnn, batch_size,data_list, num_threads,training,clear_stop):
  

  total_frames = 0.0
  total_costs = 0.0
  batch_x1 = data_list[0]
  batch_x2 = data_list[1]
  batch_y1 = data_list[2]
  batch_y2 = data_list[3]
  try: 
    while not coord.should_stop():
      bx1, bx2, by1, by2 = sess.run([batch_x1,batch_x2,batch_y1, batch_y2])
      cost = dnn.partial_fit(bx1,bx2,by1, by2, training);
      total_costs = total_costs + cost;
      total_frames = total_frames + batch_size
    
      if (total_frames/batch_size) % 1460 == 0: 
        print '>>>>costs at the ',float(total_frames) * 10 /1000 / 3600, ' is ',total_costs/total_frames ,'<<<<\n'
  except tf.errors.OutOfRangeError:
    print "Done"
    if clear_stop:
      coord.clear_stop()
      return total_costs/total_frames
  finally:
    if not clear_stop:
      coord.request_stop()
  return total_costs/total_frames

def get_mini_batch(sess, coord, dnn, file_list, in_context, out_context, batch_size, num_threads,num_epoches):
  feats, labels1, labels2 = read_and_decode(file_list, 257, 257, num_epoches)
  sess.run(tf.local_variables_initializer())
  sfeats1 = splice_feats(feats, in_context[0], in_context[1])
  sfeats2 = splice_feats(feats, out_context[0], out_context[1])
  slabels1 = splice_feats(labels1, out_context[0], out_context[1])
  slabels2 = splice_feats(labels2, out_context[0], out_context[1])

  slice_queue = tf.RandomShuffleQueue(capacity=batch_size*50,
    min_after_dequeue = 0,
    dtypes = ['float', 'float', 'float','float'],
    shapes = [[dnn.n_input,],[dnn.n_output],[dnn.n_output,],[dnn.n_output]])
  batch_x1,batch_x2, batch_y1, batch_y2 = slice_queue.dequeue_many(batch_size)
  enqueue = [slice_queue.enqueue_many([sfeats1,sfeats2, slabels1, slabels2])]*num_threads
  qr = tf.train.QueueRunner(slice_queue, enqueue )
  qr.create_threads(sess, coord=coord, start=True)
  return batch_x1,batch_x2, batch_y1, batch_y2

def main(_):

  pre_cv_costs = float('Inf')
  halving_factor = FLAGS.halving_factor
  save_dir = FLAGS.save_dir
  iter_num = FLAGS.num_iter
  in_l = FLAGS.in_left_context
  in_r = FLAGS.in_right_context
  out_l = FLAGS.out_left_context
  out_r = FLAGS.out_right_context
  input_dim = FLAGS.input_dim
  output_dim = FLAGS.output_dim
  num_layers = FLAGS.num_layers
  num_units = FLAGS.num_units
  output_layer = FLAGS.output_layer
  batch_size = FLAGS.batch_size
  active_func = tf.nn.relu
  num_threads = FLAGS.num_threads 
  train_list,len_train = process_file_list(FLAGS.train_list)
  dev_list,len_dev = process_file_list(FLAGS.dev_list)
  lr = FLAGS.learning_rate
  keep_prob = FLAGS.keep_prob
  load_model = FLAGS.load_model 
  sess = tf.Session()
  
  dnn = ff.FeedForward(input_dim*(in_l+in_r+1),(out_l + 1 + out_r)* output_dim, num_layers, [num_units], active_func, output_layer = 'linear',keep_prob=keep_prob)
  dnn.new_session(sess)
  saver = tf.train.Saver()
  if load_model != '':
    saver.restore(dnn.sess, load_model)
  dnn.assign_lr(FLAGS.learning_rate)
  
  coord = tf.train.Coordinator()
  train_batch_x1,train_batch_x2, train_batch_y1,  train_batch_y2 = get_mini_batch(sess, coord, dnn, train_list, [in_l, in_r], [out_l,out_r], batch_size,num_threads, 1)
  dev_batch_x1,dev_batch_x2,dev_batch_y1, dev_batch_y2 = get_mini_batch(sess, coord, dnn, dev_list, [in_l,in_r], [out_l, out_r], batch_size,num_threads, 1)
  
  thread = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  stime = datetime.datetime.now()
  
  cur_tr_costs = train(sess,coord, dnn, batch_size, [train_batch_x1,train_batch_x2,train_batch_y1, train_batch_y2], num_threads, True,True)
  etime = datetime.datetime.now()
  print 'Training cost: ',cur_tr_costs, etime - stime
  
  cur_cv_costs = train(sess,coord, dnn, batch_size, [dev_batch_x1,dev_batch_x2,dev_batch_y1, dev_batch_y2], num_threads, False,False)
  print 'CV cost: ', cur_cv_costs,datetime.datetime.now() - stime
  
  save_path = save_dir + '/train_iter'+str(iter_num)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  best_model = saver.save(dnn.sess, save_path+'/train_iter'+str(iter_num)+'_lr_' + str(lr) + '_tr_'+ str(cur_tr_costs)+'_cv_'+str(cur_cv_costs))

  # record the useful information
  fid = open(save_dir+'/log.txt','a')
  fid.write(best_model+' '+str(cur_cv_costs)+' ' +str(cur_tr_costs) + '\n')
  fid.close()
  coord.join(thread)
   
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input_dim',
    default = 257,
    type=int,
    help = 'Input feature dim with out context windows len.')
  parser.add_argument(
    '--output_dim',
    default = 257,
    type=int,
    help = 'Output feature dim with out context windows len.')
  parser.add_argument(
    '--in_left_context',
    default = 20,
    type= int,
    help = 'Left context lengh for slicing feature')
  parser.add_argument(
    '--in_right_context',
    default = 20,
    type= int,
    help = 'Right context lengh for slicing feature')
  parser.add_argument(
    '--out_left_context',
    default = 2,
    type= int,
    help = 'Left context lengh for slicing feature')
  parser.add_argument(
    '--out_right_context',
    default = 2,
    type= int,
    help = 'Right context lengh for slicing feature')
  parser.add_argument(
    '--num_layers',
    default=3,
    type=int,
    help = 'Number of hidden layers.')
  parser.add_argument(
    '--num_units',
    default=1024,
    type=int,
    help='Number of nuros in every layer')
  parser.add_argument(
    '--train_list',
    default='config/wsj0_sp_train_tf.lst',
    type=str,
    help='Training feature and label tf list.')
  parser.add_argument(
    '--dev_list',
    default='config/wsj0_sp_dev_tf.lst',
    type=str,
    help = 'Developement feature and label tf. list')
  parser.add_argument(
    '--num_iter',
    default=1,
    type=int,
    help='Number of training epoches')
  parser.add_argument(
    '--batch_size',
    default=128,
    type=int,
    help='Batch size')
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0005,
    help='Initinal learning rate')
  parser.add_argument(
    '--halving_factor',
    default= '0.5',
    type = float,
    help = 'Halving factor using to ajust learning rate')
  parser.add_argument(
    '--num_threads',
    type=int,
    default=12,
    help = 'The number of threads reading the tfrecords')
  parser.add_argument(
    '--save_dir',
    type= str,
    default='exp_wsj0_sp_pit/',
    help = 'Directory to put the trained model')
  parser.add_argument(
    '--load_model',
    type=str,
    default='',
    help = 'The model name we need to load, default is \'\'')
  parser.add_argument(
    '--keep_prob',
    type=float,
    default=0.8,
    help = 'Kepp probability for training dropout')
  parser.add_argument(
    '--output_layer',
    default = 'linear',
    type=str,
    help= 'The output layer type, softmox or linear')
  parser.add_argument(
    '--active_func',
    default=tf.nn.relu,
    type=str,
    help = 'The active function of hidden layers')
  FLAGS,unparsed = parser.parse_known_args()
  sys.stdout.flush()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    

