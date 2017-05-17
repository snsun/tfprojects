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
  filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=num_epochs)
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
def test(sess,coord,dnn,out_context_lst,data_list, file_list, data_dir):
  
  count = 0
  batch_x1 = data_list[0]
  batch_x2 = data_list[1]
  out_l_context = out_context_lst[0]
  out_r_context = out_context_lst[1]

  try: 
    while not coord.should_stop():
      bx1, bx2 = sess.run([batch_x1,batch_x2])
      cleaned1,cleaned2 = dnn.get_output(bx1, bx2);
      row, col = cleaned1.shape
      max_val = int(row/(out_l_context + out_r_context + 1))
      exrow = (max_val + 1)*(out_l_context + out_r_context + 1)
      excleaned1 = np.pad(cleaned1, ((0, exrow-row),(0,0)), 'minimum')
      idx = range(out_l_context, exrow, out_l_context + out_r_context + 1)

      tmp_c1 = excleaned1[idx,:]
      tmp_c1 = np.concatenate(tmp_c1)
      tmp_c1 = np.reshape(tmp_c1,[-1, 257])
      if tmp_c1.shape[0] >= row:
        data1 = tmp_c1[0:row, :]
      excleaned2 = np.pad(cleaned2, ((0, exrow-row),(0,0)), 'minimum')

      tmp_c1 = excleaned2[idx,:]
      tmp_c1 = np.concatenate(tmp_c1)
      tmp_c1 = np.reshape(tmp_c1,[-1, 257])
      if tmp_c1.shape[0] >= row:
        data2 = tmp_c1[0:row, :]

      tffilename = file_list[count]
      (_, name)=os.path.split(tffilename)
      (uttid, _) = os.path.splitext(name)
      (partname, _) = os.path.splitext(uttid)

      kaldi_writer1 = kio.ArkWriter(data_dir +'/' + partname + '_1.wav.scp')
      kaldi_writer2 = kio.ArkWriter(data_dir +'/' + partname  + '_2.wav.scp')
      kaldi_writer1.write_next_utt(data_dir +'/' + partname + '_1.wav.ark', uttid, data1)
      kaldi_writer2.write_next_utt(data_dir +'/' + partname + '_2.wav.ark', uttid, data2)
      kaldi_writer1.close()
      kaldi_writer2.close()
      count = count + 1
      if count % 500 == 0: 
        print '>>>>Processing ', count ,'<<<<\n'
  except tf.errors.OutOfRangeError:
    return
  finally:
   
      coord.request_stop()
      
def get_mini_batch(sess, coord, dnn, file_list, in_context, out_context, batch_size, num_threads,num_epoches):
  feats, labels1, labels2 = read_and_decode(file_list, 257, 257, num_epoches)
  sess.run(tf.local_variables_initializer())
  sfeats1 = splice_feats(feats, in_context[0], in_context[1])
  sfeats2 = splice_feats(feats, out_context[0], out_context[1])
  #slabels1 = splice_feats(labels1, out_context[0], out_context[1])
  #slabels2 = splice_feats(labels2, out_context[0], out_context[1])

  return sfeats1, sfeats2

def main(_):

  pre_cv_costs = float('Inf')
  halving_factor = FLAGS.halving_factor
  save_dir = FLAGS.save_dir
  data_dir = FLAGS.data_dir
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
  test_list,len_test = process_file_list(FLAGS.test_list)
  lr = FLAGS.learning_rate
  keep_prob = FLAGS.keep_prob
  load_model = FLAGS.load_model 
  sess = tf.Session()
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)


  dnn = ff.FeedForward(input_dim*(in_l+in_r+1),(out_l + 1 + out_r)* output_dim, num_layers, [num_units], active_func, output_layer = 'linear',keep_prob=keep_prob)
  dnn.new_session(sess)
  saver = tf.train.Saver()
  if load_model != '':
    saver.restore(dnn.sess, load_model)
  dnn.assign_lr(FLAGS.learning_rate)
  
  coord = tf.train.Coordinator()
  test_batch_x1,test_batch_x2 = get_mini_batch(sess, coord, dnn, test_list, [in_l, in_r], [out_l,out_r], batch_size,num_threads, 1)
  
  thread = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  stime = datetime.datetime.now()
  
  test(sess,coord, dnn, [out_l,out_r], [test_batch_x1,test_batch_x2],test_list, data_dir)
  etime = datetime.datetime.now()
  print 'Test time: ', etime - stime
  
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
    '--test_list',
    default='config/wsj0_sp_test_tf.lst',
    type=str,
    help='Test feature and label tf list.')
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
  parser.add_argument(
    '--data_dir',
    type= str,
    default='data/wsj0_sp_test/',
    help = 'Directory to put the network output')
  FLAGS,unparsed = parser.parse_known_args()
  sys.stdout.flush()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    

