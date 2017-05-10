import numpy as np
import tensorflow as tf
import tools.io_funcs.kaldi_io as kio
import models.lstm as lstm
import os, sys

def train(sess, lstm, fscp, tscp, batch_size, training):
  total_frame = 0;
  total_costs = 0
  cache_size = 100*batch_size
  feats_cache_list = []
  targets_cache_list = []
  len_list = []
  looped = False
  feats_reader = kio.ArkReader(fscp);
  targets_reader = kio.ArkReader(tscp);
  while(1):
    while(1):
      
      [keys, values, looped] = feats_reader.read_next_utt()
      targets = targets_reader.read_utt_data_from_id(keys)
      [rows, indim] = values.shape
      [rows, outdim] = targets.shape
      feats_cache_list.append(values)
      targets_cache_list.append(targets)
      len_list.append(rows)
      if len(len_list) == cache_size:
        break
      if looped:
        break;
    tmp_list = zip(len_list, range(len(len_list)))
    total_frame = total_frame + len(len_list)
    tmp_list.sort(key=lambda x: x[0])
    index =[ x[1] for x in tmp_list]
    sort_list = [ x[0] for x in tmp_list]
    i = 0
    real_len = len(len_list)
    cur_position = 0
    while cur_position < real_len:
      if real_len - cur_position > batch_size:
        stride = batch_size
      else:
        stride = real_len - cur_position
        break
      cur_index = index[cur_position:cur_position+stride]
      sequence_length = sort_list[cur_position:cur_position+stride]
      max_len = max(sequence_length)

      batch_x = np.zeros([stride, max_len, indim])
      batch_y = np.zeros([stride, max_len, outdim])
      for i,v in enumerate(cur_index):
        batch_x[i, 0:sequence_length[i], :] = feats_cache_list[v]
        batch_y[i, 0:sequence_length[i], :] = targets_cache_list[v]
      cost = lstm.train(sess, batch_x, batch_y, np.array(sequence_length))
      cur_position = cur_position + stride
     # print cost
      total_costs = total_costs + cost * sum(sequence_length)
      total_frame= total_frame + sum(sequence_length) 
    feats_cache_list = []
    targets_cache_list = []
    len_list = []
    print total_costs/total_frame
    if looped: 
      break
    
fscp = 'data/logspec_enhan_7000/clean_7000.scp'
tscp = 'data/logspec_enhan_7000/noise_7000.scp'
cv_fscp = 'data/logspec_enhan_7000/cv_clean_1000.scp'
cv_tscp = 'data/logspec_enhan_7000/cv_noise_1000.scp'
dim = 257
lstm = lstm.LSTM(size=128,
            num_layers = 2,
            max_gradient_norm = 5,
            batch_size = 10,
            learning_rate = 0.01,
            output_dim = 257, 
            last_layer = 'linear')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
max_iters = 3
for i in range(max_iters):
  train(sess, lstm, fscp, tscp, 10, True)



