import numpy as np
import tensorflow as tf
import tools.io_funcs.kaldi_io as kio
from tools.io_funcs import feats_trans
import random
import models.feed_forward as ff
import os, sys

'''
training or cv of nn
  feats_reader/targets_reader: kaldi reader 
  feats_randomizer/targets_randomizer: random select data
  l/r: left and right windows length, integer.
  training: True/Fasle, if true, traning, else cv
'''
def train(dnn, fscp, tscp, batch_size, l, r, training):
  total_frames = 0
  total_costs = 0
  
  feats_reader = kio.ArkReader(fscp);
  targets_reader = kio.ArkReader(tscp);
  feats_randomizer = feats_trans.RandomizerMask()
  targets_randomizer = feats_trans.RandomizerMask()

  while (1):  
    while(1):
      [keys, values, looped] = feats_reader.read_next_utt()
      targets = targets_reader.read_utt_data_from_id(keys)
      if looped == True:
        break
      [rows, cols] = values.shape
      splice_values = feats_trans.splice_feats(values, l, r);
      
      full = feats_randomizer.add_data(splice_values)
      targets_randomizer.add_data(targets);
  
      if full == True:
        break
    if looped == True:
      break
    real_size = feats_randomizer.real_size;
    mask = random.sample(range(real_size), real_size)
    i = 0;
    while i < real_size:
      s = i;
      e = i +  batch_size
      e = min(e, real_size)
      feats = feats_randomizer.read_batches(mask[s:e])
      targets = targets_randomizer.read_batches(mask[s:e])
      i = i + (e-s) 
      cost = dnn.partial_fit(feats, targets,training);
      total_costs = total_costs + cost;
      total_frames = total_frames +( e-s)
    
    feats_randomizer.clear();
    targets_randomizer.clear()
    if (total_frames) * 10 /1000 % 3600 < 200: 
      print '>>>>costs at the ',float(total_frames) * 10 /1000 / 3600, ' is ',total_costs/total_frames ,'<<<<\n'
  return total_costs/total_frames
 


fscp = 'data/logspec_enhan_7000/clean_7000.scp'
tscp = 'data/logspec_enhan_7000/noise_7000.scp'
cv_fscp = 'data/logspec_enhan_7000/cv_clean_1000.scp'
cv_tscp = 'data/logspec_enhan_7000/cv_noise_1000.scp'
dim = 257
l = 2 
r = 2
batch_size = 256
pre_cv_costs = float('Inf')

sess = tf.Session()
dnn = ff.FeedForward(dim*(l+r+1), dim, 4, [256], tf.nn.relu, output_layer = 'linear')
dnn.new_session(sess)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
max_iters = 3
lr = sess.run(dnn.lr)
reject = 0

for iter_num in range(max_iters):
  cur_tr_costs = train(dnn, fscp,tscp, batch_size, l, r, True);
  print "Training cost is: ", cur_tr_costs
  cur_cv_costs = train(dnn, cv_fscp, cv_tscp, batch_size, l, r, False)
  print "CV cost is: ", cur_cv_costs;
  if cur_cv_costs > pre_cv_costs:
    lr = lr/2;
    reject = reject + 1;
    if reject > 1:
      print "Training is over! The best model is ", best_model
      break
    dnn.assign_lr(lr)
    sess = tf.Session()
    dnn.new_session(sess)
    saver.restore(dnn.sess, best_model) 
  else:
    save_path = './exp/train_iter'+str(iter_num)

    if not os.path.exists(save_path):
      os.makedirs(save_path)
    best_model = saver.save(dnn.sess, save_path+'/train_iter'+str(iter_num)+'_tr_'+ str(cur_tr_costs)+'_cv_'+str(cur_cv_costs))
    pre_cv_costs = cur_cv_costs

