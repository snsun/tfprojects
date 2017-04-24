import numpy as np
import tensorflow as tf
import tools.io_funcs.kaldi_io as kio
from tools.io_funcs import feats_trans
import random
import models.feed_forward as ff

fscp = 'data/logspec_enhan_7000/clean_7000.scp'
tscp = 'data/logspec_enhan_7000/noise_7000.scp'
dim = 257
l = 2 
r = 2
total_costs = 0
total_frames = 0
x = tf.placeholder(tf.float32, [None, dim *(l+r+1)])
y_ =  tf.placeholder(tf.float32, [None, dim])

batch_size = 256

feats_reader = kio.ArkReader(fscp);
targets_reader = kio.ArkReader(fscp);
feats_randomizer = feats_trans.RandomizerMask()
targets_randomizer = feats_trans.RandomizerMask()

dnn = ff.FeedForward(dim*(l+r+1), dim, 2, [512], tf.nn.relu, output_layer = 'linear')
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
    i = i + batch_size 
    cost = dnn.partial_fit(feats, targets);
    total_costs = total_costs + cost;
    total_frames = total_frames +( e-s)
  
  feats_randomizer.clear();
  targets_randomizer.clear()
  if (total_frames) * 10 /1000 % 3600 < 200: 
    print '>>>>costs at the ',float(total_frames) * 10 /1000 / 3600, ' is ',total_costs/total_frames ,'<<<<\n'



