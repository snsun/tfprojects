import numpy as np

def splice_feats(feats, l, r):
  '''
  splice feats matrix with [-l, r] contexts
  '''
  [rows, cols] = feats.shape
  sfeats = feats.copy();

  i = 1;
  
  while i <= l :

    tmp = feats.copy()
    tmp[i:,:] = tmp[0:-i, :]
    tmp[0:i] = feats[0,:]
    sfeats = np.concatenate((tmp, sfeats), axis = 1)
    i = i+1;

  i = 1
  while i <= r:
    tmp = feats.copy()
    tmp[0:-i, :] = tmp[i:, :]
    tmp[-i:,:] = feats[-1, :]
    sfeats = np.concatenate((sfeats, tmp), axis = 1)
    i = i + 1
  return sfeats

class RandomizerMask(object):
  def __init__(self, size=10000):
    self.size = size
    self.real_size = 0
    self.position = 0
    self.cur_len = 0;
    self.full = False
    self.data_list = []
  def add_data(self, data):
    if self.cur_len < self.size: 
      self.data_list.append(data)
      self.data = np.concatenate(self.data_list)
      self.cur_len =  self.cur_len + data.shape[0] 
      self.real_size = self.cur_len
    if self.cur_len > self.size:
      self.full = True
    return self.full

  def read_batches(self, mask):
    return self.data[mask]
  
  def clear(self):
    self.full = False
    self.cur_len = 0
    self.data_list = []
    self.real_size = 0


    
