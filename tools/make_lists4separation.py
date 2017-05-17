import sys

mode = 'train'
inputscp ='data/wsj0_separation/kaldi_feats/train_inputs/feats.scp'
outputscp = 'data/wsj0_separation/kaldi_feats/train_labels/feats.scp'
lst='config/wsj0_sp_' + mode + '.lst'
fid1 = open(inputscp, 'r')
lines1 = fid1.readlines()
fid2 = open(outputscp, 'r')
lines2 = fid2.readlines()

fid1.close()
fid2.close()

fid3 = open(lst, 'w')

dict1 = {}
dict2 = {}
for line in lines2:
  l = line.rstrip('\n')
  strs = l.split(' ')
  dict1[strs[0]] = strs[1]

for line in lines1:
  line = line.rstrip('\n')
  strs = line.split(' ')
  utt = strs[0]
  cont = strs[1]
  names = utt.split('.')
  name1 = names[0]+'_1.wav'
  name2 = names[0] + '_2.wav'
  fid3.write(utt + ' ' + cont +' ' + dict1[name1] + ' ' + dict1[name2] + '\n')

fid3.close()
  



 
