#!/bin/bash

data_dir=`pwd`/data/wsj0_separation/kaldi_feats/
wav_scp=$data_dir/wav.scp
for x in test; do
  for y in inputs labels; do

  compute-spectrogram-feats --window-type="hamming" scp:$data_dir/${x}_${y}/wav.scp ark,scp:$data_dir/${x}_${y}/feats.ark,$data_dir/${x}_${y}/feats.scp &
  done
done
wait
for x in test; do
  for y in inputs labels; do
  
  compute-cmvn-stats scp:$data_dir/${x}_${y}/feats.scp $data_dir/${x}_${y}/cmvn.ark &
  done
done

