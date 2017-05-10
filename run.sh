#!/bin/bash

stage=0
config_dir=config/ # store list 'utt input1.ark:234 output1.ark:122
output_dir=data/tfrecords
mkdir $config_dir

if [ $stage -le 0 ]; then
  echo "stage=$stage, conver the kaldi data to TFRecord."
  tr_feats_scp=data/logspec_enhan_7000/train_input.scp #kaldi feats scp file path
  tr_targets_scp=data/logspec_enhan_7000/train_label.scp # kaldi targets scp file path
  cv_feats_scp=data/logsepc_enhan_7000/dev_input.scp
  cv_targets_scp=data/logspec_enhan_7000/dev_label.scp
  apply_cmvn=false
  mode=dev
  inputs_cmvn=data/inputs.cmvn
  labels_cmvn=data/labels.cmvn #if you don't want to use cmvn for labels, please let it ''
  num_threads=20
  for mode in train dev; do
    ./tools/make_lists.sh $mode data/logspec_enhan_7000/${mode}_input.scp data/logspec_enhan_7000/${mode}_label.scp  #generate the config/$mode.lst
    python ./tools/io_funcs/convert_to_records_parallel.py --apply_cmvn=$apply_cmvn \
      --mode=$mode --inputs_cmvn=$inputs_cmvn --labels_cmvn=$labels_cmvn \
      --num_threds=$num_threads
  done
fi
if [ $stage -le 1 ]; then
  echo "stage=$stage, generate the tfrecord file lists and store them in config/"
  for mode in dev train; do
    find `pwd`/$output_dir/$mode/ -iname "*tfrecords" > $config_dir/${mode}_tf.lst
  done
fi
if [ $stage -le 2 ]; then
  learning_rate=0.001
  halving_factor=0.5
  training_impr=0.01
  load_model=
  pre_cv_costs=100000.0
  reject=0
  if [ -f exp/log.txt ];then

    rm exp/log.txt
  fi
 for i in `seq 1 20`; do
    if [ $i -gt 1 ]; then
      load_model=`cat exp/best.mdl | cut -d ' ' -f 1`
    fi

    python train_dnn_tfrecords.py --num_iter=$i --learning_rate=$learning_rate --load_model=$load_model
    cur_cv_costs=`tail -n 1 exp/log.txt | cut -d ' ' -f 2` 
    if [ $(echo "$cur_cv_costs <= $pre_cv_costs" | bc) = 1 ]; then
      best_model_name=`tail -n 1 exp/log.txt | cut -d ' ' -f 1`
      echo $best_model_name > exp/best.mdl
      pre_cv_costs=$cur_cv_costs
      reject=$[$reject+1]
    else
      learning_rate=$(echo " $learning_rate * $halving_factor  "|bc -l)
    fi
    if [ $reject -gt 3 ];then
      break
    fi
  done
fi
echo 'Training Done'

if [ $stage -le 3 ]; then
  load_model=`cat exp/best.mdl`
  test_list=config/dev_tf.lst
  python test_dnn_tfrecords.py --load_model=$load_model --test_list=$test_list
fi
