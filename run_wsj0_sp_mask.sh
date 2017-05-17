#!/bin/bash

stage=2
config_dir=config/ # store list 'utt input1.ark:234 output1.ark:122
output_dir=data/tfrecords
num_layers=3
num_units=1024
out_left_context=25
out_right_context=25
in_left_context=25
in_right_context=25
keep_prob=1
apply_cmvn=1
 inputs_cmvn=data/wsj0_separation/kaldi_feats/train_inputs/cmvn.ark
  labels_cmvn=data/wsj0_separation/kaldi_feats/train_labels/cmvn.ark #if you don't want to use cmvn for labels, please let it ''
save_dir=exp_wsj0_sp_pit
if [ ! -d $config_dir ]; then
  mkdir $config_dir
fi

if [ $stage -le 0 ]; then
  echo "stage=$stage, conver the kaldi data to TFRecord."

  inputs_cmvn=data/wsj0_separation/kaldi_feats/train_inputs/cmvn.ark
  labels_cmvn=data/wsj0_separation/kaldi_feats/train_labels/cmvn.ark #if you don't want to use cmvn for labels, please let it ''
  num_threads=20
  for mode in wsj0_sp_train wsj0_sp_dev wsj0_sp_test; do
    #./tools/make_lists.sh $mode data/logspec_enhan_7000/${mode}_input.scp data/logspec_enhan_7000/${mode}_label.scp  #generate the config/$mode.lst

    python ./tools/io_funcs/convert_to_records_parallel4wsj0_sp.py \
      --mode=$mode --inputs_cmvn=$inputs_cmvn --labels_cmvn=$labels_cmvn \
      --num_threads=$num_threads --apply_cmvn=$apply_cmvn --keep_prob=$keep_prob || exit 1
  done
fi

if [ $stage -le 1 ]; then
  echo "stage=$stage, generate the tfrecord file lists and store them in config/"
  for mode in wsj0_sp_test wsj0_sp_dev wsj0_sp_train; do
    find `pwd`/$output_dir/$mode/ -iname "*tfrecords" > $config_dir/${mode}_tf.lst
  done
fi
if [ $stage -le 2 ]; then
  echo "state=$stage, begin training the model"
  learning_rate=0.001
  halving_factor=0.5
  training_impr=0.01
  load_model=
  pre_cv_costs=100000.0
  reject=0
  #if [ -f $save_dir/log.txt ];then

  #  rm $save_dir/log.txt
  #fi
 for i in `seq 1 12`; do 
    if [ $i -gt 1 ]; then
      load_model=`cat $save_dir/best.mdl | cut -d ' ' -f 1`
    fi

    python -u train_dnn_wsj0_sp_pit.py --num_iter=$i --learning_rate=$learning_rate --load_model=$load_model \
    --num_layers=$num_layers --num_units=$num_units --save_dir=$save_dir \
    --out_left_context=$out_left_context --out_right_context=$out_right_context \
    --in_left_context=$in_left_context --in_right_context=$in_right_context|| exit 1
    cur_cv_costs=`tail -n 1 $save_dir/log.txt | cut -d ' ' -f 2` 
    if [ $(echo "$cur_cv_costs <= $pre_cv_costs" | bc) = 1 ]; then
      best_model_name=`tail -n 1 $save_dir/log.txt | cut -d ' ' -f 1`
      echo $best_model_name > $save_dir/best.mdl
      pre_cv_costs=$cur_cv_costs
    else
     
      reject=$[$reject+1]
    fi
    if [ $reject -gt 0 ]; then
      learning_rate=$(echo " $learning_rate * $halving_factor  "|bc -l)
    fi
    rel_impr=$(bc <<< "scale=10; ($pre_cv_costs-$cur_cv_costs)/$pre_cv_costs")
   # if [ $(echo "$rel_impr < $training_impr" | bc) = 1 ];then
   #   if [ $reject -gt 0 ];then
   #     break
   #   fi
   # fi
  done
fi

echo 'Training Done'
mode=wsj0_sp_test
if [ $stage -le 3 ]; then
  load_model=`cat $save_dir/best.mdl`
  data_dir=`pwd`/data/wsj0_separation/test_pit/
  test_list=config/${mode}_tf.lst
  python   test_dnn_wsj0_sp_pit.py --load_model=$load_model --test_list=$test_list --data_dir=$data_dir \
  --out_left_context=$out_left_context --out_right_context=$out_right_context \
    --in_left_context=$in_left_context --in_right_context=$in_right_context|| exit 1

fi

if [ $stage -le 4 ]; then
  . ./path.sh 
  data_dir=`pwd`/data/wsj0_separation/test_pit/  #same as the stage 3
  scp_list=config/wsj0_sp_test_scp.lst
  ori_wav_path=/home/disk1/snsun/Workspace/tensorflow/kaldi/data/wsj0_separation/ori_wav/wsj0_mixed_test/
  rec_wav_path=data/wsj0_separation/rec_wav_test/
  mkdir -p $rec_wav_path
  find $data_dir -iname "*.scp" > $scp_list
  for line in `cat $scp_list`; do

    wavname=`basename -s .scp $line`
    w=`echo $wavname | awk -F '_' 'BEGIN{OFS="_"}{print $1,$2}'` 
    w=${w}.wav
    if [ $apply_cmvn -eq 1 ];then
      apply-cmvn --norm-vars=true --reverse=true $inputs_cmvn scp:$line  ark,scp:tmp_enhan.ark,tmp_enhan.scp || exit 1
    else
      copy-feats scp:$line ark,scp:tmp_enhan.ark,tmp_enhan.scp || exit 1
    fi
    python   ./tools/reconstruct_spectrogram.py tmp_enhan.scp ${ori_wav_path}/$w ${rec_wav_path}/${wavname} || exit 1
  done

rm tmp_enhan.ark
echo "Done OK!"
fi
