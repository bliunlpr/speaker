#!/bin/bash

stage=0
seq_training=false
train_type="multi_attention"
. local/parse_options.sh  # e.g. this parses the --stage option if supplied.


aishell_data=/mnt/nlpr/DATA/Audio/Chinese/AiShell/data_aishell/wav/

if [ $stage -le 0 ]; then  
  featdir=fbank/
  for x in train dev test; do
    python3 local/aishell_data_prep.py  $aishell_data/$x data/$x $featdir/$x || exit 1;
  done
fi


if [ $stage -le 1 ]; then
  for x in dev test; do
    python3 local/make_pairs.py data/$x 1500 1500 || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  
  python3 local/main.py --dataroot data --seq_training $seq_training --train_type $train_type --name speaker_seq_"$seq_training"_$train_type|| exit 1;
  
fi