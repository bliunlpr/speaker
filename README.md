# Deep Segment Attentive Embedding for Speaker Verification

This is an implementation of Deep Segment Attentive Embedding on Python 3, Pytorch. The model learn the unified speaker embeddings for utterances of variable duration. It is the 'nlpr_fight' teamwork of Tongdun speaker verification  competition. 

# Requirements
Python 3.5, Pytorch 0.4.0.
## AISHELL
You can download [AISHELL](http://www.aishelltech.com/kysjcp) to run the code.

# Usage:
You can run ```sh run.sh ``` for AISHELL, but it's recommended that you run the commands one by one. You need set ```aishell_data='Your AISHELL Dataset Dir' such as /mnt/DATA/AiShell/data_aishell/wav/``` in run.sh.

## (0) Preparing dataset
### AISHELL
``` 
## make dataset
featdir=fbank/
for x in train dev test; do
    python3 local/aishell_data_prep.py  $aishell_data/$x data/$x $featdir/$x || exit 1;
done 
## make test_pair for verification
for x in dev test; do
    python3 local/make_pairs.py data/$x 1500 1500 || exit 1;
done 
```
### Your Own Dataset
You need build train and dev directory. Each has ```feats.scp``` and ```utt2spk```.  Each line of ```feats.scp``` is "utt_id feats_path" and each line of ```utt2spk``` is "utt_id spk_id". The dev directory also need ```pair.txt```. Each line of ```pair.txt``` is "utt_id0 utt_id1 feats_path0 feats_path1 label". You can run ```python3 local/make_pairs.py dev 1500 1500``` to randomly build the ```pair.txt```.

## (1) Train model
We provide two training methods. One is based on "Generalized End-to-End Loss for Speaker Verfication"[GE2E](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462665). Anonther is our proposed Deep Segment Attentive Embedding mehod[DSAE]. The model is LSTM or CNN. We also provide different attention strategies. 
### for GE2E
```
sh run.sh --stage 2 --seq_training false --model_type lstm --train_type  base_attention | last_state | base_attention ```   
## CNN does not use attention but avgpool.
sh run.sh --stage 2 --seq_training false --model_type cnn 
```
### for DSAE
```                                                                                                                           
sh run.sh --stage 2 --seq_training true --model_type lstm | cnn  --train_type base_attention | last_state | multi_attention 
--segment_type  none | all | average
``` 
