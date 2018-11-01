# Deep Segment Attentive Embedding for Speaker Verification

This is an implementation of Deep Segment Attentive Embedding on Python 3, Pytorch. The model learn the unified speaker embeddings for utterances of variable duration. It is the 'nlpr_fight' teamwork of Tongdun speaker verification  competition. 

# Requirements
Python 3.5, Pytorch 0.4.0.
## AISHELL
You can download [AISHELL](http://www.aishelltech.com/kysjcp) to run the code.

# Usage:
You can run ```sh run.sh ``` for AISHELL, but it's recommended that you run the commands one by one. You must set ```aishell_data='Your AISHELL Dataset Dir' such as /mnt/DATA/AiShell/data_aishell/wav/``` in run.sh.

## (0) Preparing dataset
### AISHELL

``` featdir=fbank/
for x in train dev test; do
    python3 local/aishell_data_prep.py  $aishell_data/$x data/$x $featdir/$x || exit 1;
done ```

### Your Own Dataset


usage:
### for no seq_training  

```sh run.sh --stage 0 --seq_training false --model_type lstm --train_type base_attention | last_state | base_attention ```  

```sh run.sh --stage 0 --seq_training false --model_type lstm --train_type base_attention | last_state | base_attention  ```  

```sh run.sh --stage 0 --seq_training false --model_type cnn ```

  
for seq_training                                                                                                                          
  sh run.sh --stage 0 --seq_training true --model_type lstm --train_type base_attention | last_state | base_attention --segment_type none | all | average
  
  sh run.sh --stage 0 --seq_training true --model_type cnn --train_type base_attention | last_state | base_attention  --segment_type none | all | average
  
--stage   code running control                     
--seq_training  true | false  
--train_type last_state | base_attention | multi_attention
--model_type lstm | cnn                                                                                                                   
--segment_type none | all | average
