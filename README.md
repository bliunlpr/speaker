# speaker
```fchgbduskiygvchb```

## Tongdun speaker verification  competition : 'nlpr_fight' team work 

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
