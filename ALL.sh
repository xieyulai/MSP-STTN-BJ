#!/bin/bash

RECORD=1002

DATASET=All
#DATASET=Sub

#EVAL_MODE=Iteration
EVAL_MODE=Epoch

#python pre_main_short.py  --mode train --record $RECORD --dataset_type $DATASET --patch_method STTN --presume_record 1700 --presume_epoch_s 78 --keep_train 1 

python pre_main_short.py --mode train --record $RECORD --dataset_type $DATASET --patch_method STTN 

python pre_main_short.py  --mode val --record $RECORD --dataset_type $DATASET --patch_method STTN --eval_mode $EVAL_MODE

tail record/$RECORD/log.txt -n 1
