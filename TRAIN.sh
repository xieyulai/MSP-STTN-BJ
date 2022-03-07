#!/bin/bash

RECORD=0710

DATASET=All
#DATASET=Sub

#EVAL_MODE=Iteration
EVAL_MODE=Epoch

# TRAINING
#python pre_main_short.py  --mode train --record $RECORD --dataset_type $DATASET  --presume_record 1700 --presume_epoch_s 78 --keep_train 1 

python pre_main_short.py --mode train --record $RECORD --dataset_type $DATASET 

# TESTING
python pre_main_short.py  --mode val --record $RECORD --dataset_type $DATASET  --eval_mode $EVAL_MODE

tail record/$RECORD/log.txt -n 1
