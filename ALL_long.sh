#!/bin/bash
#RECORD=2547
RECORD=1003

DATASET=All
#DATASET=Sub

#python pre_main_long.py --mode train --record $RECORD --dataset_type $DATASET  --presume_record 4 --presume_epoch_s 27 --keep_train 1 

# TRAINING
python pre_main_long.py --mode train --record $RECORD --dataset_type $DATASET  

# TESTING
python pre_main_long.py --mode val --record $RECORD --dataset_type $DATASET 

tail record/$RECORD/log.txt -n 1
