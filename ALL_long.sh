#!/bin/bash
#RECORD=2547
RECORD=1003

DATASET=All
#DATASET=Sub

#python pre_main_long.py --mode train --record $RECORD --dataset_type $DATASET --patch_method STTN --presume_record 4 --presume_epoch_s 27 --keep_train 1 

python pre_main_long.py --mode train --record $RECORD --dataset_type $DATASET --patch_method STTN 

python pre_main_long.py --mode val --record $RECORD --dataset_type $DATASET --patch_method STTN 

tail record/$RECORD/log.txt -n 1
