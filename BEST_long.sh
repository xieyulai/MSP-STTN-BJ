#!/bin/bash
#RECORD=2547
#RECORD=0807
RECORD=3805

DATASET=All
#DATASET=Sub

IS_REMOVE_PROBLEM=0

python pre_main_long.py --best 1 --is_remove $IS_REMOVE_PROBLEM --mode val --record $RECORD --dataset_type $DATASET --patch_method STTN 

tail record/$RECORD/log.txt -n 1
