#!/bin/bash

DATASET=All

#STEP1
RECORD=1543


#STEP2
#RECORD=1545

#STEP3
#RECORD=0547

#STEP4
#RECORD=5547

##STEP5
#RECORD=0548

#STEP6
#RECORD=3548


IS_REMOVE_PROBLEM=0


python pre_main_short.py --best 1 --is_remove $IS_REMOVE_PROBLEM --mode val --record $RECORD --dataset_type $DATASET --patch_method STTN 

tail record/$RECORD/log.txt -n 1
