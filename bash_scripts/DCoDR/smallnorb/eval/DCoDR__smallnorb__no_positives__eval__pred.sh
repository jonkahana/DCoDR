#!/bin/bash

False=''
True='True'
dataset=smallnorb_train
exp_type=discriminative
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR__smallnorb__no_positives' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR__smallnorb__no_positives']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \





