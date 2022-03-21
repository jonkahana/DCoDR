#!/bin/bash

False=''
True='True'
dataset=smallnorb_train
exp_type=DCoDR_norec
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR_norec__smallnorb__aug_inspect_high_satur' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_norec__smallnorb__aug_inspect_high_satur']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \


