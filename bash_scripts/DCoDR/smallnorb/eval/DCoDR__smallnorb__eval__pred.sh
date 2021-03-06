#!/bin/bash

False=''
True='True'
dataset=smallnorb_train
exp_type=discriminative
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR_smallnorb' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_smallnorb']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \


python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='retrieval' \
--eval-name='DCoDR_smallnorb' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_smallnorb']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \



