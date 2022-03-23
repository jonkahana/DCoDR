#!/bin/bash

False=''
True='True'
dataset=celeba_x64_train
exp_type=DCoDR_norec
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR_norec__celeba_x64' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_norec__celeba_x64']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch="['last']" \





