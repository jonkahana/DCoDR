#!/bin/bash

False=''
True='True'
dataset=celeba_x64_train
exp_type=discriminative
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR__celeba_x64' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR__celeba_x64']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \



