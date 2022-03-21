#!/bin/bash

False=''
True='True'
dataset=cars3d_train
exp_type=DCoDR_norec
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR_norec_cars3d' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_norec_cars3d']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \


python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='retrieval' \
--eval-name='DCoDR_norec_cars3d' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_norec_cars3d']" \
--train-data-name=$dataset \
--delete-weights-folder=$False \
--chosen-epoch='last' \


