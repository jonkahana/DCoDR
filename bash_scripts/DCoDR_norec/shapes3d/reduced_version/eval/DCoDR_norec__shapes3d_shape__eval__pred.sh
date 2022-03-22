#!/bin/bash

False=''
True='True'
dataset=shapes3d__class_shape__train
exp_type=DCoDR_norec
PATH_TO_PROJECT_DIR='cache'



python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='prediction' \
--eval-name='DCoDR_norec_shapes3d_shape_50K' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_norec_shapes3d_shape_50K']" \
--train-data-name=$dataset \
--model-train-data=shapes3d__class_shape__train__reduced_50K \
--delete-weights-folder=$False \
--chosen-epoch='last' \


python -u evaluation.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--eval-type='retrieval' \
--eval-name='DCoDR_norec_shapes3d_shape_50K' \
--evaluated-exp-names="[]" \
--root-exps="['DCoDR_norec_shapes3d_shape_50K']" \
--train-data-name=$dataset \
--model-train-data=shapes3d__class_shape__train__reduced_50K \
--retrieval-dist='faiss' \
--delete-weights-folder=$False \
--chosen-epoch='last' \
