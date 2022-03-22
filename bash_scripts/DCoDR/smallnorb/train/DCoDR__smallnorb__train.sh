#!/bin/bash

False=''
True='True'
dataset=smallnorb_train
exp_type=DCoDR_multi_arg
PATH_TO_PROJECT_DIR='cache'




python -u main.py \
--base-dir=$PATH_TO_PROJECT_DIR \
--cuda=$True \
--num-workers=4 \
\
--load-weights=$False \
--load-weights-exp='debug' \
--load-weights-epoch='last' \
\
--exp-name='DCoDR_smallnorb' \
--data-name=$dataset \
--test-data-name='smallnorb_test' \
\
--batch-size=64 \
--epochs=200 \
--content-dim=128 \
--class-dim=256 \
--use-pretrain=$True \
\
--enc-arch='moco_resnet50' \
\
$exp_type \
--use-fc-head=$True \
\
--tau=0.2 \
--num-pos=1 \
--num-rand-negs=64 \
--class-negs=$True \
--num-b-cls-samp=32 \
--num-b-cls=4 \
\
--gen-arch='lord' \
--gen-lr=0.0003 \
--reconstruction-decay=0.3 \
\
--shifting-key='tau' \
--shifting-args="[0.2]" \
