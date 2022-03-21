#!/bin/bash

False=''
True='True'
dataset=celeba_x64_train
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
--exp-name='DCoDR__celeba_x64' \
--data-name=$dataset \
--batch-size=64 \
--epochs=200 \
--content-dim=128 \
--class-dim=256 \
--use-pretrain=$True \
\
--enc-arch='moco_resnet50' \
\
$exp_type \
--use-fc-head=$False \
\
--tau=0.2 \
--num-pos=1 \
--num-rand-negs=64 \
--class-negs=$True \
--num-b-cls-samp=32 \
--num-b-cls=4 \
\
--use-class=$True \
--gen-arch='lord' \
--gen-lr=0.001 \
--reconstruction-decay=0.3 \
--use-adv-loss=$False \
\
--shifting-key='tau' \
--shifting-args="[0.2]" \
