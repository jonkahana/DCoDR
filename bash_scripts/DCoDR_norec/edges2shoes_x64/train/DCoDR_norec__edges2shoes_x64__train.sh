#!/bin/bash

False=''
True='True'
dataset=edges2shoes_x64_train
exp_type=DCoDR_norec_multi_arg
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
--exp-name='DCoDR_norec_edges2shoes_x64' \
--data-name=$dataset \
\
--batch-size=128 \
--epochs=200 \
--content-dim=128 \
--class-dim=256 \
--use-pretrain=$True \
\
--enc-arch='moco_resnet50' \
\
--fn_start_c=64 \
--fn_out_c=32 \
--fn_res_norm='instance' \
--fn_res_blocks=5 \
--fn_res_n_down=4 \
\
$exp_type \
--use-fc-head=$False \
--class-dependent-loss=$False \
--cls-depend-freeze-pos-encoder=$True \
--class-dependent-loss-start=200 \
\
--tau=0.1 \
--num-pos=1 \
--num-rand-negs=64 \
--class-negs=$True \
--num-b-cls-samp=32 \
--num-b-cls=4 \
--shifting-key='tau' \
--shifting-args="[0.1]" \
