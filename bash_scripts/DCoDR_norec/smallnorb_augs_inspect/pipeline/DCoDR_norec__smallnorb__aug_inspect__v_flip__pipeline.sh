#!/bin/bash

# train
bash bash_scripts/DCoDR_norec/smallnorb_augs_inspect/train/DCoDR_norec__smallnorb__aug_inspect__v_flip__train.sh

# eval
bash bash_scripts/DCoDR_norec/smallnorb_augs_inspect/eval/DCoDR_norec__smallnorb__aug_inspect__v_flip__eval__pred.sh

