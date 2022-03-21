#!/bin/bash




# train
bash bash_scripts/DCoDR_norec/smallnorb_augs_inspect/train/DCoDR_norec__smallnorb__aug_inspect__random_erasing__train.sh

# eval
bash bash_scripts/DCoDR_norec/smallnorb_augs_inspect/eval/DCoDR_norec__smallnorb__aug_inspect__random_erasing__eval__pred.sh

