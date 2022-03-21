#!/bin/bash



# train
bash bash_scripts/DCoDR_norec/celeba_x64/train/DCoDR_norec__celeba_x64__train.sh

# eval
bash bash_scripts/DCoDR_norec/celeba_x64/eval/DCoDR_norec__celeba_x64__eval_pred.sh

