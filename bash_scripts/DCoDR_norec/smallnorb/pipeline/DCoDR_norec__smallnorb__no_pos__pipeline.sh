#!/bin/bash



# train
bash bash_scripts/DCoDR_norec/smallnorb/train/DCoDR_norec__smallnorb__no_pos__train.sh

# eval
bash bash_scripts/DCoDR_norec/smallnorb/eval/DCoDR_norec__smallnorb__no_pos__eval__pred.sh

