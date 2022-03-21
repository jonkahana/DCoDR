#!/bin/bash



# train
bash bash_scripts/DCoDR/smallnorb/train/DCoDR__smallnorb__no_negatives__train.sh

# eval
bash bash_scripts/DCoDR/smallnorb/eval/DCoDR__smallnorb__no_negatives__eval__pred.sh

