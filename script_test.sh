#!/bin/bash
MODEL="${1:-dac}" 

##BASELINE CODECS [HEVC, VVC, VVENC]
if [ "$MODEL" = "baselines" ]; then
python run.py --mode test --log_dir results --config config/${MODEL}.yaml
fi

##DAC
if [ "$MODEL" = "dac" ]; then
python run.py --mode test --log_dir results --config config/${MODEL}.yaml --checkpoint checkpoints/dac.pt
fi

#HDAC
if [ "$MODEL" = "hdac" ]; then
python run.py --mode test --log_dir results --config config/${MODEL}.yaml --checkpoint checkpoints/hdac.pt
fi


#RDAC
if [ "$MODEL" = "rdac" ]; then
python run.py --mode test --log_dir results --config config/${MODEL}.yaml --checkpoint checkpoints/rdac.pt
fi



