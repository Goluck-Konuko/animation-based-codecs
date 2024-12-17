#!/bin/bash
MODEL="${1:-dac}" 
##
# dac - Animation-only framework adapted from FOMM and formulated for low-bitrate compression
# hdac - Animation + enhancement base layer encoded with HEVC/VVC
# hdac_hf - Animation + enhancement base layer encoded with HEVC/VVC with High Frequency Shuttling
# rdac - Predictive animation and residual coding
#       - End-to-end trained to animate and compress the residual difference between the animated and original frames
#       - Uses the variational autoencoder network from Image compression
#        with design features enabling the compression of sparse residuals
#      - Can be trained to exploit temporal correlations between neighbouring frame residual
# mrdac - MUlti reference animation-based coding with contrastive learning and weighted feature aggregation                       
#Other parameters are configured from the yaml files in <config/*> folder
#######
#PS:: REMOVE --debug flag to actually train the models.
#######
python run.py --config train_config/${MODEL}.yaml --log_dir cpks/${MODEL} --debug