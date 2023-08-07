#!/bin/bash
MODEL="${1:-rdac}" 
##
# dac - Animation-only framework adapted from FOMM and formulated for low-bitrate compression
# rdac - Predictive animation and residual coding
#       - End-to-end trained to animate and compress the residual difference between the animated and original frames
#       - Uses the variational autoencoder network from Image compression
#        with design features enabling the compression of sparse residuals
# rdac_t - RDAC exploiting the temporal correlations between neighbouring residual frames
#                       
#Other parameters are configured from the yaml files in <config/*> folder
#######
#PS:: REMOVE --debug flag to actually train the models.
#######
python run.py --config config_ids/${MODEL}.yaml --project_id $PROJECT_ID --log_dir cpks/${MODEL} --step $STEP --rate 5 --debug