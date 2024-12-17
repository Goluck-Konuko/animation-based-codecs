#!/bin/bash
MODEL="${1:-dac}" 

python run.py --mode test --log_dir results --config test_config/${MODEL}.yaml 



