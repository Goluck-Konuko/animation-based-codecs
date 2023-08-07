#!/bin/bash
MODEL="${1:-rdac}" 
python run.py --mode test --log_dir results --config config_ids/${MODEL}.yaml --checkpoint pretrained/${MODEL}/rd/rd_1.pth.tar
python run.py --mode test --log_dir results --config config_ids/${MODEL}.yaml --checkpoint pretrained/${MODEL}/rd/rd_2.pth.tar
python run.py --mode test --log_dir results --config config_ids/${MODEL}.yaml --checkpoint pretrained/${MODEL}/rd/rd_3.pth.tar
python run.py --mode test --log_dir results --config config_ids/${MODEL}.yaml --checkpoint pretrained/${MODEL}/rd/rd_4.pth.tar
python run.py --mode test --log_dir results --config config_ids/${MODEL}.yaml --checkpoint pretrained/${MODEL}/rd/rd_5.pth.tar