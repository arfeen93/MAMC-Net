#!/bin/bash

ZSLDG=('clipart' 'infograph' 'painting' 'quickdraw' 'sketch')
DATA='/home/arfeen/dataset/'
#for target in ${ZSLDG[@]} ;
python ../main.py --zsl --dg --dataset_name domainnet --target sketch --config_file ../configs/zsl+dg/sketch.json --data_root $DATA --name sketch-zsldg
#python -m torch.distributed.launch --nproc_per_node=3 ../main.py --zsl --dg --target quickdraw --config_file ../configs/zsl+dg/quickdraw.json --data_root $DATA --name quickdraw-zsldg
