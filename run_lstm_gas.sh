#!/bin/bash
python train.py --data_name "" --model LSTM --rela_perm "none" --edge_type "dist" --device "cuda" --var_type sat --data_name "meshPEBI_sg.pt" --model_name "meshPEBI_sg_trained"