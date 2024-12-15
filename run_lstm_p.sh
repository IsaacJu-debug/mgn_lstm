#!/bin/bash
python train.py --data_name "" --model LSTM --rela_perm "none" --edge_type "dist" --device "cuda" --var_type p --data_name "meshPEBI_p.pt" --model_name "meshPEBI_p_trained"