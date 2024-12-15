# GNN-Flow

Relevant Papers: Meshgraphnet [arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409); Graph ConvLSTM [arxiv.org/abs/1612.07659] (https://arxiv.org/abs/1612.07659)


## Overview

Current version contains full implmentation python code of GNN-flow and its jupyter notebook version. 
The repo at this point only contains the hexahedron dataset used for prescreening models. 
Later a PEBI dataset will be added to this repo. 

## Setup 

Before running codes, you need to first change corresponding configurable parameters in setup bash files.
Then, use the following command to build virtual environment and install dependencies on different clusters: 

csevolta: 

    source setup_env_volta.sh

cypress:

     source setup_env_cypress.sh
     
Setup files will also yield a startup file (start_env_cypress.sh) that can be used for loading proper module and
setting up internet proxy. Once the virtual envoirnment is properly set up, you do not need to install libs again.
Just use the generated startup file to switch to the proper environment and setup internet proxy. The command to 
use the startup file is: 

    source start_env_cypress.sh

Note that cypress machine, equipped with A100, compiled with a new version of NVIDIA-SMI, requires a newer combination
torch-1.11.0+cu113. 
   
## Datasets

All datasets are placed in the corresponding project folder (GNN+flow) under gpfs directory. The 
complete absolute path where you can find a dataset is as follow:

Hexahedron: 

    cd /data/gpfs/Projects/CSE_HPML/GNN+flow/00_dataset/01_hexa
    
Pebi (in development):


## Examples
The following examples are all based on an interactive bash session with single GPU. Multi-GPU jobs are not configured yet. 

Python script:

    cd /folder/where/you/put/the/scripts
    
    python ./train.py --num_layers 12 --hidden_dim 20 --train_size 40 --test_size 10 > ./main_out/main_000.out

Notebook: 

    cd /folder/where/you/put/the/notebook
    
    jupyter notebook --no-browser --ip 0.0.0.0 --port 8880 --NotebookApp.token=''
    

 


