# MGN-LSTM - Learning CO2 plume migration in faulted reservoirs with Graph Neural Networks

In this work, we introduce a model architecture, [MGN-LSTM] (https://www.sciencedirect.com/science/article/pii/S0098300424001948#fig2), for solving a dynamic CO<sub>2</sub>-water multiphase flow problem in the context of carbon capture and storage (CCS). The figure below shows that schematic of MGN-LSTM, where 

<p align="center"><img src="figs/Schematic_MGN_LSTM.png" alt="structure" align="center" width="600px"></p>


## Overview

Current version contains full implmentation python code of GNN-flow and its jupyter notebook version. 
The repo at this point only contains the hexahedron dataset used for prescreening models. 
Later a PEBI dataset will be added to this repo. 

## Setup 

Before running codes, you need to first change corresponding configurable parameters in setup bash files.
Then, use the following command to build virtual environment and install dependencies on different clusters: 


     source setup_env.sh
     
Setup files will also yield a startup file (activate_mgn_lstm.sh) that can be used for loading proper module and
setting up internet proxy. Once the virtual envoirnment is properly set up, you do not need to install libs again.
Just use the generated startup file to switch to the proper environment and setup internet proxy. The command to 
use the startup file is: 

    source activate_mgn_lstm.sh
   
## Datasets

*  You can download the dataset associated with reproducing the baseline of this work from the following link:
https://drive.google.com/drive/folders/1IXCqlKnlf8tPsXrfFRyG_F-ehYK-9JZn?usp=sharing

* 

## Pretrained models

*  First, you can directly use our pretrained surrogate models with MGN-LSTM from the following link:
https://drive.google.com/drive/folders/1V5RvJxm4WiEJVZkKur2cf3nUdtKtzPPY?usp=sharing

* Use 'evaluate_mgn_lstm_gas.ipynb' to evaluate the performance of gas saturation model.

## Running training scripts
* For training a MGN-LSTM for gas saturation fields, run the following bash command
    ```bash
    bash run_lstm_gas.sh
    ```

* For training a MGN-LSTM for pressure fields, run the following bash command
     ```bash
    bash run_lstm_p.sh
    ```
  
## Using pretrained MGN_LSTM

### Download pretrained model
* The trained model parameters associated with this code can be downloaded [here](https://drive.google.com/drive/folders/1V5RvJxm4WiEJVZkKur2cf3nUdtKtzPPY?usp=sharing)


## Issues?
* If you have an issue in running the code please [raise an issue](https://github.com/IsaacJu-debug/mgn_lstm/issues)

## Citation
If you find our work useful and relevant to your research, please cite:
```
@article{ju2024learning,
        title={Learning CO2 plume migration in faulted reservoirs with Graph Neural Networks},
        author={Ju, Xin and Hamon, Fran{\c{c}}ois P and Wen, Gege and Kanfar, Rayan and Araya-Polo, Mauricio and Tchelepi, Hamdi A},
        journal={Computers \& Geosciences},
        volume={193},
        pages={105711},
        year={2024},
        publisher={Elsevier}
}
```

