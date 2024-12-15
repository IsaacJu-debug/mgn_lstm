import torch
print(torch.__version__)

import os
import argparse
import random

import numpy as np
import mesh_model
import stats
import uti_func
from uti_func import ErrorMetrics, count_model_params

import wandb
from torch_geometric.loader import DataLoader
from tqdm import trange
import copy
import time
def time_it():
  return time.time_ns() / (10 ** 9) # convert to floating-point seconds

############################### Training/testing loops ##############################

"""Training functions"""
def train(dataset, device, stats_list, args, comp_args, PATH=None):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''
    #Define the model name for saving
    if (args.model_name == ''):
        if (args.seed_list == ''):
            # default values
            args.model_name= 'model'+ args.model +'_var'+ args.var_type + '_node_based' + str(args.node_based) + \
                    args.data_type + '_edge'+ args.edge_type + \
                '_relPerm' + args.rela_perm + \
                    '_skip'+ str(args.skip)  + '_roll_num' + str(args.rollout_num) +\
                   '_well_weight' + str(args.well_weight) +\
                   '_model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)
        else:
            # tunning random seed
            args.model_name= 'model'+ args.model+'_seed' + str(args.seed) +'_var'+ args.var_type + \
                  '_node_based' + str(args.node_based) + args.data_type + '_edge'+ args.edge_type + \
                '_relPerm' + args.rela_perm + \
                    '_skip'+ str(args.skip)  + '_roll_num' + str(args.rollout_num) +\
                   '_well_weight' + str(args.well_weight) +\
                   '_model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)


    wandb_name = args.model_name
    wandb.init(mode="disabled") # turn off wandb logging
    #wandb.init(project=args.project_name, entity=args.wandb_usr, name=wandb_name)
    wandb.config.update(args)

    #args.anim_name = model_name
    #torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(dataset[:args.train_size],
                            batch_size=args.batch_size, shuffle=args.shuffle) # each LSTM takes mesh_sizes * timestep
    test_loader = DataLoader(dataset[args.train_size:args.train_size+ args.test_size],
                            batch_size=args.test_batch_size, shuffle=args.shuffle)

    #The statistics of the data are decomposed
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 1 # the dynamic variables have the shape of 1 (saturation )
    gas_error = ErrorMetrics(args.var_type) # error metrics for gas saturation
    
    # Load the pretrained feature extractor
    if (args.use_rnn):
        # Recurrent_MGN
        mgn_model = mesh_model.MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(args.device)
        if (args.pre_trained):
            # load pre-trained model
            assert PATH is not None, "Pre-trained model is not given!"
            mgn_model.load_state_dict(torch.load(PATH, map_location=args.device))

        #gnn_model_summary(mgn_model, args, 'mgn_pretrained')
        model = mesh_model.TransferTempoMGN( mgn_model, args.hidden_dim, num_classes,
                                args).to(device)

    else:
        # MGN
        model = mesh_model.MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                        args).to(args.device)

    scheduler, opt = uti_func.build_optimizer(args, model.parameters())
    #Show the model parameters
    uti_func.gnn_model_summary(model, args)
    
    model_params = count_model_params(model)
    print('Model parameter num {}'.format(model_params))
    wandb.run.summary["model_params"] = model_params

    # train
    losses = []
    test_losses = []
    velo_val_losses = []
    best_test_loss = np.inf
    best_val_rmse_loss = np.inf
    best_model = None

    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        cost = 0

        for batch in loader:
            #Note that normalization must be done before it's called. The stats.unnormalized
            #data needs to be preserved in order to correctly calculate the loss
            batch=batch.to(device)
            cost = 0
            roll_out_loss = 0

            for num in range(args.rollout_num):
                if (num == 0 and args.use_rnn):
                    h_0 = torch.zeros(batch.x.shape[0], args.hidden_dim).to(device)
                    c_0 = torch.zeros(batch.x.shape[0], args.hidden_dim).to(device)

                if (args.use_rnn):
                    #print('h_0 {} c_0 {}'.format(h_0.device, c_0.device))
                    pred, h_0, c_0 = model(batch, mean_vec_x,std_vec_x,mean_vec_edge,
                                            std_vec_edge, h_0, c_0)
                    loss = model.loss(pred,batch,mean_vec_y,std_vec_y, num)
                else:
                    if (args.noise):
                        # Injecting noise is only used for one-step model
                        # perturb (input, output) pairs with a zero-mean gaussian distribution
                        # current verison adopts a hard-coded noise_scale (0.003), used in deepmind
                        zero_size = torch.zeros(batch.x[:, 0].size(), dtype=torch.float32)
                        noise = torch.normal(zero_size, std=args.noise_scale)
                        # saturation
                        batch.x[:, 0] += noise
                        batch.y[:, num] += noise

                    pred = model(batch, mean_vec_x,std_vec_x,mean_vec_edge, std_vec_edge)
                    loss = model.loss(pred,batch,mean_vec_y,std_vec_y, num)

                if (args.rollout_num > 1):
                    # For rollout larger than 1, namely previous state needs to attend next state
                    batch.x[:, 0] = stats.unnormalize( pred.squeeze(), mean_vec_y[num], std_vec_y[num] )
                    if (args.rela_perm != 'none'):
                        #print('sg mean {}'.format(torch.mean(batch_tmp.x[:, 0])))
                        gs_rela_perm, _ = uti_func.calc_rela_perm(args, comp_args, batch.x[:, 0], 1. - batch.x[:, 0])  # calculate gs rela perm
                        #print('min {} max {}'.format(torch.min(gs_rela_perm), torch.max(gs_rela_perm)))
                        #print(gs_rela_perm.shape)
                        #print(torch.mean(gs_rela_perm))
                        batch.x[:, -1] = gs_rela_perm              # update cell-wise rela perm
                        if (args.debug_log > 0):
                            # print update gs rela perm at well location
                            well_mask = torch.argmax(batch.x[:,4:9],dim=1)==torch.tensor(mesh_model.NodeType.WELL)
                            print('Rollout time {}'.format(num))
                            print('gs y at well: \n{}'.format(batch.y[well_mask, num]))
                            print('sg at well: \n{}'.format(batch.x[well_mask, 0]))
                            print('sg rela at well: \n{}'.format(batch.x[well_mask, -1]))


                cost = cost + args.loss_weight_list[num] * loss
                roll_out_loss += cost.item()
                num += 1

            roll_out_loss /= num
            cost.backward()         #backpropagate loss
            opt.step()
            opt.zero_grad()         #zero gradients each time
            total_loss += roll_out_loss
            num_loops += 1

            if (args.use_rnn):
                del h_0
                del c_0

        total_loss /= num_loops
        losses.append(total_loss)
        
        #Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % 10 == 0:
            if (args.save_velo_val):
                # save saturation evaluation
                test_loss, velo_val_rmse = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y, args.save_velo_val, loss_weight_list =args.loss_weight_list)
                velo_val_losses.append(velo_val_rmse.item())

                wandb.log({"test_loss": test_loss.item(),
                           "{}_val_loss".format(args.var_type): velo_val_rmse.item()})
            else:
                test_loss, _ = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y, args.save_velo_val, error_metric=gas_error,
                                loss_weight_list =args.loss_weight_list)

            test_losses.append(test_loss.item())

            # saving model
            if not os.path.isdir( args.checkpoint_dir ):
                os.mkdir(args.checkpoint_dir)

            PATH = os.path.join(args.checkpoint_dir, args.model_name+'.csv')
            #df.to_csv(PATH,index=False)

            #save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_val_rmse_loss = velo_val_rmse
                best_model = copy.deepcopy(model)

                wandb.run.summary["best_test_loss"] = best_test_loss
                wandb.run.summary["best_{}_error".format(args.var_type)] = best_val_rmse_loss
                
        else:
            #If not the tenth epoch, append the previously calculated loss to the
            #list in order to be able to plot it on the same plot as the training losses
            if (args.save_velo_val):
              test_losses.append(test_losses[-1])
              velo_val_losses.append(velo_val_losses[-1])

        if(epoch%args.output_freq==0):
            if (args.save_velo_val):
                print("train loss", str(round(total_loss, 5)),
                      "test loss", str(round(test_loss.item(), 5)),
                      "{} val loss {}".format( args.var_type, str(round(velo_val_rmse.item(), 5))))
            else:
                print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))

            if(args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, args.model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )

    wandb.finish()  # Finish the wandb run
    return test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader

"""Testing functions"""
def test(loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y, is_validation, error_metric=None,
          delta_t=0.01, save_model_preds=False, model_type=None, loss_weight_list=None):

    '''
    Calculates test set losses and validation set errors.
    '''
    loss_test =0
    velo_rmse = 0
    num_loops=0

    for data in loader:
        data=data.to(device)
        with torch.no_grad():
            cost = 0
            roll_out_loss = 0
            velo_rmse_rollout = 0
            for num in range(args.rollout_num):

                if (num == 0 and args.use_rnn):
                    h_0 = torch.zeros(data.x.shape[0], args.hidden_dim).to(device)
                    c_0 = torch.zeros(data.x.shape[0], args.hidden_dim).to(device)

                if (args.use_rnn):
                    #print('h_0 {} c_0 {}'.format(h_0.device, c_0.device))
                    pred, h_0, c_0 = test_model(data, mean_vec_x,std_vec_x,
                                                mean_vec_edge,std_vec_edge, h_0, c_0)
                else:
                    pred = test_model(data, mean_vec_x,std_vec_x,mean_vec_edge,
                                   std_vec_edge)

                if (args.rollout_num > 1):
                    # For rollout larger than 1, namely previous state needs to attend next state
                    data.x[:, 0] = stats.unnormalize( pred.squeeze(), mean_vec_y[num], std_vec_y[num] )
                    if (args.rela_perm.lower() != 'none'):
                        #print('sg mean {}'.format(torch.mean(batch_tmp.x[:, 0])))
                        gs_rela_perm, _ = uti_func.calc_rela_perm(args, comp_args, data.x[:, 0], 1. - data.x[:, 0])  # calculate gs rela perm
                        #print(gs_rela_perm.shape)
                        #print(torch.mean(gs_rela_perm))
                        data.x[:, -1] = gs_rela_perm              # update cell-wise rela perm

                loss = test_model.loss(pred, data, mean_vec_y, std_vec_y, num)
                cost = cost + loss_weight_list[num] * loss # total loss, later being back-propagated
                # Implement a multi-step loss
                roll_out_loss += cost

                if (is_validation):
                    #Like for the MeshGraphNets model, calculate the mask over which we calculate
                    #flow loss and add this calculated RMSE value to our val error
                    # pred gives normalized saturation increment
                    eval_velo =  stats.unnormalize( pred.squeeze(), mean_vec_y[num], std_vec_y[num] )
                    gs_velo = data.y[:, num].squeeze()
                    if error_metric:
                        velo_rmse_rollout += error_metric(eval_velo, gs_velo)
                    else:
                        #error = torch.sum((eval_velo - gs_velo) ** 2, axis = -1)
                        error =  (eval_velo - gs_velo)** 2
                        velo_rmse_rollout += torch.sqrt( torch.mean(error) )

                num += 1

            roll_out_loss /= num
            velo_rmse_rollout /= num
            if (args.use_rnn):
                del h_0
                del c_0

        loss_test += roll_out_loss
        velo_rmse += velo_rmse_rollout

        num_loops+=1
        # if velocity is evaluated, return velo_rmse as 0
    return loss_test/num_loops, velo_rmse/num_loops

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

############################### Setup hyperparameters ##############################
def main(args, comp_args):

    # find the dataset folder
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'datasets')
    checkpoint_dir = os.path.join(root_dir, 'best_models')
    postprocess_dir = os.path.join(root_dir, 'animations')
    modelsummary_dir = os.path.join(root_dir, 'model_details')

    # weight loss list for multiple MGN or LSTM
    if (args.loss_weight_list == ''):
        # no specific loss weights is given
        args.loss_weight_list = np.linspace(1.0, 1.0, args.rollout_num)
    else:
        args.loss_weight_list = np.array([int(item) for item in args.loss_weight_list.split(',')])

    print('loss_weight_list: {}'.format(args.loss_weight_list))

    if (args.model.upper() == 'LSTM'):
        # rollout number * multistep size
        total_step = args.rollout_num * args.step
    else:
        # mgn: single step rollout
        total_step = args.rollout_num * args.total_ts

    args.total_ts = total_step

    if (args.data_name != ''):
        file_path = os.path.join(dataset_dir, args.data_name)
        #stats_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_ms.pt')
    else:
        args.data_name = 'mesh{}_data{}_var{}_model{}_totalTs{}_skip{}_multistep{}_{}edge_{}label_{}relPerm.pt'.format(
                                                                                                          args.data_type,
                                                                                                          args.hete_type,
                                                                                                          args.var_type,
                                                                                                            args.model,
                                                                                                          args.total_ts,
                                                                                                          args.skip,
                                                                                                          args.step,
                                                                                                          args.edge_type,
                                                                                                          args.label_type,
                                                                                                          args.rela_perm)

        print('No input data is given. Using input parameters to find the following file \n{}'.format(args.data_name))

        #raise NotImplementedError("Unknown data")
        file_path = os.path.join(dataset_dir, args.data_name)
    dataset = uti_func.read_process_data(file_path)

    if (args.use_rnn == True):
	      assert args.model.upper() == 'LSTM', "model {} is wrongly given!".format(args.model)
    else:
	      assert args.model.upper() == 'MGN', "model {} is wrongly given!".format(args.model)

    ## TODO: CHECK PERFORMANCE OF STAT CHANGES BY ITERATING THROUGH ALL DATASETS AND CHECKING
    ##       THE MEAN AND VAR OF NORMALIZED DATA
    # check the availability of GPU
    device = args.device if torch.cuda.is_available() else 'cpu'
    #args.device = device
    print('Getting {}...'.format(device))

    args.device = "cpu" # This is necessary for running get_stats function below
    stats_list = stats.get_stats(dataset, args, comp_args)
    print('stats_list: \n{}'.format(stats_list))
    args.device = device

    # Start the training
    test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader = train(dataset, device,
                                                                                          stats_list, args, comp_args)

    print("Min test set loss: {0}".format(min(test_losses)))
    print("Minimum loss: {0}".format(min(losses)))
    if (args.save_velo_val):
        print("Minimum saturation validation loss: {0}".format(min(velo_val_losses)))

    # Run test for our best model to save the predictions!
    #test(test_loader, best_model, is_validation=False, save_model_preds=True, model_type=model)
    #print()
    uti_func.save_plots(args, losses, test_losses, velo_val_losses)

if __name__ == '__main__':
    # Input arguments
    # convert string into bool for argparser to work
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help='', default='LSTM') # LSTM; MGN
    argparser.add_argument('--use_rnn', type=str2bool, help='', default=True) # LSTM; MGN
    argparser.add_argument('--loss_type', type=str, help='', default='rmse') # RMSE; L2 not implemented yet
    argparser.add_argument('--device', type=str, help='', default='cuda') # cuda could vary from cuda:0 to cuda:3 depending on how many avaialble GPUs
    argparser.add_argument('--noise', type=str2bool, help='', default=False) # inject noise for one-step predictions
    argparser.add_argument('--noise_scale', type=float, help='', default=0.003) # noise scale
    # wandb related
    argparser.add_argument('--wandb_usr', type=str, help='Weights & Biases user.')
    argparser.add_argument('--project_name', type=str, help='', default='mgn_lstm') # wandb project name

    # data name related
    argparser.add_argument('--data_name', type=str, help='', default='', required = True)
    argparser.add_argument('--data_type', type=str, help='', default='PEBI')
    argparser.add_argument('--hete_type', type=str, help='', default='hete')
    argparser.add_argument('--var_type', type=str, help='', default='sat')
    argparser.add_argument('--rollout_num', type =int, help = '', default = 11)
    argparser.add_argument('--step', type=int, help='', default=1)  # Multistep number. This only makes sense when rollout is True.
                                                                          # Multistep training trick for LSTM at each training step
    argparser.add_argument('--total_ts', type =int, help = '', default = 11) # LSTM: args.rollout_num * args.step
                                                                            # MGN: args.rollout_num * args.total_ts
    argparser.add_argument('--label_type', type=str, help='', default='y') # dy incremet; y state
    argparser.add_argument('--edge_type', type=str, help='', default='dist') # dist and trans
    argparser.add_argument('--rela_perm', type=str, help='', default='table', required=True) # none means no rela_perm is calculated; look-up table; coery-brook equations
    argparser.add_argument('--skip', type=int, help='', default=5) # skip 1 means 10 days. skip 5 means 50 days
    argparser.add_argument('--node_based', type=str2bool, help='', default=False) # if the MGN is node-based, the edge-based message will not be computed
    # Training related
    argparser.add_argument('--node_type_index', type=int, help='', default=3) # the starting index of node type, used for locating speical nodes
                                                                               # current version: sat, perm, volume, [type0, type1, type2, type3], (rela_perm)
    argparser.add_argument('--batch_size', type=int, help='', default=10)
    argparser.add_argument('--test_batch_size', type=int, help='', default=5)
    argparser.add_argument('--num_layers', type=int, help='', default=10)
    argparser.add_argument('--hidden_dim', type=int, help='', default=100)
    argparser.add_argument('--epochs', type=int, help='', default=500)
    argparser.add_argument('--opt', type=str, help='', default='adam')
    argparser.add_argument('--opt_scheduler', type=str, help='', default='none')
    argparser.add_argument('--opt_restart', type=int, help='', default=0)
    argparser.add_argument('--weight_decay', type=float, help='', default=5e-4)
    argparser.add_argument('--lr', type=float, help='', default=0.001)
    argparser.add_argument('--loss_weight_list', help='delimited list input',
                            type=str, default = '') #
    argparser.add_argument('--debug_log', type=int, help='', default=0) # output debug log: 0 stands for no debug info; 1 for rela info; 2 input; output shape;
    argparser.add_argument('--well_weight', type=float, help='', default=0.700)
    argparser.add_argument('--train_size', type=int, help='', default=450)
    argparser.add_argument('--test_size', type=int, help='', default=50)

    # temporal model related
    argparser.add_argument('--need_edge_weight', type=str2bool, help='', default=False)
    argparser.add_argument('--lstm_filter_size', type=int, help='', default=8)
    argparser.add_argument('--normalized', type=str2bool, help='', default=True) # flag for using normalized input and output
    argparser.add_argument('--pre_trained', type=str2bool, help='', default=False) # Import the trained meshgraphnet one-step model as a feature extractor
    argparser.add_argument('--shuffle', type=str2bool, help='', default=False)

    # inspection output related
    argparser.add_argument('--model_name', type=str, help='', default='' )
    argparser.add_argument('--save_velo_val', type=str2bool, help='', default=True)
    argparser.add_argument('--save_best_model', type=str2bool, help='', default=True)
    argparser.add_argument('--output_freq', type=int, help='', default=50) # every epoch losses will be printted out for inspection

    # directories-related
    argparser.add_argument('--modelsummary_dir', type=str, help='', default='./model_details/')
    argparser.add_argument('--checkpoint_dir', type=str, help='', default='./best_models/')
    argparser.add_argument('--postprocess_dir', type=str, help='', default='./2d_loss_plots/')
    argparser.add_argument('--rela_perm_dir', type=str, help='', default='./tables/')

    # random seed
    argparser.add_argument('--seed_list', help='delimited list input',
                            type=str, default = '') #
    argparser.add_argument('--seed', type=int, help='', default=5)

    args = argparser.parse_args()

    # computational arguments
    for c_args in [
            {'is_initial':True,
            'coord_sg':'',
             'coord_sw':'',
             'rel_sg':'',
             'rel_sw':'', },
        ]:
            comp_args = objectview(c_args)

    if (args.seed_list == ''):
        # no specific seed_list is given
        # no need for tuning
        args.seed_list = [args.seed]
    else:
        args.seed_list = [int(item) for item in args.seed_list.split(',')]

    for seed in args.seed_list:
        # setup random seed
        args.seed = seed
        print('Current seed: {}'.format(args.seed))

        torch.manual_seed(args.seed)  #Torch
        random.seed(args.seed)        #Python
        np.random.seed(args.seed)     #NumPy
        t0 = time_it()
        main(args, comp_args)
        t1 = time_it()
        print('Took {} hrs to finish the case with seed {}'.format(np.abs(t1 - t0)/3600.0, args.seed))
        args.loss_weight_list = ''
        args.model_name = ''
