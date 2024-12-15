import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
############################### Build optimizer ##############################
def build_optimizer(args, params):
    """
    args:
    params:
    """
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

############################### utility functions ##############################
def read_process_data(file_path):
    """
    train_mesh_list gives the mesh to retrive
    time_step_num gives how many timesteps for each mesh to retrive, which equals train_size + test_size
    """
    return torch.load(file_path)

from torchsummary import summary
############################### Save model details ##############################
def gnn_model_summary(model, args ):
    """

    """
    model_params_list = list(model.named_parameters())
    # saving details
    if not os.path.isdir( args.modelsummary_dir ):
        os.mkdir(args.modelsummary_dir)

    with open( os.path.join(args.modelsummary_dir, args.model_name + '.txt'), 'w') as summary:
    # Record model details
        summary.write("----------------------------------------------------------------\n")
        line_new = "{:>20}  {:>25} {:>15}\n".format("Layer.Parameter", "Param Tensor Shape", "Param #")
        summary.write(line_new)
        summary.write("----------------------------------------------------------------\n")
        for elem in model_params_list:
            p_name = elem[0]
            p_shape = list(elem[1].size())
            p_count = torch.tensor(elem[1].size()).prod().item()
            line_new = "{:>20}  {:>25} {:>15}\n".format(p_name, str(p_shape), str(p_count))
            summary.write(line_new)
        summary.write("----------------------------------------------------------------\n")
        total_params = sum([param.nelement() for param in model.parameters()])
        summary.write("Total params: {}\n".format(total_params))
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary.write("Trainable params: {}\n".format(num_trainable_params) )
        summary.write("Non-trainable params: {}".format(total_params - num_trainable_params))

#######################
def save_plots(args, losses, test_losses, velo_val_losses):

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PATH = os.path.join(args.postprocess_dir,  args.model_name + '.pdf')

    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss" + " - " + args.model)
    plt.plot(test_losses, label="test loss" + " - " + args.model)
    #if (args.save_velo_val):
    #    plt.plot(velo_val_losses, label="velocity loss" + " - " + args.model_type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()
    f.savefig(PATH, bbox_inches='tight')

def calc_rela_perm(args, comp_args, sg, sw):
    """
    args.rela_perm: rela_perm options. if rela_perm is table, coordinates and sw should be read
    sg: sg field (torch tensor)
    sw: sw field (torch tensor)
    """
    if (comp_args.is_initial == True):
        if (args.rela_perm == 'table'):
            assert args.rela_perm_dir != '', '{} mode requires perm table!'.format(args.rela_perm)

            coord_sg_filename = os.path.join(args.rela_perm_dir, 'phaseVolFraction_gas.txt')
            coord_sw_filename = os.path.join(args.rela_perm_dir, 'phaseVolFraction_water.txt')
            rela_sg_filename = os.path.join(args.rela_perm_dir, 'relPerm_gas.txt')
            rela_sw_filename = os.path.join(args.rela_perm_dir, 'relPerm_water.txt')

            comp_args.coord_sg , comp_args.rel_sg  = np.loadtxt(coord_sg_filename), np.loadtxt(rela_sg_filename)
            comp_args.coord_sw , comp_args.rel_sw  = np.loadtxt(coord_sw_filename), np.loadtxt(rela_sw_filename)

        elif (args.rela_perm == 'coery'):
            # to be implemented
            pass
        else:
            # to be implemented
            pass
        comp_args.is_initial = False

    if (args.rela_perm == 'table'):
        sg_inter = torch.tensor(np.interp(sg.cpu().detach().numpy(), comp_args.coord_sg, comp_args.rel_sg)).to(args.device)
        sw_inter = torch.tensor(np.interp(sw.cpu().detach().numpy(), comp_args.coord_sw, comp_args.rel_sw)).to(args.device)

    elif (args.rela_perm == 'coery'):
        pass

    return sg_inter, sw_inter

class ErrorMetrics(object):
    def __init__(self, var,
                  type='rela'):
        super(ErrorMetrics, self).__init__()
        self.var = var # variable to be compared
        self.type = type # type of error metrics
    
    def gas_plume_error(self, s_g, s_g_hat):
        epislon = 1e-12
        # s_g and s_g_hat are PyTorch tensors of shape (B*h*w, 1)
        indicator = ((torch.abs(s_g) > 0.01) | (torch.abs(s_g_hat) > 0.01)).float()
        sum_indicators = torch.sum(indicator)
        abs_diff = torch.abs(s_g - s_g_hat) * indicator
        
        # Calculate gas plume error
        error = torch.sum(abs_diff) / sum_indicators if sum_indicators >= epislon else 0.0
        return error

    def __call__(self, x, y):

        if self.var == 'sg' or self.var == 'sat':
            # we gonna use gas plume error
            return self.gas_plume_error(x, y)
        elif self.var == 'pressure' or self.var == 'p':
            # we gonna use relative pressure error
            pass
        else:
            # we gonna use relative velocity error
            pass

###################### return all model paramters
def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )