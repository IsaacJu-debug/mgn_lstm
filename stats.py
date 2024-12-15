import torch
import uti_func
import mesh_model

def normalize(to_normalize,mean_vec,std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats(data_list, args, comp_args,
              use_single_dist = True):
    '''
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    '''

    #mean and std of the node features are calculated
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])

    #mean and std of the edge features are calculated
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])

    #mean and std of the output parameters are calculated
    if (use_single_dist):
        mean_vec_y=torch.zeros([1])
        std_vec_y=torch.zeros([1])
    else:
        mean_vec_y=torch.zeros(data_list[0].y.shape[1:])
        std_vec_y=torch.zeros(data_list[0].y.shape[1:])

    if (args.rela_perm == ''):
        # not use rela perm as node feature
        pass
    else:
        # use rela perm as a node feature
        mean_vec_rel_gs =torch.zeros([1])
        std_vec_rel_gs =torch.zeros([1])
        num_accs_rel_gs = 0
    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 10**6
    iter = 0
    #Define a very small value for normalizing to
    eps=torch.tensor(1e-16)
    zeros = torch.tensor(0.0)
    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge=0
    num_accs_y=0
    var_scale = 1.0
    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        #Add to the
        mean_vec_x+=torch.sum(dp.x,dim=0)/var_scale
        std_vec_x+=torch.sum(dp.x**2,dim=0)/var_scale
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]
        #print(args.rela_perm)
        if (args.rela_perm == 'none'):
            pass
        else:
            #print(torch.unique(dp.y[:, :]))
            gs_rela_perm, _ = uti_func.calc_rela_perm(args, comp_args, dp.y, 1. - dp.y)
            if (args.debug_log > 1):
                well_mask = torch.argmax(dp.x[:,4:9],dim=1)==torch.tensor(mesh_model.NodeType.WELL)
                print('label y at the well \n{} '.format(dp.y[well_mask]))
                print('rela gs at the well \n{} '.format(gs_rela_perm[well_mask]))

            #print(torch.unique(gs_rela_perm[:,:]))
            mean_vec_rel_gs +=torch.sum(gs_rela_perm)
            std_vec_rel_gs +=torch.sum(gs_rela_perm**2)
            #print(mean_vec_y)
            num_accs_rel_gs +=  dp.y.shape[1] * dp.y.shape[0]

        if (use_single_dist):
            mean_vec_y+=torch.sum(dp.y)/var_scale
            std_vec_y+=torch.sum(dp.y**2)/var_scale
            #print(mean_vec_y)
            num_accs_y+=  dp.y.shape[1] * dp.y.shape[0]
        else:
            mean_vec_y+=torch.sum(dp.y, dim = 0)
            std_vec_y+=torch.sum(dp.y**2, dim = 0)
            num_accs_y+=dp.y.shape[0]


        if(num_accs_x>max_accumulations or num_accs_edge>max_accumulations or num_accs_y>max_accumulations):
            print('Read {} samples'.format(iter))
            print('num_accs_x {} num_accs_edge {} num_accs_y {}'.format(num_accs_x,num_accs_edge,num_accs_y))
            break
        iter += 1

    mean_vec_x = mean_vec_x/num_accs_x
    var_vec_x = torch.maximum((std_vec_x/num_accs_x - mean_vec_x**2), zeros)
    std_vec_x = torch.maximum(torch.sqrt(var_vec_x),eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)

    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2),eps)

    if (args.rela_perm  == ''):
        pass
    else:
        mean_vec_rel_gs = mean_vec_rel_gs/num_accs_rel_gs
        std_vec_rel_gs = torch.maximum(torch.sqrt(std_vec_rel_gs/num_accs_rel_gs - mean_vec_rel_gs**2),eps)

    if (use_single_dist ):
        #print(mean_vec_x.shape)
        #print(mean_vec_y.shape)
        mean_vec_x[0] = mean_vec_y
        std_vec_x[0] = std_vec_y
        mean_vec_y = torch.ones(dp.y.shape[1] ) * mean_vec_y
        std_vec_y = torch.ones(dp.y.shape[1] ) * std_vec_y
        if (args.rela_perm != 'none'):
            mean_vec_x[-1] = mean_vec_rel_gs
            std_vec_x[-1] = std_vec_rel_gs
    
    #print('assign {} to porosity std'.format(1.7263e-04))
    #std_vec_x[2] = torch.tensor(1.7263e-04) 
    mean_std_list=[mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y]
    print('Read {} samples'.format(iter))
    
    return mean_std_list
