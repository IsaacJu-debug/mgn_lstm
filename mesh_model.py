import torch

#import torch_geometric
#import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional

from torch.nn import Parameter, Linear, Sequential, LayerNorm, ReLU
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric_temporal.nn.recurrent import GConvLSTM
import enum
import stats

class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study:
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    WELL = 1
    FAULT = 2
    BOUNDARY = 3
    SIZE = 4


""" GCN-based model"""
""" Modified from https://github.com/locuslab/cfd-gcn/blob/master/models.py"""
class MeshGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, improved=False,
                 cached=False, bias=True, fine_marker_dict=None):
        super().__init__()
        self.sdf = None
        in_channels += 1  # account for sdf

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            convs.append(GCNConv(channels[i], channels[i+1], improved=improved,
                                 cached=cached, bias=bias))
        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x

""" Original Meshgraphnet model"""
class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge,
                 hidden_dim, output_dim, args,
                 emb=False):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type (node_position is encoded in edge attributes)
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """
        self.device = args.device
        self.well_weight = args.well_weight
        self.data_type = args.data_type
        self.num_layers = args.num_layers
        self.node_type_index = args.node_type_index
        self.node_based = args.node_based
        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))
        if not self.node_based:
            self.edge_encoder = Sequential(Linear( input_dim_edge , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )


        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim, node_based=self.node_based))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )


    def build_processor_model(self):
        return ProcessorLayer


    def forward(self,data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = stats.normalize(x,mean_vec_x,std_vec_x)
        edge_attr=stats.normalize(edge_attr,mean_vec_edge,std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x,edge_attr = self.processor[i](x,edge_index,edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)

    def loss(self, pred, inputs,mean_vec_y,std_vec_y, num):
        #Define the node types that we calculate loss for
        #Get the loss mask for the nodes of the types we calculate loss for
        #Need more delibrations
        if (self.data_type.upper() == 'HEXA'):
            well_loss_mask = (torch.argmax(inputs.x[:,1:],dim=1)==torch.tensor(0)) # extra weight (well)
            normal_loss_mask = (torch.argmax(inputs.x[:,1:],dim=1)==torch.tensor(1))

        if (self.data_type.upper() == 'PEBI'):
            # Hard-coded index for node type
            well_loss_mask = torch.logical_or((torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.WELL)),
                                             (torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.FAULT))) # extra weight (well)
            normal_loss_mask = torch.logical_or((torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.NORMAL)),
                                                (torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.BOUNDARY)))

        #Normalize labels with dataset statistics.
        labels = stats.normalize(inputs.y[:, num],mean_vec_y[num],std_vec_y[num]).unsqueeze(-1)

        #Find sum of square errors
        error=torch.sum((labels-pred)**2,axis=1)

        #Root and mean the errors for the nodes we calculate loss for
        loss=torch.sqrt(torch.mean(error[normal_loss_mask])) + \
        self.well_weight * torch.sqrt(torch.mean(error[well_loss_mask]))
        #loss=torch.sqrt(torch.mean(error))

        return loss

"""Recurrent MGN model"""

class TransferTempoMGN(torch.nn.Module):
    def __init__(self, mgn_model, hidden_dim, output_dim, args, emb=False):
        super(TransferTempoMGN, self).__init__()
        """
        input: mgn_model: a pretrained Meshgraphnet
        """
        # initialize FeatureExtractor class, which has a complete forward function and returns
        # the last layer of processor
        self.device = args.device
        self.well_weight = args.well_weight
        self.data_type = args.data_type
        self.num_layers = args.num_layers
        self.node_type_index = args.node_type_index
        self.need_edge_weight = args.need_edge_weight
        self.node_based = args.node_based
        #self.feature_extractor = mgn_model
        self.feature_extractor = nn.ModuleList(mgn_model.children())[:-1]

        if (args.pre_trained):
            self.decoder = nn.ModuleList(mgn_model.children())[-1]
            for param in mgn_model.parameters():
                param.requires_grad = False
        else:
            # Fine-tuned a decoder. certainly we can the pre-trained one too
            self.decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )

        # Stack a consLSTM model after the last layer of processor is finished
        self.lstm_filter_size = args.lstm_filter_size
        self.recurrent_model = GConvLSTM(hidden_dim, hidden_dim, self.lstm_filter_size)

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self,data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge, h_0, c_0):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors

        h_0: hidden state from previous timestep
        c_0: cell state from previous timestep
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = stats.normalize(x,mean_vec_x,std_vec_x)
        edge_attr=stats.normalize(edge_attr,mean_vec_edge,std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.feature_extractor[0](x) # output shape is the specified hidden dimension
        if not self.node_based:
            edge_attr = self.feature_extractor[1](edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            if not self.node_based:
                x, edge_attr = self.feature_extractor[2][i](x,edge_index,edge_attr)
            else:
                x, _ = self.feature_extractor[1][i](x,edge_index,edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        # step 3: feed the propagated node embeddings into convLSTM
        if (self.need_edge_weight):
            edge_weight = edge_attr
        else:
            edge_weight = torch.ones( edge_attr.shape[0] ).to(self.device)

        h_new, c_new = self.recurrent_model(x, edge_index, edge_weight, h_0, c_0)
        # step 4: decode latent node embeddings into physical quantities of interest

        # step 5: return hidden state and cell state
        return self.decoder(h_new), h_new, c_new

    def loss(self, pred, inputs,mean_vec_y,std_vec_y, num):
        #Define the node types that we calculate loss for

        #Get the loss mask for the nodes of the types we calculate loss for
        #Need more delibrations
        if (self.data_type.upper() == 'HEXA'):
            well_loss_mask = (torch.argmax(inputs.x[:,1:],dim=1)==torch.tensor(0)) # extra weight (well)
            normal_loss_mask = (torch.argmax(inputs.x[:,1:],dim=1)==torch.tensor(1))

        if (self.data_type.upper() == 'PEBI'):
            # Hard-coded index for node type
            well_loss_mask = torch.logical_or((torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.WELL)),
                                             (torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.FAULT))) # extra weight (well)
            normal_loss_mask = torch.logical_or((torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.NORMAL)),
                                                (torch.argmax(inputs.x[:,self.node_type_index:self.node_type_index + NodeType.SIZE],dim=1)==torch.tensor(NodeType.BOUNDARY)))

        #stats.normalize labels with dataset statistics.
        labels = stats.normalize(inputs.y[:, num],mean_vec_y[num],std_vec_y[num]).unsqueeze(-1)

        #Find sum of square errors
        error=torch.sum((labels-pred)**2,axis=1)

        #Root and mean the errors for the nodes we calculate loss for
        loss=torch.sqrt(torch.mean(error[normal_loss_mask])) + \
        self.well_weight * torch.sqrt(torch.mean(error[well_loss_mask]))
        #loss=torch.sqrt(torch.mean(error))

        return loss

"""ProcessorLayer inherits from the PyG MessagePassing base class and handles processor/GNN part of the architecture. 👇

## ProcessorLayer Class: Edge Message Passing, Aggregation, and Updating

## Edge and Node MLP
"""

class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 node_based=False, agg_method='sum', **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.node_based = node_based
        self.agg_method = agg_method

        if not node_based:
            self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        if hasattr(self, 'edge_mlp'):
            self.edge_mlp[0].reset_parameters()
            self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        if not self.node_based:
            #print('edge_index {}'.format(edge_index.shape))
            #print(edge_index)
            out, updated_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
            updated_nodes = torch.cat([x, out], dim=1)
            updated_nodes = x + self.node_mlp(updated_nodes)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
            #print('x shape {}; out shape {}'.format(x.shape, out.shape))
            updated_nodes = torch.cat([x, out], dim=1) # residual connection
            #print('updated_nodes shape {}'.format(updated_nodes.shape))
            updated_nodes = x + self.node_mlp(updated_nodes)
            updated_edges = None

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """
        if not self.node_based:
            updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
            return self.edge_mlp(updated_edges)+edge_attr
        else:
            return x_j # return the raw embeddings of targe nodes

    def aggregate(self, inputs, index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """
        # The axis along which to index number of nodes.
        node_dim = 0
        #print('inputs {} index {}'.format(inputs.shape, index.shape))
        #print(index)
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce = self.agg_method)
        if not self.node_based:
            return out, inputs
        else:
            #print('inputs shape {}'.format(inputs.shape))
            #print('out shape {}'.format(out.shape))
            return out
