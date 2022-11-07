"""
All of those code are copied from https://github.com/CCSI-Toolset/MGN 
thanks to them !

This is a GraphNetworks

"""
from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum

class GNN(Module):
    #default values based on MeshGraphNets paper/supplement
    def __init__(self, 
            #data attributes:
            in_dim_node, #includes data window, node type, inlet velocity 
            in_dim_edge, #distance and relative coordinates
            out_dim, #includes x-velocity, y-velocity, volume fraction, pressure (or a subset)
            out_dim_node=128, out_dim_edge=128, 
            hidden_dim_node=128, hidden_dim_edge=128,
            hidden_layers_node=2, hidden_layers_edge=2,
            #graph processor attributes:
            mp_iterations=5,
            hidden_dim_processor_node=128, hidden_dim_processor_edge=128, 
            hidden_layers_processor_node=2, hidden_layers_processor_edge=2,
            mlp_norm_type='LayerNorm',
            #decoder attributes:
            hidden_dim_decoder=128, hidden_layers_decoder=2,
            output_type='acceleration',
            out_index = [0],
            delta_t = 0.005,
            #other:
            **kwargs):

        super(GNN, self).__init__()

        # batch norm 1D for normalizing edge input
        self.edge_normalizer = nn.BatchNorm1d(in_dim_edge)

        self.node_encoder = MLP(in_dim_node, out_dim_node, 
            hidden_dim_node, hidden_layers_node, 
            mlp_norm_type)
        self.edge_encoder = MLP(in_dim_edge, out_dim_edge, 
            hidden_dim_edge, hidden_layers_edge, 
            mlp_norm_type)
        self.graph_processor = GraphProcessor(mp_iterations, out_dim_node, out_dim_edge,
            hidden_dim_processor_node, hidden_dim_processor_edge, 
            hidden_layers_processor_node, hidden_layers_processor_edge,
            mlp_norm_type)
        self.node_decoder = MLP(out_dim_node, out_dim, hidden_dim_decoder, hidden_layers_decoder, None)
        self.output_type = output_type

        self.out_index = out_index
        self.delta_t = delta_t

    def forward(self, graph):

        # normalize edge input
        edge_attr = self.edge_normalizer(graph.edge_attr)

        out = self.node_encoder(graph.x)

        edge_attr = self.edge_encoder(edge_attr)
        out, _ = self.graph_processor(out, graph.edge_index, edge_attr)
        out = self.node_decoder(out) 

        return out * self.delta_t + graph.x[:, self.out_index] 

class MLP(nn.Module):
    #MLP with LayerNorm
    def __init__(self, 
            in_dim, 
            out_dim=128, 
            hidden_dim=128,
            hidden_layers=2,
            norm_type='LayerNorm'):

        '''
        MLP
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert (norm_type in ['LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm'])
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class EdgeProcessor(nn.Module):

    def __init__(self, 
            in_dim_node=128, in_dim_edge=128,
            hidden_dim=128, 
            hidden_layers=2,
            norm_type='LayerNorm'):

        '''
        Edge processor
        
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(2 * in_dim_node + in_dim_edge, 
            in_dim_edge, 
            hidden_dim,
            hidden_layers,
            norm_type)

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = cat([src, dest, edge_attr], -1) #concatenate source node, destination node, and edge embeddings
        out = self.edge_mlp(out)
        out += edge_attr #residual connection

        return out

class NodeProcessor(nn.Module):
    def __init__(self, 
            in_dim_node=128, in_dim_edge=128,
            hidden_dim=128, 
            hidden_layers=2,
            norm_type='LayerNorm'):

        '''
        Node processor
        
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(in_dim_node + in_dim_edge,  
            in_dim_node,
            hidden_dim,
            hidden_layers,
            norm_type)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        row, col = edge_index

        nb_node = x.shape[0]

        out = scatter_sum(edge_attr, col, dim=0, dim_size=nb_node) #aggregate edge message by target
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x #residual connection

        return out

def build_graph_processor_block(in_dim_node=128, in_dim_edge=128,
        hidden_dim_node=128, hidden_dim_edge=128, 
        hidden_layers_node=2, hidden_layers_edge=2,
        norm_type='LayerNorm'):

    '''
    Builds a graph processor block
    
    in_dim_node: input node feature dimension
    in_dim_edge: input edge feature dimension
    hidden_dim_node: number of nodes in a hidden layer for graph node processing
    hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
    hidden_layers_node: number of hidden layers for graph node processing
    hidden_layers_edge: number of hidden layers for graph edge processing
    norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    '''

    return MetaLayer(
            edge_model=EdgeProcessor(in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type),
            node_model=NodeProcessor(in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type)
        )

class GraphProcessor(nn.Module):
    def __init__(self, 
        mp_iterations=15, 
        in_dim_node=128, in_dim_edge=128,
        hidden_dim_node=128, hidden_dim_edge=128, 
        hidden_layers_node=2, hidden_layers_edge=2,
        norm_type=None):

        '''
        Graph processor
        mp_iterations: number of message-passing iterations (graph processor blocks)
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(GraphProcessor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(build_graph_processor_block(in_dim_node, in_dim_edge,
                hidden_dim_node, hidden_dim_edge, 
                hidden_layers_node, hidden_layers_edge, 
                norm_type))

    def forward(self, x, edge_index, edge_attr):
        for block in self.blocks:
            x, edge_attr, _ = block(x, edge_index, edge_attr)

        return x, edge_attr      
