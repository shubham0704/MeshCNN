import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        # print(f'out channels -> {out_channels}')
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        # pdb.set_trace()
        # print(x.shape) # torch.Size([1, 5, 16])
        x = x.squeeze(-1)
        # print(x.shape) # torch.Size([1, 5, 16])
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        x = self.conv(G)
        return x

    def flatten_gemm_inds(self, Gi):
        '''
        notes:
        This function ensures that edge ids are unique across batches
        when we load 2 meshes they can have same edge ids
        when doing batch processing we need to ensure that edge ids
        are unique across batch hence we add extra index across batch
        '''
        (b, ne, nn) = Gi.shape # 1, 16, 5
        ne += 1 # ne = 17
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        # b * ne  = 17
        
        '''
        batch_n = [[0, 0, ..., 0]
                    ]
                    1 x 17
        if more batches
        batch_n =  [[0, 0, ..., 0]
                    ],
                [1, 1, ..., 1]
                    ]
                    2 x 17
        '''
        add_fac = batch_n * ne
        '''
        add_fac = [[0, 0, ..., 0]
                    ]
                    1 x 17
        if more batches
        add_fac =  [[0, 0, ..., 0]
                    ],
                [17, 17, ..., 17]
                    ]
                    2 x 17
        '''
        add_fac = add_fac.view(b, ne, 1) # 1 x 17 x 1
        '''
        add_fac = [[[0], [0], ..., [0]]
                    ]
                    1 x 17
        if more batches
        add_fac =  [[[0], [0], ..., [0]]
                    ],
                [[17], [17], ..., [17]]
                    ]
                    2 x 17

        '''
        add_fac = add_fac.repeat(1, 1, nn) # repeat 1 across batch 1 across num edges 5 times across num features
        '''
        add_fac = [[[0, 0, 0, 0, 0], [0,...], ..., [0,...]]
                    ]
                    1 x 17 x 5
        if more batches
        add_fac =  [[[0, 0, 0, 0, 0], [0,...], ..., [0,...]]
                    ],
                [[17, 17, 17, 17, 17], [17,...], ..., [17,...]]
                    ]
                    2 x 17 x 5
                    
        '''
        # add_fac = [1, 17, 5]
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :] # Gi + add_fac[0:1, 1:17, 0:5]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        # pdb.set_trace()
        # print('entering create gemm')
        # print(x) # torch.Size([1, 5, 16]) 5 dimension features for each edge
        # print(Gi)
        Gishape = Gi.shape # torch.Size([1, 16, 5])
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift
        # print(x)
        
        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        # print('flattening Gi')
        # print(Gi_flat)
        odim = x.shape # torch.Size([1, 5, 17])
        x = x.permute(0, 2, 1).contiguous() # torch.Size([1, 5, 17])
        # print(x) # torch.Size([1, 17, 5])
        x = x.view(odim[0] * odim[2], odim[1])
        # print(x) # torch.Size([17, 5])
        f = torch.index_select(x, dim=0, index=Gi_flat) # torch.Size([1, 16, 5, 5])
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1) # 1, 16, 5, 5 # for each node 5 neighbors each having 5 features
        f = f.permute(0, 3, 1, 2)
        # f -> torch.Size([1, 5, 16, 5]) # last dimension has neighbor_id
        
        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # [[1.,  2., 12., 13.],  [2.,  0., 14.,  3.]]->  [[ 0.,  1.,  2., 12., 13.],[ 1.,  2.,  0., 14.,  3.]]
        # xsz is the number of edges you want to pad each mesh to, in our example its 16
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm # torch.Size([16, 5])
