import torch
import torch.nn as nn

import utils
from models.layers import Attention, Mlp
from models.conditions import TimestepEmbedder, CategoricalEmbedder, ClusterContinuousEmbedder
import numpy as np

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Denoiser(nn.Module):
    def __init__(
        self,
        max_n_nodes,
        hidden_size=384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.1,
        Xdim=118,
        Edim=5,
        ydim=3,
        task_type='regression',
        graph_type = "uniform",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.x_embedder = nn.Linear(Xdim + max_n_nodes * Edim, hidden_size, bias=False)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedding_list = torch.nn.ModuleList()
        self.graph_type = graph_type
        
        self.shared_null_emb = nn.Embedding(1, hidden_size)
        self.y_embedding_list.append(ClusterContinuousEmbedder(2, hidden_size, drop_condition, self.shared_null_emb))
        for i in range(ydim - 2):
            if task_type == 'regression':
                self.y_embedding_list.append(ClusterContinuousEmbedder(1, hidden_size, drop_condition, self.shared_null_emb))
            else:
                self.y_embedding_list.append(CategoricalEmbedder(2, hidden_size, drop_condition, self.shared_null_emb))


        self.encoders = nn.ModuleList(
            [
                SELayer(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.out_layer = OutLayer(
            max_n_nodes=max_n_nodes,
            hidden_size=hidden_size,
            atom_type=Xdim,
            bond_type=Edim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.encoders :
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.out_layer.adaLN_modulation[0], 0)

    def forward(self, x, e, node_mask, y, t, index_x, index_e, unconditioned, condition_index, train=False):
        
        force_drop_id = torch.zeros_like(y.sum(-1))
        force_drop_id[torch.isnan(y.sum(-1))] = 1
        if unconditioned:
            force_drop_id = torch.ones_like(y[:, 0])
        
        x_in, e_in, y_in = x, e, y
        bs, n, _ = x.size()
        x = torch.cat([x, e.reshape(bs, n, -1)], dim=-1)
        x = self.x_embedder(x)

        c1 = self.t_embedder(t)

        c2_list = []
        for i in range(1, self.ydim):
            if i == 1:
                c2_list.append(self.y_embedding_list[i-1](y[:, :2], self.training, force_drop_id, t)) 
            else:
                c2_list.append(self.y_embedding_list[i-1](y[:, i:i+1], self.training, force_drop_id, t)) 
        # (cond_num, bs, hidden_size)
        c2_tensor = torch.stack(c2_list, dim=0)
        # (bs, cond_num, hidden_size)
        c2_tensor = c2_tensor.permute(1, 0, 2)  
        if train:
            # [bs,cond_num,hidden]
            selected_embs = torch.gather(c2_tensor, dim=1, index=condition_index.unsqueeze(-1).expand(-1, -1, c2_tensor.size(2)))  

            # [bs]
            # k = torch.randint(1, self.ydim, (bs,), device=c2_tensor.device)
            probs = torch.tensor([0.4, 0.0, 0.0, 0.6], device=c2_tensor.device) 
            probs = probs / probs.sum() 
            k_idx = torch.multinomial(probs, num_samples=bs, replacement=True)
            k = k_idx + 1

            # [bs,cond_num]
            idxs = torch.arange(self.ydim-1, device=c2_tensor.device).unsqueeze(0).expand(bs, -1)
            mask = idxs < k.unsqueeze(1) 

            # [bs,cond_num,1]
            mask_f = mask.unsqueeze(-1).float()
            # [bs, hidden_size]
            sum_embs = (selected_embs * mask_f).sum(dim=1)
            c2 = sum_embs / k.unsqueeze(1).float()
        else: 
            # (bs, 1, hidden_size)
            c2 = torch.gather(c2_tensor, dim=1, index=condition_index.unsqueeze(-1).expand(-1, 1, c2_tensor.size(2)))
            # (bs, hidden_size)
            c2 = c2.squeeze(1)
        c = c1 + c2
        
        for i, block in enumerate(self.encoders):
            x = block(x, c, node_mask)

        # X: B * N * dx, E: B * N * N * de
        X, E, y = self.out_layer(x, x_in, e_in, c, t, node_mask)

        #following the final output process of SEDD 
        if self.graph_type == "absorb":
            # Not yet configured for Uniform
            esigm1_log = torch.where(t < 0.5, torch.expm1(t), t.exp() - 1).log().to(X.dtype)[:, None, None]
            X = X - esigm1_log - np.log(X.shape[-1] - 1)# this will be approximately averaged at 0
            E = E - esigm1_log[:, :, None] - np.log(E.shape[-1] - 1)
        X = torch.scatter(X, -1, index_x.unsqueeze(-1), torch.zeros_like(X[..., :1]))
        E = torch.scatter(E, -1, index_e.unsqueeze(-1), torch.zeros_like(E[..., :1]))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class SELayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.dropout = 0.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, **block_kwargs
        )

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=self.dropout,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c, node_mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * modulate(self.norm1(self.attn(x, node_mask=node_mask)), shift_msa, scale_msa)
        x = x + gate_mlp.unsqueeze(1) * modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x


class OutLayer(nn.Module):
    # Structure Output Layer
    def __init__(self, max_n_nodes, hidden_size, atom_type, bond_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        final_size = atom_type + max_n_nodes * bond_type
        self.xedecoder = Mlp(in_features=hidden_size, 
                            out_features=final_size, drop=0)

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )


    def forward(self, x, x_in, e_in, c, t, node_mask):
        x_all = self.xedecoder(x)
        B, N, D = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        
        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out

        bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)
        bond_out = e_in + bond_out

        ##### standardize adj_out
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))

        return atom_out, bond_out, None
