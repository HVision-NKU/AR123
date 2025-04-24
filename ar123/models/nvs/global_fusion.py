import torch
import torch.nn as nn

class DualLSTMGlobalFusion(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim

        self.lstmcell1 = nn.LSTMCell(input_size=embed_dim,
                                hidden_size=embed_dim)

        self.lstmcell2 = nn.LSTMCell(input_size=embed_dim,
                                hidden_size=embed_dim)
        
        self.mlp = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)


    def forward(self, global_embeds, ramp=None):
        '''
        global_embeds: (bs, 2k-1, 1, 1024)
        encoder_hidden_states: (77, 1024)
        ramp: (77, 1)
        '''
        h1, c1 = self.init_hidden(global_embeds.shape[0], device=global_embeds.device)
        h2, c2 = self.init_hidden(global_embeds.shape[0], device=global_embeds.device)

        x_list1 = [global_embeds[:, 0, ...]] + [global_embeds[:, idx, ...] for idx in range(1, global_embeds.shape[1], 2)]
        x_list2 = [global_embeds[:, 0, ...]] + [global_embeds[:, idx, ...] for idx in range(2, global_embeds.shape[1], 2)]

        for x_t in x_list1:
            x_t = x_t.squeeze(-2)
            h1, c1 = self.lstmcell1(x_t, [h1, c1])
        h1 = h1.unsqueeze(-2)

        for x_t in x_list2:
            x_t = x_t.squeeze(-2)
            h2, c2 = self.lstmcell2(x_t, [h2, c2])
        h2 = h2.unsqueeze(-2)

        # updated_global_embeds = (h1 + h2) / 2
        updated_global_embeds = self.mlp(torch.cat([h1, h2], dim=-1))
        if ramp is None:
            return updated_global_embeds
        else:
            return updated_global_embeds * ramp


    def init_hidden(self, batch_size, device='cuda:0'):
        return (torch.zeros(batch_size, self.embed_dim).to(device),
                torch.zeros(batch_size, self.embed_dim).to(device))   

    

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image




# class DualLSTMGlobalFusion(nn.Module):
#     def __init__(self, embed_dim=1024):
#         super().__init__()
#         self.embed_dim = embed_dim

#         self.lstmcell1 = nn.LSTMCell(input_size=embed_dim,
#                                 hidden_size=embed_dim)

#         self.lstmcell2 = nn.LSTMCell(input_size=embed_dim,
#                                 hidden_size=embed_dim)
        
#         self.mlp = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)


#     def forward(self, global_embeds, ramp=None):
#         '''
#         global_embeds: (bs, 2k-1, 1, 1024)
#         encoder_hidden_states: (77, 1024)
#         ramp: (77, 1)
#         '''
#         h1, c1 = self.init_hidden(global_embeds.shape[0], device=global_embeds.device)
#         h2, c2 = self.init_hidden(global_embeds.shape[0], device=global_embeds.device)

#         x_list1 = [global_embeds[:, 0, ...]] + [global_embeds[:, idx, ...] for idx in range(1, global_embeds.shape[1], 2)]
#         x_list2 = [global_embeds[:, 0, ...]] + [global_embeds[:, idx, ...] for idx in range(2, global_embeds.shape[1], 2)]

#         for x_t in x_list1:
#             x_t = x_t.squeeze(-2)
#             h1, c1 = self.lstmcell1(x_t, [h1, c1])
#         h1 = h1.unsqueeze(-2)

#         for x_t in x_list2:
#             x_t = x_t.squeeze(-2)
#             h2, c2 = self.lstmcell2(x_t, [h2, c2])
#         h2 = h2.unsqueeze(-2)

#         # updated_global_embeds = (h1 + h2) / 2
#         updated_global_embeds = self.mlp(torch.cat([h1, h2], dim=-1))
#         if ramp is None:
#             return updated_global_embeds
#         else:
#             return updated_global_embeds * ramp


#     def init_hidden(self, batch_size, device='cuda:0'):
#         return (torch.zeros(batch_size, self.embed_dim).to(device),
#                 torch.zeros(batch_size, self.embed_dim).to(device))   

    

# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
#         super().__init__()
#         if use_residual:
#             assert in_dim == out_dim
#         self.layernorm = nn.LayerNorm(in_dim)
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, out_dim)
#         self.use_residual = use_residual
#         self.act_fn = nn.GELU()

#     def forward(self, x):
#         residual = x
#         x = self.layernorm(x)
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.fc2(x)
#         if self.use_residual:
#             x = x + residual
#         return x
