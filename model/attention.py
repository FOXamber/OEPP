# @Author:Yilu Wu
# @Time:2024/2/26
# @Description:
import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads,T,max_seq_len=10):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.T=T

        self.state_encoder = nn.Sequential(
            nn.Linear(6 * input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.ReLU()
        )

        # self.state_encoder = nn.Sequential(
        #     nn.Linear(2*input_dim, input_dim),
        #     nn.ReLU()
        # )

        self.state_linears = nn.ModuleList([
            nn.Linear(2*input_dim,hidden_dim) for _ in range(T)
        ])

        # self.in_linear = nn.Linear(input_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, 768)

        # 多层自注意力机制
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # 前馈全连接层
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 1 * hidden_dim),
                nn.ReLU(),
                nn.Linear(1 * hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])

        # Layer Normalization
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.attention_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.positional_encoding = nn.Parameter(self.generate_positional_encoding(max_seq_len, hidden_dim),
                                                requires_grad=False)

    def generate_positional_encoding(self, max_seq_len, hidden_dim):
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, inputs):
        # inputs: B * T * D
        #input: B*(2*D）
        # batch_size = inputs.size()
        x = self.state_encoder(inputs) #B*D
        x_2 = x
        x = torch.zeros(inputs.shape[0],self.T,self.input_dim).cuda()
        for i in range(self.T):
            x[:,i,:] = self.state_linears[i](x_2)
        #B*T*D
        x = x + self.positional_encoding[:self.T, :].to(x.device)

        for layer in range(self.num_layers):
            x_2 = x
            # Self-Attention
            x = x.permute(1, 0, 2)  # T * B * D
            x, _ = self.self_attention_layers[layer](x, x, x)
            x = x.permute(1, 0, 2)  # B * T * D

            # Add & Norm
            x = self.attention_norm_layers[layer](x_2 + x)

            # Feed-Forward
            x_2 = x
            x = self.feed_forward_layers[layer](x)

            # Add & Norm
            x = self.ffn_norm_layers[layer](x_2 + x)

        x = self.out_linear(x)
        output_list = []
        for i in range(self.T):
            text_embedding = x[:,i,:]
            output_list.append(text_embedding)
        return output_list
