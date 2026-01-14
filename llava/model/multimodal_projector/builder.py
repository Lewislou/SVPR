import torch
import torch.nn as nn
import re
import torch.nn.functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class QueryFormer(nn.Module):
    def __init__(self, input_dim, output_length, num_heads):
        super(QueryFormer, self).__init__()
        self.query = nn.Parameter(torch.randn(1, output_length, input_dim))  # 可学习的查询向量
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim),
        )

    def forward(self, x):
        """
        x: 输入特征, shape (B, N, C)
        returns: 固定长度的输出, shape (B, M, C)
        """
        B, N, C = x.size()

        # 将查询向量扩展到批次大小
        query = self.query.expand(B, -1, -1)  # shape (B, M, C)

        # 交换批次和序列维度以适配多头注意力模块
        query = query.transpose(0, 1)  # shape (M, B, C)
        
        # 自注意力
        self_attn_output, _ = self.self_attn(query, query, query)  # shape (M, B, C)
        self_attn_output = self.norm1(self_attn_output + query)  # 残差连接和归一化

        # 交叉注意力
        x = x.transpose(0, 1)  # shape (N, B, C)
        cross_attn_output, _ = self.cross_attn(self_attn_output, x, x)  # shape (M, B, C)
        cross_attn_output = self.norm2(cross_attn_output + self_attn_output)  # 残差连接和归一化

        # MLP层
        output = self.norm3(self.mlp(cross_attn_output) + cross_attn_output)  # shape (M, B, C)
        output = output.transpose(0, 1)  # shape (B, M, C)

        return output
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        query_atten = QueryFormer(input_dim=768, output_length=128, num_heads=8)
        modules = [query_atten,nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
