import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads *5

        # assert (
        #     self.head_dim * heads == embed_size
        # ), "Embedding size needs to be divisible by heads"

        # 独立的线性层用于生成键、查询和值
        self.keys = nn.Linear(embed_size, embed_size*5)
        self.queries = nn.Linear(embed_size, embed_size*5)
        self.values = nn.Linear(embed_size, embed_size*5)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x, mask=None):
        N = x.shape[0]
        seq_length = x.shape[1]

        # 生成键、查询和值
        keys = self.keys(x).view(N, seq_length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, seq_length, self.heads, self.head_dim)
        values = self.values(x).view(N, seq_length, self.heads, self.head_dim)

        # 计算注意力分数
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # 可选：应用掩码
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=-1)

        # 应用注意力权重到值上
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, seq_length, self.heads * self.head_dim
        )

        # 线性层输出
        out = self.fc_out(out)

        return out, attention

if __name__ == "__main__":

    # 示例使用：
    embed_size = 1
    heads = 1
    batch_size = 1
    seq_length = 10

    # 随机生成输入数据
    x = torch.rand(batch_size, seq_length, embed_size)
    mask = None  # 可以提供掩码来忽略某些位置

    # 初始化自注意力层
    self_attention = SelfAttention(embed_size, heads)

    # 前向传播
    output, attention_weights = self_attention(x, mask)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)
    # print(x, attention_weights)