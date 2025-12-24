import math

import torch
from torch import nn


# 注意力计算函数
def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 50, 8]

    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    b, h, t, d = Q.shape

    # 除以每个头维数的平方根,做数值缩放
    score /= d ** 0.5

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(b, t, h*d)

    return score


# 多头注意力计算层
class MultiHead(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.fc_Q = nn.Linear(d_model, d_model)
        self.fc_K = nn.Linear(d_model, d_model)
        self.fc_V = nn.Linear(d_model, d_model)

        self.out_fc = nn.Linear(d_model, d_model)

        # 规范化之后,均值是0,标准差是1
        # BN是取不同样本做归一化
        # LN是取不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
        """
        [[[-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047]],

        [[ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761]]]"""

        # norm = torch.nn.LayerNorm(normalized_shape=4, elementwise_affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
        """
        [[[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]],

        [[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]]]"""

        self.norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=True)

        self.dropout = nn.Dropout(p=0.1)
        self.num_heads = num_heads

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V = [b, 50, 32]
        b, t, d = Q.shape
        h = self.num_heads

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, t, h, d // h).permute(0, 2, 1, 3)
        K = K.reshape(b, t, h, d // h).permute(0, 2, 1, 3)
        V = V.reshape(b, t, h, d // h).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score


# 位置编码层
class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, vocab_size=39):
        super().__init__()
        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(seq_len, d_model)
        for i in range(seq_len):
            for j in range(d_model):
                pe[i, j] = get_pe(i, j, d_model)
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]
        embed = self.embed(x)

        # 词编码和位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed = embed + self.pe
        return embed


# 全连接输出层
class FullyConnectedOutput(nn.Module):
    def __init__(self, d_model, num_hiddens):
        super().__init__()
        self.fc = torch.nn.Sequential(
            nn.Linear(in_features=d_model, out_features=num_hiddens),
            nn.ReLU(),
            nn.Linear(in_features=num_hiddens, out_features=d_model),
            nn.Dropout(p=0.1),
        )

        self.norm = nn.LayerNorm(normalized_shape=d_model,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out