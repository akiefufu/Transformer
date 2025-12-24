import torch
from torch import nn

from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, num_hiddens):
        super().__init__()
        self.mh = MultiHead(num_heads, d_model)
        self.fc = FullyConnectedOutput(d_model, num_hiddens)

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, num_hiddens):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(EncoderLayer(num_heads, d_model, num_hiddens))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, num_hiddens):
        super().__init__()

        self.mh1 = MultiHead(num_heads, d_model)
        self.mh2 = MultiHead(num_heads, d_model)

        self.fc = FullyConnectedOutput(d_model, num_hiddens)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.mh2(y, x, x, mask_pad_x)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)

        return y


class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, num_hiddens):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DecoderLayer(num_heads, d_model, num_hiddens))

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        for layer in self.layers:
            y = layer(x, y, mask_pad_x, mask_tril_y)
        return y


# 主模型
class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, seq_len, num_hiddens, vocab_size=39):
        super().__init__()
        self.embed_x = PositionEmbedding(seq_len, d_model)
        self.embed_y = PositionEmbedding(seq_len, d_model)
        self.encoder = Encoder(num_layers, num_heads, d_model, num_hiddens)
        self.decoder = Decoder(num_layers, num_heads, d_model, num_hiddens)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, y):
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        x, y = self.embed_x(x), self.embed_y(y)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]
        y = self.fc_out(y)

        return y