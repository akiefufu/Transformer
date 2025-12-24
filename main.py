import torch
from torch import nn

from data import zidian_y, loader, zidian_xr, zidian_yr
from mask import mask_pad, mask_tril
from model import Transformer

def train(model, loss, optim, sched, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for i, (x, y) in enumerate(loader):
            # x = [8, 50]
            # y = [8, 51]
            x = x.to(device)
            y = y.to(device)

            # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
            # [8, 50, 39]
            pred = model(x, y[: , :-1])

            # [8, 50, 39] -> [400, 39]
            pred = pred.reshape(-1, 39)

            # [8, 51] -> [400]
            y_true = y[:, 1:].reshape(-1)

            # 忽略pad
            select = y_true != zidian_y['<PAD>']
            pred = pred[select, :]
            y_true = y_true[select]

            l = loss(pred, y_true)
            optim.zero_grad()
            l.backward()
            optim.step()

            total_loss += l.item()

            # [select, 39] -> [select]
            pred_label = pred.argmax(dim=1)
            total_correct += (pred_label == y_true).sum().item()
            total_tokens += y_true.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_tokens
        lr = optim.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | lr={lr:.6f} | loss={avg_loss:.4f} | acc={accuracy:.4f}")

        sched.step()

def predict(model, x):
    model.eval()

    # x = [1, 50]
    x = x.to(device)
    with torch.no_grad():
        # [1, 1, 50, 50]
        mask_pad_x = mask_pad(x)

        # 初始化输出,这个是固定值
        # [1, 50]
        # [[0,2,2,2...]]
        target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
        target = torch.LongTensor(target).unsqueeze(0).to(device)

        # x编码,添加位置信息
        # [1, 50] -> [1, 50, 32]
        x = model.embed_x(x)

        # 编码层计算,维度不变
        # [1, 50, 32] -> [1, 50, 32]
        x = model.encoder(x, mask_pad_x)

        # 遍历生成第1个词到第49个词
        for i in range(49):
            # [1, 50]
            y = target

            # [1, 1, 50, 50]
            mask_tril_y = mask_tril(y)

            # y编码,添加位置信息
            # [1, 50] -> [1, 50, 32]
            y = model.embed_y(y)

            # 解码层计算,维度不变
            # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
            y = model.decoder(x, y, mask_pad_x, mask_tril_y)

            # 全连接输出,39分类
            # [1, 50, 32] -> [1, 50, 39]
            out = model.fc_out(y)

            # 取出当前词的输出
            # [1, 50, 39] -> [1, 39]
            out = out[:, i, :]

            # 取出分类结果
            # [1, 39] -> [1]
            out = out.argmax(dim=1).detach()

            # 以当前词预测下一个词,填到结果中
            target[:, i + 1] = out

            if out == zidian_y['<EOS>']:
                break

    pad = torch.tensor([zidian_y['<PAD>']] * 50, device=device)

    target = torch.cat(
        [target.squeeze(0), pad],
        dim=0
    ).unsqueeze(0)
    target = target[:, :51]

    return target

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    num_heads = 4
    num_layers = 6  # Transformer 中的EncoderLayer和DecoderLayer的个数
    d_model = 32
    seq_len = 50
    vocab_size = 39
    num_hiddens = 64    # 输出全连接层的隐藏单元个数

    model = Transformer(num_layers, num_heads, d_model, seq_len, num_hiddens).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    num_epochs = 10

    train(model, loss, optim, sched, num_epochs)

    # 测试
    for i, (x, y) in enumerate(loader):
        break

    for i in range(8):
        print(i)
        print(''.join([zidian_xr[i] for i in x[i].tolist()]))
        print(''.join([zidian_yr[i] for i in y[i].tolist()]))
        print(''.join([zidian_yr[i] for i in predict(model, x[i].unsqueeze(0))[0].tolist()]))
