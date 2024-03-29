import torch
import torch.nn as nn
import numpy as np


from .attention import Seq_Transformer


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        # self.num_channels = configs.input_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(
            configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2,
                      configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(
            patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)
        # self.seq_transformer = RTransformer(d_model=self.num_channels, rnn_type='GRU', ksize=6, n_level=3, n=1, h=4, dropout=configs.dropout)

    def forward(self, features_aug1, features_aug2):
        # print(features_aug1, features_aug2)
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        # randomly pick time stamps
        # print(seq_len, self.timestep)
        # 生成0到（序列长度-timestep）的随机数
        t_samples = torch.randint(
            seq_len - self.timestep, size=(1,)).long().to(self.device)
        nce = 0  # average over timestep and batch
        # (timestep, batch, 特征长度) 第一维度每个变量是要预测的下一个时间点
        encode_samples = torch.empty(
            (self.timestep, batch, self.num_channels)).float().to(self.device)
        # for i in np.arange(1, self.timestep + 1):
        #     encode_samples[i - 1] = z_aug2[:, t_samples +
        #                                    i, :].view(batch, self.num_channels)
        for i in np.arange(1, self.timestep + 1):
            # print(z_aug2[:, t_samples + i, :].shape, '+++++++++++++')
            encode_samples[i - 1] = z_aug2[:, t_samples +
                                           i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]
        # print(forward_seq.shape)
        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty(
            (self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
