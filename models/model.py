from torch import dropout, nn
import torch
import torch.nn.functional as F
# from .TCN import TemporalConvNet
# from .transformer import Transformer


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=configs.kernel_size//2),
            nn.BatchNorm1d(64),
            Mish(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=configs.kernel_size, stride=1, bias=False, padding=configs.kernel_size//2),
            nn.BatchNorm1d(64),
            Mish(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=configs.kernel_size, stride=1, bias=False, padding=configs.kernel_size//2),
            nn.BatchNorm1d(128),
            Mish(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=configs.kernel_size, stride=1, bias=False, padding=configs.kernel_size//2),
            nn.BatchNorm1d(128),
            Mish(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(128, configs.final_out_channels, kernel_size=configs.kernel_size,
                      stride=1, bias=False, padding=configs.kernel_size//2),
            nn.BatchNorm1d(configs.final_out_channels),
            Mish(),
            nn.MaxPool1d(kernel_size=2, stride=configs.stride, padding=1),
            nn.Dropout(configs.dropout)
        )

        # self.tcn = TemporalConvNet(
        #     configs.input_channels, configs.dilations, configs.kernel_size, configs.dropout)
        # self.tst = Transformer(configs.d_input, configs.d_model, configs.d_output,
        #                        configs.q, configs.v, configs.h, configs.N,
        #                        attention_size=configs.attention_size,
        #                        dropout=configs.dropout, chunk_mode=configs.chunk_mode, pe=configs.pe)
        self.logits = nn.Linear(
            configs.features_len * configs.final_out_channels, configs.num_classes)
        # self.logits = nn.Linear(
        #     configs.d_model * configs.d_output, configs.num_classes)

    def forward(self, x_in):
        # print(x_in.shape)
        x = self.conv_block1(x_in)
        # print(x.shape, '111')
        x = self.conv_block2(x) + x
        # print(x.shape, '222')
        x = self.conv_block3(x)
        # print(x.shape, '333')
        x = self.conv_block4(x) + x
        # print(x.shape, '444')
        x = self.conv_block5(x)
        # print(x.shape, '555')
        # x = self.tcn(x_in)
        # x = self.tst(x_in)
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        # print(x_flat.shape, "------------")
        logits = self.logits(x_flat)
        # print(logits.shape, "================")
        return logits, x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x