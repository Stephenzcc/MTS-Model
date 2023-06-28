from torch import dropout, nn
import torch
import torch.nn.functional as F

from .TCN import TemporalBlock
# from .transformer import Transformer



# My
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.configs = configs

        self.norm = nn.BatchNorm1d(configs.input_channels),
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=2, stride=1, dilation=1, bias=False, padding=(0)),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 32, kernel_size=configs.kernel_size, stride=configs.stride, dilation=1, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.Dropout(configs.dropout)
        )


        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2, stride=1, dilation=3, bias=False, padding=(1)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
            nn.BatchNorm1d(64),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=2, stride=1, dilation=5, bias=False, padding=(2)),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
            nn.BatchNorm1d(128),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, configs.final_out_channels, kernel_size=2, stride=1, dilation=5, bias=False, padding=(2)),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),

            nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=2, stride=1, dilation=1, bias=False, padding=(1)),
            nn.BatchNorm1d(configs.final_out_channels),
        )

        self.relu_pool = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.res_layer1 = nn.Conv1d(configs.input_channels, 32, kernel_size=1, stride=configs.stride, padding=0)
        self.res_layer2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        self.res_layer3 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
        self.res_layer4 = nn.Conv1d(128, configs.final_out_channels, kernel_size=1, stride=1)

        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose1d(configs.final_out_channels, 128, kernel_size=2, stride=1, dilation=1, bias=False, padding=(1)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=2, stride=1, dilation=5, bias=False, padding=(2)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, dilation=1, bias=False, padding=(0)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=2, stride=1, dilation=5, bias=False, padding=(2)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2, dilation=1, bias=False, padding=(0)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=2, stride=1, dilation=3, bias=False, padding=(1)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2, dilation=1, bias=False, padding=(0)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose1d(32, configs.input_channels, kernel_size=configs.kernel_size, stride=configs.stride, dilation=1, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(configs.input_channels),
            nn.ReLU(),
            nn.Linear((configs.features_len * 8 - 1) * configs.stride - 2 * configs.kernel_size//2 + configs.kernel_size, configs.seq_length),
            nn.ReLU(),
        )
        

        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        if x_in.shape[1] != self.configs.input_channels:
            x_in = x_in.transpose(1,2)
        # print(x_in.shape, self.res_layer1(x_in).shape)
        x = self.conv_block1(x_in) + self.res_layer1(x_in)
        # print(x.shape)
        x = self.relu_pool(x)
        # print(x.shape)
        x = self.conv_block2(x) + self.res_layer2(x)
        # print(x.shape)
        x = self.relu_pool(x)
        # print(x.shape)
        x = self.conv_block3(x) + self.res_layer3(x)
        # print(x.shape)
        x = self.relu_pool(x)
        # print(x.shape)
        x = self.conv_block4(x) + self.res_layer4(x)
        # print(x.shape)
        x = self.relu_pool(x)
        # print(x.shape)
        d = self.decoder_layer1(x)
        # print(d.shape)
        d = self.decoder_layer2(d)
        # print(d.shape)
        d = self.decoder_layer3(d)
        # print(d.shape)
        d = self.decoder_layer4(d)
        # print(d.shape)
        x_flat = x.reshape(x.shape[0], -1)
        # print(x_flat.shape)
        logits = self.logits(x_flat)
        return logits, x, d

# # My
# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
#         self.configs = configs

#         self.norm = nn.BatchNorm1d(configs.input_channels),
#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(configs.input_channels, 32, kernel_size=2, stride=1, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),

#             nn.Conv1d(32, 32, kernel_size=configs.kernel_size, stride=configs.stride, dilation=1, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.Dropout(configs.dropout)
#         )


#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=2, stride=1, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(64),
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, 128, kernel_size=2, stride=1, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),

#             nn.Conv1d(128, 128, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(128),
#         )

#         self.conv_block4 = nn.Sequential(
#             nn.Conv1d(128, configs.final_out_channels, kernel_size=2, stride=1, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),

#             nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=2, stride=1, dilation=1, bias=False, padding=(1)),
#             nn.BatchNorm1d(configs.final_out_channels),
#         )

#         self.relu_pool = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         self.res_layer1 = nn.Conv1d(configs.input_channels, 32, kernel_size=1, stride=configs.stride, padding=0)
#         self.res_layer2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
#         self.res_layer3 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
#         self.res_layer4 = nn.Conv1d(128, configs.final_out_channels, kernel_size=1, stride=1)

#         self.decoder_layer1 = nn.Sequential(
#             nn.ConvTranspose1d(configs.final_out_channels, 128, kernel_size=2, stride=1, dilation=1, bias=False, padding=(1)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.ConvTranspose1d(128, 128, kernel_size=2, stride=1, dilation=5, bias=False, padding=(2)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#         )
#         self.decoder_layer2 = nn.Sequential(
#             nn.ConvTranspose1d(128, 64, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 64, kernel_size=2, stride=1, dilation=5, bias=False, padding=(2)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
#         self.decoder_layer3 = nn.Sequential(
#             nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 32, kernel_size=2, stride=1, dilation=3, bias=False, padding=(1)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2, dilation=1, bias=False, padding=(0)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#         )
#         self.decoder_layer4 = nn.Sequential(
#             nn.ConvTranspose1d(32, configs.input_channels, kernel_size=configs.kernel_size, stride=configs.stride, dilation=1, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(configs.input_channels),
#             nn.ReLU(),
#             nn.Linear((configs.features_len * 8 - 1) * configs.stride - 2 * configs.kernel_size//2 + configs.kernel_size, configs.seq_length),
#             nn.ReLU(),
#         )
        

#         self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

#     def forward(self, x_in):
#         if x_in.shape[1] != self.configs.input_channels:
#             x_in = x_in.transpose(1,2)
#         # print(x_in.shape, self.res_layer1(x_in).shape)
#         # x = self.conv_block1(x_in) + self.res_layer1(x_in)
#         x = self.conv_block1(x_in)
#         # print(x.shape)
#         x = self.relu_pool(x)
#         # print(x.shape)
#         # x = self.conv_block2(x) + self.res_layer2(x)
#         x = self.conv_block2(x)
#         # print(x.shape)
#         x = self.relu_pool(x)
#         # print(x.shape)
#         # x = self.conv_block3(x) + self.res_layer3(x)
#         x = self.conv_block3(x) 
#         # print(x.shape)
#         x = self.relu_pool(x)
#         # print(x.shape)
#         # x = self.conv_block4(x) + self.res_layer4(x)
#         x = self.conv_block4(x)
#         # print(x.shape)
#         x = self.relu_pool(x)
#         # print(x.shape)
#         d = self.decoder_layer1(x)
#         # print(d.shape)
#         d = self.decoder_layer2(d)
#         # print(d.shape)
#         d = self.decoder_layer3(d)
#         # print(d.shape)
#         d = self.decoder_layer4(d)
#         # print(d.shape)
#         x_flat = x.reshape(x.shape[0], -1)
#         # print(x_flat.shape)
#         logits = self.logits(x_flat)
#         return logits, x, d

# # My
# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
#         self.configs = configs
#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(configs.input_channels, 32, kernel_size=2, stride=configs.stride, dilation=1, bias=False, padding=(1)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),

#             nn.Conv1d(32, 32, kernel_size=configs.kernel_size, stride=configs.stride, dilation=1, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
            
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout),
#         )


#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=2, stride=1, dilation=2, bias=False, padding=(1)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
            
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, configs.final_out_channels, kernel_size=2, stride=1, dilation=5, bias=False, padding=(1)),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),

#             nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),
            
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         # if configs.input_channels != configs.final_out_channels:
#         #     self.res_layer = nn.Sequential(
#         #         nn.Conv1d(configs.input_channels, configs.final_out_channels, kernel_size=1, stride=1),
#         #         nn.MaxPool1d(kernel_size=configs.seq_length - configs.features_len + 1, stride=1, padding=0),
#         #     )
#         # else:
#         #     self.res_layer = None

#         self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

#     def forward(self, x_in):
#         # print(x_in.shape)
#         if x_in.shape[1] != self.configs.input_channels:
#             x_in = x_in.transpose(1,2)  
#         x = self.conv_block1(x_in)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x) 
#         # print(x.shape)
#         # x = x + residual
#         # print(x.shape) 
#         x_flat = x.reshape(x.shape[0], -1)
#         # print(x_flat.shape)
#         logits = self.logits(x_flat)
#         return logits, x, x

# --------------------------------------------------------------------------------------------

# My
# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
#         self.configs = configs

#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(configs.input_channels, 32, kernel_size=2, stride=configs.stride, dilation=1, bias=False, padding=(1)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),

#             nn.Conv1d(32, 32, kernel_size=configs.kernel_size, stride=configs.stride, dilation=1, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
            
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout),
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=2, stride=1, dilation=2, bias=False, padding=(1)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
            
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, configs.final_out_channels, kernel_size=2, stride=1, dilation=4, bias=False, padding=(1)),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),

#             nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=8, stride=1, dilation=1, bias=False, padding=(4)),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),
            
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         # if configs.input_channels != configs.final_out_channels:
#         #     self.res_layer = nn.Sequential(
#         #         nn.Conv1d(configs.input_channels, configs.final_out_channels, kernel_size=1, stride=1),
#         #         nn.MaxPool1d(kernel_size=configs.seq_length - configs.features_len + 1, stride=1, padding=0),
#         #     )
#         # else:
#         #     self.res_layer = None

#         self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

#     def forward(self, x_in):
#         # print(x_in.shape)
#         if x_in.shape[1] != self.configs.input_channels:
#             x_in = x_in.transpose(1,2)   
#         x = self.conv_block1(x_in)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         # print(x.shape)
#         # x = x + residual
#         # print(x.shape) 
#         x_flat = x.reshape(x.shape[0], -1)
#         # print(x_flat.shape)
#         logits = self.logits(x_flat)
#         return logits, x

# TS-TCC
# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()

#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
#                       stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout)
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         model_output_dim = configs.features_len
#         self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

#     def forward(self, x_in):
#         # print(x_in.shape)
#         x = self.conv_block1(x_in)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         # print(x.shape) 
#         x_flat = x.reshape(x.shape[0], -1)
#         # print(x_flat.shape) 
#         logits = self.logits(x_flat)
#         return logits, x