import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from sklearn.preprocessing import StandardScaler


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # 若只有1维，将最里层包裹一下
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)
        # make sure the Channels in second dim 真
        if X_train.shape[1] != config.input_channels:
            X_train = np.transpose(X_train, (0, 2, 1)) 

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    # train_dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset, test_dataset])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size if training_mode!='use' else 1,
                                               shuffle=True if training_mode!='use' else False, drop_last=configs.drop_last,
                                               num_workers=0)
    print('train_loader loaded!')

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)
    print('valid_loader loaded!')                                           

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size if training_mode!='use' else 1,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    print('test_loader loaded!')
    return train_loader, valid_loader, test_loader
