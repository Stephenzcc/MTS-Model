import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
import json
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy
from sklearn.cluster import DBSCAN, AffinityPropagation, SpectralClustering, KMeans
from umap import UMAP
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger





def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def save_Feature(dataset, raw, outs, rawLabel):
    raw = np.array(raw)
    outs = np.array(outs)
    # np.save(file='./data.npy', arr=raw)
    np.save(file='./data_embedding.npy', arr=outs)
    # norm_raw = np.apply_along_axis(normalization, axis=2, arr=raw)

    # 两步降维
    # plt.figure()
    # norm_raw = np.array(norm_raw)
    # [N, M, T] = norm_raw.shape

    # axes_1 = plt.subplot(1,2,1)
    # raw_m = np.transpose(norm_raw, (2, 1, 0))
    # raw_m = raw_m.reshape(T, -1).T
    # pca_m = PCA(n_components = 1)
    # raw_m = pca_m.fit_transform(raw_m)
    # print(pca_m.components_)
    # raw_m = raw_m.reshape((M, N))
    # tsne = TSNE(n_components=2, perplexity=1)
    # reduc1 = tsne.fit_transform(raw_m)
    # scatter_1 = plt.scatter(reduc1[:,0],reduc1[:,1])
    # # pca_m = PCA(n_components = 2)
    # # raw_m = pca_m.fit_transform(raw_m)
    # print(raw_m.shape)

    # axes_2 = plt.subplot(1,2,2)
    # raw_t = np.transpose(norm_raw, (1, 0, 2))
    # raw_t = raw_t.reshape(M, -1).T
    # pca_t = PCA(n_components = 1)
    # raw_t = pca_t.fit_transform(raw_t)
    # print(pca_t.components_)
    # raw_t = raw_t.reshape((T, N))
    # tsne = TSNE(n_components=2, perplexity=30)
    # reduc2 = tsne.fit_transform(raw_t)
    # scatter_1 = plt.scatter(reduc2[:,0],reduc2[:,1])
    # # pca_t = PCA(n_components = 2)
    # # raw_t = pca_t.fit_transform(raw_t)
    # print(raw_t.shape)

    # plt.show()

    # # with open("./DRTM.json", "w", encoding='utf-8') as f:
    # #     json.dump({'DRT': reduc1.tolist(), 'DRM': reduc2.tolist()}, f, indent=2, ensure_ascii=False)
    # with open("./Contribution.json", "w", encoding='utf-8') as f:
    #     json.dump({'contributionT': pca_m.components_.tolist(), 'contributionM': pca_t.components_.tolist()}, f, indent=2, ensure_ascii=False)



    # 聚类特征，降维
    plt.figure()

    axes_1 = plt.subplot(1,2,1)
    print(raw.shape)
    raw_np = raw.reshape(raw.shape[0], -1)
    cluster_raw = KMeans(n_clusters=8).fit(raw_np)
    tsne = TSNE(n_components=2, perplexity=30)
    # tsne = UMAP(n_neighbors=30, n_components=2, n_epochs=500, init='spectral', min_dist=0.2, random_state=42)
    reduc = tsne.fit_transform(raw_np)
    print(Counter(rawLabel))
    scatter_1 = plt.scatter(reduc[:,0],reduc[:,1], c = cluster_raw.labels_)
    axes_1.legend(*scatter_1.legend_elements(), title='classes')

    axes_2 = plt.subplot(1,2,2)
    outs = np.array(outs)
    # feature_pool = []
    # for index, o in enumerate(outs):
    #     mean = np.mean(o, 0)
    #     for mea in mean:
    #         feature_pool.append(np.mean(o, 0))
    # save_outs = pd.DataFrame(outs)
    # save_outs.to_csv('./feature.csv', encoding='utf-8')

    print(outs.shape)
    cluster_outs = KMeans(n_clusters=8).fit(outs)
    tsne = TSNE(n_components=2, perplexity=30)
    # tsne = UMAP(n_neighbors=30, n_components=2, n_epochs=500, init='spectral', min_dist=0.2, random_state=42)
    reduc = tsne.fit_transform(outs)
    print(Counter(cluster_outs.labels_))
    scatter_2 = plt.scatter(reduc[:,0],reduc[:,1], c = cluster_outs.labels_)
    axes_2.legend(*scatter_2.legend_elements(), title='classes')

    # scatter = []
    # for index, rec in enumerate(reduc):
    #     [x, y] = rec
    #     label = cluster_outs.labels_[index]
    #     scatter.append([np.around(np.float(x), 2), np.around(np.float(y), 2), int(label)])
    # with open("./scatter.json", "w", encoding='utf-8') as f:
    #     json.dump({'scatter': scatter}, f, indent=2, ensure_ascii=False)

    plt.show()