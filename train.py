import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
import pytorch_metric_learning
from pytorch_metric_learning import losses, miners, trainers, testers, samplers
from tqdm import tqdm
from sklearn import preprocessing
from model import CNN, FCN
from logger import Logger
from data_loader import DataLoader
from parameter_loader import read_parameters
import os


class Trainer:

    def __init__(self, setting_csv_path, index):
        self.parameters_dict = read_parameters(setting_csv_path, index)                                    #全ハイパーパラメータが保存されたディクショナリ
        self.model_name = self.parameters_dict["model_name"]                                        #モデル名
        self.log_path = os.path.join(self.parameters_dict["base_log_path"], self.model_name)                                             #ログの保存先
        self.batch_size = int(self.parameters_dict["batch_size"])
        self.learning_rate = float(self.parameters_dict["learning_rate"])                           #学習率
        self.momentum = float(self.parameters_dict["momentum"])                                     #慣性
        self.weight_decay = float(self.parameters_dict["weight_decay"])                             #重み減衰
        self.img_size = (int(self.parameters_dict["width"]),int(self.parameters_dict["height"]))    #画像サイズ
        self.logger = Logger(self.log_path)                                                         #ログ書き込みを行うLoggerクラスの宣言
        self.num_class = int(self.parameters_dict["num_class"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  #GPUが利用可能であればGPUを利用
        self.trunk = CNN(self.num_class).to(self.device)                                                            #CNN部分を定義
        #self.trunk = nn.DataParallel(self.trunk.to(self.device))                                    #TwoStreamなので並列で動かせるようにする
        #学習済み重みファイルがあるか確認しあれば読み込み
        if os.path.isfile(os.path.join(self.log_path, self.model_name, self.model_name)):
            print("Trained weight file exists")
            self.trunk.load_state_dict(torch.load(os.path.join(self.log_path, self.model_name)))
        self.embedder = nn.DataParallel(FCN([self.num_class, 64]).to(self.device))                  #FCN部分を定義し並列で動かせるようにする

        #CNN部分の最適化手法の定義
        self.trunk_optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.learning_rate)
        #self.trunk_optimizer = torch.optim.SGD(self.trunk.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        #CNN部分の最適化手法の定義
        self.embedder_optimizer = torch.optim.SGD(self.embedder.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.loss = losses.ArcFaceLoss(margin=float(self.parameters_dict["margin"]), scale=int(self.parameters_dict["scale"]), num_classes=self.num_class, embedding_size=512).to(self.device)
        self.loss_optimizer = torch.optim.SGD(self.loss.parameters(), lr=0.01)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.data_loader = DataLoader(data_path=self.parameters_dict["data_path"],
                                      batch_size=int(self.parameters_dict["batch_size"]),
                                      img_size=self.img_size,
                                      train_ratio=float(self.parameters_dict["train_ratio"]))
        #self.sampler = samplers.MPerClassSampler(self.data_loader.dataloaders["train"].targets, m=3, length_before_new_iter=len(self.data_loader.dataloaders["train"]))

        self.models = {"trunk": self.trunk, "embedder": self.embedder}
        self.optimizers = {"trunk_optimizer": self.trunk_optimizer, "embedder_optimizer": self.embedder_optimizer, "metric_loss_optimizer": self.loss_optimizer}
        self.loss_funcs = {"metric_loss": self.loss}
        self.mining_funcs = {"tuple_miner": self.miner}
        self.trainer = None


        self.gallery_labels = []
        self.query_labels = []




    #def middle_layer_model(self, model):
    #    middle_layer = [layer.output for layer in model.layers[2:20]]
    #    activation_model = Model

    def train(self):
        print("Train phase")
        print("Train", self.model_name)

        epochs = int(self.parameters_dict["epochs"])

        with tqdm(range(epochs)) as pbar:
            for epoch in enumerate(pbar):
                i = epoch[0]
                pbar.set_description("[Epoch %d]" % (i+1))
                loss_result = 0.0
                acc = 0.0
                val_loss_result = 0.0
                val_acc = 0.0

                self.trunk.train()
                j = 1
                for inputs, labels in self.data_loader.dataloaders["train"]:
                    pbar.set_description("[Epoch %d (Iteration %d)]" % ((i+1), j))
                    j = j + 1
                    inputs = inputs.to(self.device)
                    labels = torch.tensor(labels).to(self.device)
                    outputs = self.trunk(inputs)
                    loss = self.loss(outputs, labels)

                    self.optimizers["trunk_optimizer"].zero_grad()
                    loss.backward()

                    self.optimizers["trunk_optimizer"].step()
                    self.optimizers["metric_loss_optimizer"].step()

                    _, preds = torch.max(outputs, 1)
                    loss_result += loss.item()
                    acc += torch.sum(preds == labels.data)

                else:
                    with torch.no_grad():
                        self.trunk.eval()
                        pbar.set_description("[Epoch %d (Validation)]" % (i+1))
                        for val_inputs, val_labels in self.data_loader.dataloaders["val"]:
                            val_inputs = val_inputs.to(self.device)
                            val_labels = torch.tensor(val_labels).to(self.device)
                            val_outputs = self.trunk(val_inputs)
                            val_loss = self.loss(val_outputs, val_labels)

                            _, val_preds = torch.max(val_outputs, 1)
                            val_loss_result += val_loss.item()
                            val_acc += torch.sum(val_preds == val_labels.data)

                    epoch_loss = loss_result / len(self.data_loader.dataloaders["train"].dataset)
                    epoch_acc = acc / len(self.data_loader.dataloaders["train"].dataset)
                    val_epoch_loss = val_loss_result / len(self.data_loader.dataloaders["val"].dataset)
                    val_epoch_acc = val_acc.float() / len(self.data_loader.dataloaders["val"].dataset)
                    self.logger.collect_history(loss=epoch_loss, accuracy=epoch_acc, val_loss=val_epoch_loss, val_accuracy=val_epoch_acc)
                    self.logger.writer.add_scalars("losses", {"train":epoch_loss,"validation":val_epoch_loss}, (i+1))
                    self.logger.writer.add_scalars("accuracies", {"train":epoch_acc, "validation":val_epoch_acc}, (i+1))

                pbar.set_postfix({"loss":epoch_loss, "accuracy": epoch_acc.item(), "val_loss":val_epoch_loss, "val_accuracy": val_epoch_acc.item()})

        torch.save(self.trunk.state_dict(), os.path.join(self.log_path,self.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()

    #def end_of_epoch


    def search(self):
        cache_dir = self.parameters_dict["cache_dir"]
        batch_size = int(self.parameters_dict["batch_size"])
        truncation_size = int(self.parameters_dict["truncation_size"])
        kd = int(self.parameters_dict["k_dict"])
        kq = int(self.parameters_dict["k_query"])
        gamma = int(self.parameters_dict["gamma"])

        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        gallery_num = len(self.data_loader.img_path_dict["gallery"])
        gallery_loader = self.data_loader.load_database(mode=True)
        gallery = []
        with tqdm(total=len(gallery_loader), unit="batch") as pbar:
            with torch.no_grad():
                for gallery_imgs, gallery_labels in gallery_loader:
                    pbar.set_description("[gallery's feature extracting]")
                    features = self.model.feature_extractor(gallery_imgs.to(self.device))
                    for i in range(len(features)):
                        feature = features[i].to("cpu").view(-1, 4*4*256).numpy()
                        gallery.append(feature[0])
                        self.gallery_labels.append(gallery_labels[i].to("cpu").numpy())
                    pbar.update(1)

        #with torch.no_grad():
        #    print("[gallery's feature extracting]")
        #    for gallery_imgs, gallery_labels in gallery_loader:
        #        features = self.model.feature_extractor(gallery_imgs.to(self.device))
        #        for i in range(len(features)):
        #            feature = features[i].to("cpu").view(-1, 4*4*256).numpy()
        #            gallery.append(feature[0])
        #            self.gallery_labels.append(gallery_labels[i].to("cpu").numpy())

        query = []
        query_loader = self.data_loader.load_database(mode=False)
        with tqdm(total=len(query_loader), unit="batch") as pbar:
            with torch.no_grad():
                for query_imgs, query_labels in query_loader:
                    pbar.set_description("[query's feature extracting]")
                    features = self.model.feature_extractor(query_imgs.to(self.device))
                    for i in range(len(features)):
                        feature = features[i].to("cpu").view(-1, 4*4*256).numpy()
                        query.append(feature[0])
                        self.query_labels.append(query_labels[i].to("cpu").numpy())
                    pbar.update(1)

        #with torch.no_grad():
        #    print("[query's feature extracting]")
        #    for query_imgs, query_labels in query_loader:
        #        features = self.model.feature_extractor(query_imgs.to(self.device))
        #        for i in range(len(features)):
        #            feature = features[i].to("cpu").view(-1, 4*4*256).numpy()
        #            query.append(feature[0])
        #            self.query_labels.append(query_labels[i].to("cpu").numpy())

        print(np.array(gallery).shape)
        n_query = len(query)
        stacks = np.vstack([query, gallery])
        print(stacks.shape)
        #diffusion = Diffusion(stacks, cache_dir)
        #offline = diffusion.get_offline_results(truncation_size, kd)
        features = preprocessing.normalize(offline, norm="l2", axis=1)
        scores = features[:n_query] @ features[n_query:].T
        ranks = np.argsort(-scores.todense())

        np.savetxt(os.path.join(self.log_path, "ranks.csv"), ranks)
        n = int(self.parameters_dict["look_num"])
        self.calc_RatK(ranks, n)
        self.save_retrieved_image(ranks, n)

    def calc_RatK(self, ranks, n=20):
        n_query = len(self.query_labels)
        total = n_query * n
        collect = 0
        ratk = []
        for gallery_index in range(n):
            for query_index in range(n_query):
                if self.query_label_name[self.query_labels[query_index]] == self.gallery_label_name[self.gallery_labels[ranks[query_index,gallery_index]]]:
                    collect += 1
            print("R@K( k = ", gallery_index+1, ") = ", (collect/total))
            ratk.append(collect/total)

        x = [i+1 for i in range(n)]
        plt.figure()
        plt.plot(x, ratk)
        plt.xticks(x)
        plt.ylim(0, 1)
        plt.title("mean R@K(n=20)")
        plt.savefig(os.path.join(self.log_path, "R@K.png"))
        plt.close()

    def save_retrieved_image(self, ranks, n=19):
        n_query = len(self.query_labels)
        col = 5
        row = math.ceil((n+1)/col)
        for query_index in range(n_query):
            plt.figure()
            plt.subplot(row, col, 1)
            plt.imshow(self.data_loader.load_img(mode=False, index=query_index), cmap="gray")
            plt.axis("off")
            for gallery_index in range(n):
                plt.subplot(row, col, gallery_index+2)
                plt.imshow(self.data_loader.load_img(mode=True, index=ranks[query_index,gallery_index]), cmap="gray")
                plt.axis("off")
            plt.savefig(os.path.join(self.log_path, (self.query_label_name[self.query_labels[query_index]] + "_" + str(query_index) + ".png")))
        plt.close("all")
