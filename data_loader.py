import os
import numpy as np
import joblib
import csv
import torch
from torch import utils
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

class DataLoader:
    def __init__(self, data_path, batch_size, img_size, train_ratio=0.8):
        self.dataset_path = os.path.join(data_path, "dataset")                  #画像データのパス
        self.gallery_path = os.path.join(data_path, "gallery")
        self.query_path = os.path.join(data_path, "query")
        self.img_path_dict = {}
        self.img_size = img_size
        self.batch_size = batch_size                #バッチサイズ
        #画像の変換方法の選択
        self.train_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])])
        self.validation_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])])
        self.load_data()
        self.dataloaders = self.import_image(batch_size=batch_size, train_ratio=train_ratio)

#gallery(画像データベース)とquery(検索対象画像データベース)への画像のパスの登録を行う関数
#画像を読み込むのを楽にするための処理なので実際に画像を読み込むわけではない
#各要素の0番目に画像パス，1番目にクラス名が登録される
    def load_data(self):
        gallery_img_list = []
        query_img_list = []

        for symbol in tqdm(os.listdir(self.gallery_path), desc="[make list for gallery(gallery data)]"):
            symbol_path = os.path.join(self.gallery_path, symbol)
            for image in os.listdir(symbol_path):
                img_path = os.path.join(symbol_path, image)
                gallery_img_list.append((img_path, symbol))

        self.img_path_dict["gallery"] = gallery_img_list
        print("Complete making list")
        print("gallery list length : ", len(self.img_path_dict["gallery"]))

        for symbol in tqdm(os.listdir(self.query_path), desc="[make list for query]"):
            symbol_path = os.path.join(self.query_path, symbol)
            for image in os.listdir(symbol_path):
                img_path = os.path.join(symbol_path, image)
                query_img_list.append((img_path, symbol))


        self.img_path_dict["query"] = query_img_list
        print("Complete making list")
        print("query list length : ", len(self.img_path_dict["query"]))

#学習に使うデータの読み込みを行う関数
#Argument
#dataset_path: データセットが格納されているディレクトリのパス
#batch_size: バッチサイズ
#train_ratio: データ全体から学習に使うデータの割合(デフォルト値: 0.8)
#img_size: 画像のサイズ タプルで指定すること(デフォルト値: (64, 64))
#
    def import_image(self, batch_size, train_ratio=0.8):
        #torchvision.datasets.ImageFolderで画像のディレクトリ構造を元に画像読み込みとラベル付与を行ってくれる
        #transformには前処理を記述
        #(x*255. - 127.5)/127.5で値の範囲を[-1.,1.]にする
        data = datasets.ImageFolder(root=self.dataset_path, transform=self.train_transform)

        train_size = int(train_ratio * len(data))           #学習データ数
        val_size = len(data) - train_size                   #検証データ数
        data_size = {"train":train_size, "val":val_size}    #それぞれのデータ数をディクショナリに保存

        train_data, val_data = utils.data.random_split(data, [train_size, val_size])    #torcn.utils.data.random_splitで重複なしにランダムな分割が可能

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)   #学習データのデータローダー
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)      #検証データのデータローダー
        dataloaders = {"train":train_loader, "val":val_loader}                                  #それぞれのデータローダーをディクショナリに保存
        #datas = {"train":train_data, "val":val_data}
        return dataloaders#, datas

#"""
#galleryデータとqueryデータを読み込むためのdataloaderを返す関数
#Argument
#mode: galleryを読み込むかqueryを読み込むか指定(True: gallery, False: query)
#Return
#gallery/query_loader: それぞれのデータを読み込むための関数
#"""
    def load_database(self, mode):
        transform = transforms.Compose([transforms.Resize(self.img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])])

        gallery_data = datasets.ImageFolder(root=self.gallery_path, transform=self.validation_transform)
        query_data = datasets.ImageFolder(root=self.query_path, transform=self.validation_transform)

        #gallery_loader = MakeDataLoader(gallery_data)
        #query_loader = MakeDataLoader(query_data)
        gallery_loader = torch.utils.data.DataLoader(gallery_data)
        query_loader = torch.utils.data.DataLoader(query_data)

        if mode:
            print("[mode : gallery]")
            return gallery_loader
        else:
            print("[mode : query]")
            return query_loader


#引数をもとに目的の画像を読み込んで返す関数
#Argument
#mode: galleryから持ってくるかqueryから持ってくるか指定
#index: 持ってくる画像の番号
#Return
#image: 画像のndarray
    def load_img(self, mode, index):
        load = ["gallery" if mode else "query"][0]

        if index > len(self.img_path_dict[load]):
            print(" ***Error occured*** ")
            print("index over length of list")
            print("returned red image")
            return Image.new(mode="RGB", size=self.img_size, color=(256, 0, 0))

        #print("load ",self.img_path_dict[load][index][0])
        image = Image.open(self.img_path_dict[load][index][0])
        image = image.convert("RGB")
        image = image.resize(self.img_size)

        return np.asarray(image)

class MakeDataLoader(utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.targets = []
        print("Making target list...")
        for i in range(len(data)):
            self.targets.append(data[i][1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index][0], self.targets[index]
        return img, target
