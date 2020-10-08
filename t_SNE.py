# -*- coding: utf-8 -*-
from train import Trainer
import tensorflow as tf
import os
import time
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from tqdm import tqdm

def t_SNE():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    setting_csv_path = "./setting.csv"
    trainer = Trainer(setting_csv_path=setting_csv_path)
    model_file = trainer.model_name + ".h5"
    if not os.path.isfile(os.path.join(trainer.log_path, model_file)):
        print("Trained weight file does not exist")
        trainer.train()

    all_images_path, all_class_label, query_labels, gallery_labels, n_query, n_gallery = trainer.data_loader.get_merge_list()
    class_labels = []
    for i in range(len(all_class_label)):
        class_labels.append([all_class_label[i]])
    embedding_layer = K.function([trainer.model.layers[2].layers[0].input], [trainer.model.layers[2].layers[46].output])        #Siamese NetworkのCNN部分のみ(特徴抽出器

    #特徴抽出
    embeddings = []
    with tqdm(range(len(all_images_path))) as progress_bar:
        for loop in enumerate(progress_bar):
            i = loop[0]
            #プログレスバーのタイトル設定
            dot = ""
            if i%25==0:
                for j in range((i//25)%5):
                    dot = dot + "."
                description = "[Feature extructing" + dot + "]"
                progress_bar.set_description(description)
            #画像の読み込み
            images_list, _ = trainer.data_loader.convert_pathlist_to_image_and_label(all_images_path[i], np.zeros((len(all_images_path[i]), 1)), shuffle=False)
            #images_list = np.asarray()
            #特徴を抽出して配列に追加
            embedding = embedding_layer([np.expand_dims(images_list[0][0, :, :, :], axis=0),])
            embedding = np.asarray(embedding)
            embeddings.append(embedding[0,0,:])

    embeddings = np.asarray(embeddings)

    print("Dimension compressing...")
    start = time.time()
    embeddings_reduced = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    finish = time.time() - start
    print("Complete")
    print("Execution time : {0}".format(finish) + "[sec]")
    print(embeddings_reduced.shape)
    concatenated = np.hstack((embeddings_reduced, class_labels))
    df = pd.DataFrame(concatenated, columns=["x", "y", "class"])
    print(df)
    df.to_csv(os.path.join(trainer.log_path, "dataframe.csv"))
    grouped = df.groupby('class')
    print(grouped.size())

    fig, ax = plt.subplots()
    ax.tick_params(bottom=False,
                   left=False,
                   right=False,
                   top=False)
    ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
    #ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
    #ax.yaxis.set_major_locator(mpl.ticker.AutoLocator())
    print("Now plotting...")
    start = time.time()
    for name, group in grouped:
        ax.plot(group.x.astype("float32"), group.y.astype("float32"), marker='o', linestyle='', ms=2, label=name)
    flg=3
    plt.legend(loc='center left', bbox_to_anchor=(1., .5), fontsize=flg, title='Function').get_title().set_fontsize(flg)
    plt.savefig(os.path.join(trainer.log_path, "T-SNE.png"))
    plt.close()
    finish = time.time() - start
    print("Complete")
    print("Execution time : {0}".format(finish) + "[sec]")

if __name__ == "__main__":
    t_SNE()
