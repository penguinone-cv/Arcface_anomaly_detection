import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:

    def __init__(self, log_path):
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def collect_history(self, loss, accuracy, val_loss, val_accuracy):
        self.loss_history.append(loss)
        self.acc_history.append(accuracy)
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_accuracy)

    def draw_graph(self):
        plt.plot(self.loss_history, label="loss")
        plt.plot(self.val_loss_history, label="val_loss")
        plt.legend()
        plt.savefig(os.path.join(self.log_path, "loss.png"))
        plt.gca().clear()
        plt.plot(self.acc_history, label="accuracy")
        plt.plot(self.val_acc_history, label="val_accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.log_path, "accuracy.png"))
