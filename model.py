import torch
import torch.nn.functional as F
import torch.nn as nn

#CNN部分の定義クラス
class CNN(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        #入力がカラー画像であれば3次元
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),     #input:64×64×1      output:64×64×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    #input:64×64×64     output:64×64×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    #input:64×64×64     output:64×64×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, 3, padding=1),   #input:32×32×64     output:32×32×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  #input:32×32×128    output:32×32×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  #input:32×32×128    output:32×32×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, 3, padding=1),  #input:16×16×128      output:16×16×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  #input:16×16×256      output:16×16×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  #input:16×16×256      output:16×16×256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, 3, padding=1),  #input:8×8×256      output:8×8×512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  #input:8×8×512      output:8×8×512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  #input:8×8×512      output:8×8×512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(4*4*512, 2048),              #input:8×8×256(1dim vec)    output:1×1×2058
            nn.ReLU(inplace = True),
            nn.Linear(2048, 1024),                  #input:1×1×2048      output:1×1×1024
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),                  #input:1×1×1024      output:1×1×512
            nn.ReLU(inplace = True),
        )

        self.classifier = nn.Linear(512, num_class)

    #順伝播
    def forward(self, x):
        x = self.feature_extractor(x)
        #Fully connected layers
        x = x.view(-1, 4*4*512)     #convert to 1dim vector
        x = self.linear(x)
        #x = self.classifier(x)

        return x

#特徴の結合後の全結合ネットワークの定義クラス
class FCN(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
