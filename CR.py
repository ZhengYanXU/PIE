import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import torch




class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(pretrained=True).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class NegLoss:
    def __init__(self):
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))

    def __call__(self, x, y_hat):
        style_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        style_gram = [gram(fmap) for fmap in style_features]
        L_style = 0
        for i in range(0 , y_hat.size(0) , 2):
            y_hat_i = y_hat[i:i+1]
            y_hat_features_i = self.vgg(y_hat_i)
            y_hat_gram_i = [gram(fmap) for fmap in y_hat_features_i]
            for j in range(2):
                L_style += self.mse_sum(y_hat_gram_i[j], style_gram[j])
        L_style /= y_hat.size(0)
        return L_style


class NpairLoss:
    def __init__(self):
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))

    def __call__(self, anchor, postive , negative):
        anchor_features = self.vgg(anchor)


        anchor_gram = [gram(fmap) for fmap in anchor_features]

        D_AP = 0
        num = 0.0
        for i in range(0, postive.size(0), 2):
            y_hat_i = postive[i:i + 1]
            y_hat_features_i = self.vgg(y_hat_i)
            y_hat_gram_i = [gram(fmap) for fmap in y_hat_features_i]
            for j in range(3):
                D_AP += self.mse_sum(y_hat_gram_i[j], anchor_gram[j])
            D_AN = 0
            num = 0
            loss = 0.0
            for i in range(0 , negative.size(0) , 2):
                    y_hat_i = negative[i:i+1]
                    y_hat_features_i = self.vgg(y_hat_i)
                    y_hat_gram_i = [gram(fmap) for fmap in y_hat_features_i]
                    for j in range(3):
                        D_AN += self.mse_sum(y_hat_gram_i[j], anchor_gram[j])
                    #计算两个距离之差
                    Distance = D_AN - D_AP
                    x = torch.exp(Distance)
                    loss = loss + torch.log(1 + x)
                    D_AN = 0
                    num = num + 1
        loss = loss / num
        return loss

class InfoNceLoss:
    def __init__(self):
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))

    def __call__(self, anchor, postive , negative):
        anchor_features = self.vgg(anchor)
        anchor_gram = [gram(fmap) for fmap in anchor_features]

        D_AP = 0
        num = 0.0

        for i in range(0, postive.size(0), 2):
            y_hat_i = postive[i:i + 1]
            y_hat_features_i = self.vgg(y_hat_i)
            y_hat_gram_i = [gram(fmap) for fmap in y_hat_features_i]
            for j in range(3):
                D_AP += self.mse_sum(y_hat_gram_i[j], anchor_gram[j])

            D_AN = 0.0
            loss = 0.0
            for i in range(0 , negative.size(0) , 2):
                    y_hat_i = negative[i:i+1]
                    y_hat_features_i = self.vgg(y_hat_i)
                    y_hat_gram_i = [gram(fmap) for fmap in y_hat_features_i]
                    for j in range(3):
                        D_AN += self.mse_sum(y_hat_gram_i[j], anchor_gram[j])
                    #计算两个距离之差
                    x = torch.exp(D_AN)
                    y = torch.exp(D_AP)
                    loss = loss - torch.log(x/(x + y))
                    D_AN = 0
                    num = num + 1
        loss = loss / num
        return loss

