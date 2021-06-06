import torch.nn as nn
from torch.autograd import Variable
from torch.cuda import FloatTensor

import numpy as np


def update_layer(layer, requires_grad):
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


def average_preds(x, nets):
    y = 0
    for i in range(len((nets))):
        y += nets[i](x).data
    y /= (i+1.0)
    return Variable(y)


def forward_(self, x, state = "test"):
    assert(state in ["test", "all", "only_joint", "encoder"])
    if state == "test":
        f = average_preds(x, self.encoders)
        class_prob = average_preds(f.clone(), self.class_classifiers)
        joint_prob = average_preds(f.clone(), self.joint_classifiers)
    else:
        if state == "only_joint":
            self.encoders[self.inds[0]] = update_layer(
                self.encoders[self.inds[0]], requires_grad=True)
            self.class_classifiers[self.inds[1]] = update_layer(
                self.class_classifiers[self.inds[1]], requires_grad=False)
            self.joint_classifiers[self.inds[2]] = update_layer(
                self.joint_classifiers[self.inds[2]], requires_grad=True)
        elif state == "encoder":
            self.encoders[self.inds[0]] = update_layer(
                self.encoders[self.inds[0]], requires_grad=True)
            self.class_classifiers[self.inds[1]] = update_layer(
                self.class_classifiers[self.inds[1]], requires_grad=False)
            self.joint_classifiers[self.inds[2]] = update_layer(
                self.joint_classifiers[self.inds[2]], requires_grad=False)
        elif state == "all":
            self.encoders[self.inds[0]] = update_layer(
                self.encoders[self.inds[0]], requires_grad=True)
            self.class_classifiers[self.inds[1]] = update_layer(
                self.class_classifiers[self.inds[1]], requires_grad=True)
            self.joint_classifiers[self.inds[2]] = update_layer(
                self.joint_classifiers[self.inds[2]], requires_grad=True)

        f = self.encoders[self.inds[0]](x)
        class_prob = self.class_classifiers[self.inds[1]](f.clone())
        joint_prob = self.joint_classifiers[self.inds[2]](f.clone())

    return class_prob, joint_prob


class large_joint_classifier(nn.Module):
    def __init__(self, config, is_drop=False, is_bn_aff=True):
        super(large_joint_classifier, self).__init__()
        self.drop = nn.Dropout(p = config['drop_prob'])
        self.is_drop = is_drop
        self.linears = nn.Linear(128, 2*config['nb_class'])
    def forward(self, x):
        if self.is_drop:
            x = self.drop(x)
        x = self.linears(x)
        return x


class large_class_classifier(nn.Module):
    def __init__(self, config, is_drop=False, is_bn_aff=True):
        super(large_class_classifier, self).__init__()
        self.drop = nn.Dropout(p = config['drop_prob'])
        self.is_drop = is_drop
        self.linears = nn.Linear(128, config['nb_class'])
    def forward(self, x):
        if self.is_drop:
            x = self.drop(x)
        x = self.linears(x)
        return x


class large_encoder(nn.Module):
    def __init__(self, config, in_ch=3, cnn_bias=True, mom_bn=0.1, bn_eps=10**-5, is_bn_aff=True):
        super(large_encoder, self).__init__()
        self.config = config
        self.inst_norm = nn.InstanceNorm2d(3, affine=True)
        self.lr = nn.LeakyReLU(0.1)
        self.mp2_2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop = nn.Dropout(p = config['drop_prob'])

        ch_sizes = [128, 128, 128, 256, 256, 256, 512, 256, 128]
        self.nb_layer = len(ch_sizes)

        self.bns = nn.ModuleList([])
        for ch_size in ch_sizes:
            self.bns.append(nn.BatchNorm2d(ch_size,
                                           eps=bn_eps,
                                           momentum=mom_bn,
                                           affine=is_bn_aff).cuda())

        self.linears = nn.ModuleList([nn.Conv2d(in_ch,
                                                ch_sizes[0],
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                bias=cnn_bias)])

        for i in range(self.nb_layer-4):
            self.linears.append(nn.Conv2d(ch_sizes[i],
                                          ch_sizes[i+1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=cnn_bias))

        self.linears.append(nn.Conv2d(ch_sizes[self.nb_layer-4],
                                      ch_sizes[self.nb_layer-3],
                                      kernel_size=3,
                                      stride=1,
                                      padding=0,
                                      bias=cnn_bias))

        self.linears.append(nn.Conv2d(ch_sizes[self.nb_layer-3],
                                      ch_sizes[self.nb_layer-2],
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=cnn_bias))

        self.linears.append(nn.Conv2d(ch_sizes[self.nb_layer-2],
                                      ch_sizes[self.nb_layer-1],
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=cnn_bias))

        if in_ch == 1:
            pool_size = 5
        elif in_ch == 3:
            pool_size = 6
        self.avg_6 = nn.AvgPool2d(pool_size, stride=2, padding=0)

    def forward(self, x):
        if self.config['instance_norm']:
            x = self.inst_norm(x)

        for i in range(3):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        x = self.mp2_2(x)
        x = self.drop(x)
        if self.training and self.config['std'] > 10**-4:
            x += Variable(FloatTensor(x.size()).normal_())*self.config['std']

        for i in range(3, 6):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        x = self.mp2_2(x)
        x = self.drop(x)
        if self.training and self.config['std'] > 10**-4:
            x += Variable(FloatTensor(x.size()).normal_())*self.config['std']

        for i in range(6, self.nb_layer):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        x = self.avg_6(x)
        x = x.view(x.size(0),-1)

        return x


class conv_large(nn.Module):
    def __init__(self, config, in_ch=3, cnn_bias=True, mom_bn=0.1):
        super(conv_large, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([])
        for _ in range(config['nb_e']):
            self.encoders.append(large_encoder(config=config,
                                               in_ch=in_ch,
                                               cnn_bias=cnn_bias,
                                               mom_bn=mom_bn).cuda())

        self.class_classifiers = nn.ModuleList([])
        for _ in range(config['nb_cc']):
            self.class_classifiers.append(large_class_classifier(config=config,
                                                                 is_drop=True))

        self.joint_classifiers = nn.ModuleList([])
        for _ in range(config['nb_jc']):
            self.joint_classifiers.append(large_joint_classifier(config=config,
                                                                 is_drop=True))

    def forward(self, x, state = "test"):
        class_prob, joint_prob = forward_(self, x, state = state)
        return class_prob, joint_prob


class small_joint_classifier(nn.Module):
    def __init__(self, config, in_ch=3, cnn_bias=True, mom_bn=0.1, bn_eps=10**-5, is_bn_aff=True, is_avg_pool=True):
        super(small_joint_classifier, self).__init__()
        self.config = config
        self.lr = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(p=config['drop_prob'], inplace=True)
        self.is_avg_pool = is_avg_pool

        self.bns = nn.ModuleList([])
        input_ch = 64
        ch_sizes = [64, 64, 64]

        for ch_size in ch_sizes:
            self.bns.append(nn.BatchNorm2d(ch_size, eps=bn_eps, momentum=mom_bn, affine=is_bn_aff).cuda())
        self.bns.append(nn.BatchNorm1d(2*config['nb_class'], eps=bn_eps, momentum=mom_bn, affine=is_bn_aff).cuda())

        self.linears = nn.ModuleList([nn.Conv2d(input_ch, ch_sizes[0], kernel_size=3, stride=1, padding=0, bias=cnn_bias).cuda()])
        for i in range(len(ch_sizes)-1):
            self.linears.append(nn.Conv2d(ch_sizes[i], ch_sizes[i+1], kernel_size=3, stride=1, padding=0, bias=cnn_bias).cuda())
        self.linears.append(nn.Linear(ch_sizes[2], 2*config['nb_class']))

        if in_ch == 1:
            pool_size = 5
        elif in_ch == 3:
            pool_size = 2
        self.avg_6 = nn.AvgPool2d(pool_size, stride=2, padding = 0)

    def forward(self, x):
        self.drop(x)
        if self.training and self.config['std'] > 10**-4:
            x += Variable(FloatTensor(x.size()).normal_())*self.config['std']

        for i in range(3):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        # do not call average pooling for image with length 28:
        if self.is_avg_pool:
            x = self.avg_6(x)
        x = x.view(x.size(0),-1)

        x = self.linears[3](x)
        if self.config['is_bn']:
            x = self.bns[3](x)
        return x


class small_class_classifier(nn.Module):
    def __init__(self, config, in_ch=3, cnn_bias=True, mom_bn=0.1, bn_eps = 10**-5, is_bn_aff=True, is_avg_pool=True):
        super(small_class_classifier, self).__init__()
        self.config = config
        self.lr = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(p=config['drop_prob'], inplace=True)
        self.is_avg_pool = is_avg_pool

        self.bns = nn.ModuleList([])
        input_ch = 64
        ch_sizes = [64, 64, 64]

        self.nb_layer = len(ch_sizes)

        for ch_size in ch_sizes:
            self.bns.append(nn.BatchNorm2d(ch_size, eps=bn_eps, momentum=mom_bn, affine=is_bn_aff).cuda())
        self.bns.append(nn.BatchNorm1d(config['nb_class'], eps=bn_eps, momentum=mom_bn, affine=is_bn_aff).cuda())

        self.linears = nn.ModuleList([nn.Conv2d(input_ch, ch_sizes[0], kernel_size=3, stride=1, padding=0, bias=cnn_bias).cuda()])
        for i in range(len(ch_sizes)-1):
            self.linears.append(nn.Conv2d(ch_sizes[i], ch_sizes[i+1], kernel_size=3, stride=1, padding=0, bias=cnn_bias).cuda())
        self.linears.append(nn.Linear(ch_sizes[-1], config['nb_class']))

        if in_ch == 1:
            pool_size = 5
        elif in_ch == 3:
            pool_size = 2
        self.avg_6 = nn.AvgPool2d(pool_size, stride=2, padding = 0)

    def forward(self, x):
        self.drop(x)
        if self.training and self.config['std'] > 10**-4:
            x += Variable(FloatTensor(x.size()).normal_())*self.config['std']

        for i in range(self.nb_layer):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        # do not call average pooling for image with length 28:
        if self.is_avg_pool:
            x = self.avg_6(x)

        x = x.view(x.size(0),-1)

        x = self.linears[self.nb_layer](x)
        if self.config['is_bn']:
            x = self.bns[self.nb_layer](x)
        return x


class small_encoder(nn.Module):
    def __init__(self, config, in_ch=3, cnn_bias=True, mom_bn=0.1, bn_eps=10**-5, is_bn_aff=True):
        super(small_encoder, self).__init__()
        self.config = config
        self.inst_norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.lr = nn.LeakyReLU(0.1)
        self.mp2_2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop = nn.Dropout(p=config['drop_prob'], inplace=True)

        self.bns = nn.ModuleList([])
        ch_sizes = [64, 64, 64, 64, 64, 64]
        for ch_size in ch_sizes:
            self.bns.append(nn.BatchNorm2d(ch_size, eps=bn_eps, momentum=mom_bn, affine=is_bn_aff).cuda())

        self.linears = nn.ModuleList([nn.Conv2d(in_ch, ch_sizes[0], kernel_size=3,
                                                stride=1, padding=1, bias=cnn_bias)])
        for i in range(len(ch_sizes)-1):
            self.linears.append(nn.Conv2d(ch_sizes[i], ch_sizes[i+1], kernel_size=3,
                                          stride=1, padding=1, bias=cnn_bias))

    def forward(self, x):
        if self.config['instance_norm']:
            x = self.inst_norm(x)

        for i in range(3):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        x = self.mp2_2(x)
        self.drop(x)
        if self.training and self.config['std'] > 10**-4:
            x += Variable(FloatTensor(x.size()).normal_())*self.config['std']

        for i in range(3, 6):
            x = self.linears[i](x)
            if self.config['is_bn']:
                x = self.bns[i](x)
            x = self.lr(x)

        x = self.mp2_2(x)
        return x


class small_cnn(nn.Module):
    def __init__(self, config, in_ch=3, cnn_bias=True, mom_bn=0.1, is_avg_pool=True):
        super(small_cnn, self).__init__()
        self.config = config
        self.inds = []

        self.encoders = nn.ModuleList([])
        for _ in range(config['nb_e']):
            self.encoders.append(small_encoder(config=config,
                                               in_ch=in_ch,
                                               cnn_bias=cnn_bias,
                                               mom_bn=mom_bn).cuda())

        self.class_classifiers = nn.ModuleList([])
        for _ in range(config['nb_cc']):
            self.class_classifiers.append(small_class_classifier(config=config,
                                                                 in_ch=in_ch,
                                                                 cnn_bias=cnn_bias,
                                                                 mom_bn=mom_bn,
                                                                 is_avg_pool=is_avg_pool).cuda())

        self.joint_classifiers = nn.ModuleList([])
        for _ in range(config['nb_jc']):
            self.joint_classifiers.append(small_joint_classifier(config=config,
                                                                 in_ch=in_ch,
                                                                 cnn_bias=cnn_bias,
                                                                 mom_bn=mom_bn,
                                                                 is_avg_pool=is_avg_pool).cuda())

    def forward(self, x, state = "test"):
        class_prob, joint_prob = forward_(self, x, state = state)
        return class_prob, joint_prob
