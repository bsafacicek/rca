import yaml, re, os, glob, shutil, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if isinstance(m, nn.BatchNorm2d):
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant_(m.weight, 1) # gamma
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0) # beta
            init.constant_(m.running_var, 1) # variance
            init.constant_(m.running_mean, 0)
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            init.constant_(m.bias, 0)


def load_net(config):
    if config['net_name'] == "small_cnn":
        from networks import small_cnn
        is_avg_pool = False if config['source_dataset'] == 'MNIST28' else True
        net = small_cnn(config=config, is_avg_pool=is_avg_pool).cuda()
        init_params(net)
    elif config['net_name'] == "conv_large":
        from networks import conv_large
        net = conv_large(config).cuda()
        init_params(net)
    return net


def log(text, log_file):
    print(text)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(text + '\n')
            f.flush()
            f.close()


def log_losses(config, epoch, lr, st_time):
    log("epoch=%d/%d,lr=%f,time=%d" % (epoch+1, config["num_epochs"], lr, time.time()-st_time), config['out_path'])

    log("loss_ce=%.3f, loss_ent=%.3f, loss_vat=%.3f" % (config['loss_ce'], config['loss_ent'], config['loss_vat']), config['out_path'])
    log("loss_src_joint=%.3f, loss_trg_joint=%.3f" % (config['loss_src_joint'], config['loss_trg_joint']), config['out_path'])
    log("loss_feat_adv=%.3f"%(config['loss_feat_adv']), config['out_path'])

    log("acc_source_train=%.3f,acc_target_train=%.3f" % (config['acc_source_train'], config['acc_target_train']), config['out_path'])

    log("acc_source_class_valid=%.3f,acc_source_joint_valid=%.3f" % (config['acc_source_class_valid'],config['acc_source_joint_valid']), config['out_path'])
    log("acc_target_class_valid=%.3f,acc_target_joint_valid=%.3f" % (config['acc_target_class_valid'],config['acc_target_joint_valid']), config['out_path'])


def log_losses_source_only(config, epoch, lr, st_time):
    log('epoch=%d/%d,lr=%f,time=%d' % (epoch+1, config['num_epochs'], lr, time.time()-st_time), config['out_path'])
    log('loss_ce=%.3f,acc_source_train=%.3f' % (config['loss_ce'], config['acc_source_train']), config['out_path'])
    log('acc_source_class_valid=%.3f' % (config['acc_source_class_valid']), config['out_path'])
    log("acc_target_class_valid=%.3f" % (config['acc_target_class_valid']), config['out_path'])


def lambda_schedule(config, config_init, epoch):
    # reset lambda values:
    for key, val in config_init.items():
        if 'lambda_' in key:
            config[key] = val

    # update lambda values:
    if config['is_curriculum']:
        if epoch < config["num_epochs"]*(1.0/15):
            config['lambda_ent'] = 0
            config['lambda_src_vat'] = 0
            config['lambda_trg_vat'] = 0
            config['lambda_trg_joint'] = 0
            config['lambda_src_feat_adv'] = 0
            config['lambda_trg_feat_adv'] = 0
            config['lambda_consis_src'] = 0
            config['lambda_consis_trg'] = 0
        elif epoch < config["num_epochs"]*(2.0/15):
            config['lambda_trg_joint'] = 0
            config['lambda_src_feat_adv'] = 0
            config['lambda_trg_feat_adv'] = 0
            config['lambda_consis_src'] = 0
            config['lambda_consis_trg'] = 0
        elif epoch < config["num_epochs"]*(6.0/15):
            config['lambda_src_feat_adv'] = 0
            config['lambda_trg_feat_adv'] = 0

    return config


class torch_optimizer(nn.Module):
    def __init__(self, config, net, is_encoder):
        self.lr_scheme = config["lr_scheme"]
        self.num_epochs = config["num_epochs"]
        self.max_lr = config["max_lr"]

        if self.lr_scheme == "SGD":
            self.optimizer = optim.SGD([{'params': net.parameters()}], self.max_lr,
                                        momentum=0.9, weight_decay=10**-4, nesterov=False)

        elif self.lr_scheme == "ADAM":
            self.optimizer = optim.Adam([{'params': net.parameters()}], lr=self.max_lr,
                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=10**-4)


    def update_lr(self, epoch):
        if self.lr_scheme == "SGD":
            if epoch < self.num_epochs*(7.0/15):
                lr = self.max_lr
            elif epoch < self.num_epochs*(10.0/15):
                lr = self.max_lr/2.0
            elif epoch < self.num_epochs*(12.0/15):
                lr = self.max_lr/4.0
            elif epoch < self.num_epochs*(14.0/15):
                lr = self.max_lr/8.0
            else:
                lr = self.max_lr/16.0

        elif self.lr_scheme == "ADAM":
            if epoch < self.num_epochs*(7.0/15):
                lr = self.max_lr
            elif epoch < self.num_epochs*(10.0/15):
                lr = self.max_lr/2.0
            elif epoch < self.num_epochs*(12.0/15):
                lr = self.max_lr/4.0
            elif epoch < self.num_epochs*(14.0/15):
                lr = self.max_lr/8.0
            else:
                lr = self.max_lr/16.0

        for g in self.optimizer.param_groups:
            g['lr'] = lr

        return lr, self.optimizer


def get_config(config):
    with open(config, 'r') as stream:
        config = yaml.load(stream)

    # initialize accurices:
    keys = ['acc_source_class_valid', 'acc_source_joint_valid',
            'acc_target_class_valid', 'acc_target_joint_valid']

    for key in keys:
        config[key] = 0

    return config


def get_run_id(out_dir):
    dir_names = os.listdir(out_dir)
    r = re.compile("^\\d+")
    run = 0
    for dir_name in dir_names:
        m = r.match(dir_name)
        if m is not None:
            i = int(m.group())
            run = max(run, i + 1)
    return run


def copy_src_files(out_name):
    directory = 'src/' + out_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for source in glob.glob("*.py"):
        destination = directory + source
        shutil.copyfile(source, destination)

    source = "configs/"
    shutil.copytree(source, directory + source)
