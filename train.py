import time
import os
import argparse
from copy import deepcopy

import torch

from training_loop import training_loop, testing_loop, training_loop_source_only
from datasets import load_datasets
from utils import load_net, log, lambda_schedule, torch_optimizer, get_config
from utils import get_run_id, copy_src_files, log_losses, log_losses_source_only


parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', default="SVHN", type=str)
parser.add_argument('--target_dataset', default="MNIST", type=str)
parser.add_argument('--source_only_train', action='store_true')
args = parser.parse_args()

out_name = "%s2%s"%(args.source_dataset, args.target_dataset)
if args.source_only_train:
    out_name += '_source_only'

config_name = "configs/" + out_name + ".yaml"
out_dir = 'results/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

config_init = get_config(config_name)
run = get_run_id(out_dir)
out_name = "%04d_" %(run) + out_name
copy_src_files(out_name)
config_init["num_iter_per_epoch"] = 400 * 200 // (config_init["b_source"] + config_init["b_target"])
config_init['nb_class'] = 9 if args.source_dataset in ["CIFAR", "STL"] else 10
config_init['out_path'] = out_dir + out_name + ".txt"
config_init['source_dataset'] = args.source_dataset
config_init['target_dataset'] = args.target_dataset
config_init['source_only_train'] = args.source_only_train

# Log experiment settings.
for key in sorted(config_init):
    conf = key + " -> " + str(config_init[key])
    log(conf, config_init['out_path'])

# Load datasets.
trainloader_source, testloader_source, trainloader_target, testloader_target = load_datasets(config_init)

# Load network.
net = load_net(config_init)

# Load optimizer.
my_optimizer = torch_optimizer(config_init, net, is_encoder=False)

# To keep initial loss weights (lambda).
config = deepcopy(config_init)

for epoch in range(config["num_epochs"]):
    st_time = time.time()

    lr, optimizer = my_optimizer.update_lr(epoch)
    config = lambda_schedule(config, config_init, epoch)

    if config_init['source_only_train']:
        config = training_loop_source_only(net=net,
                                           optimizer=optimizer,
                                           trainloader_source=trainloader_source,
                                           config=config,
                                           epoch=epoch)
    else:
        config = training_loop(net=net,
                               optimizer=optimizer,
                               trainloader_source=trainloader_source,
                               trainloader_target=trainloader_target,
                               config=config,
                               epoch=epoch)

    config = testing_loop(net, testloader_source, config, is_source=True)
    config = testing_loop(net, testloader_target, config, is_source=False)

    print(out_name)
    if config_init['source_only_train']:
        log_losses_source_only(config, epoch, lr, st_time)
    else:
        log_losses(config, epoch, lr, st_time)
