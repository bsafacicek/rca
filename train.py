import time, os, argparse
from copy import deepcopy
import torch

from training_loop import training_loop, testing_loop
from datasets import load_datasets
from utils import load_net, log, lambda_schedule, torch_optimizer, get_config, get_run_id, copy_src_files, log_losses

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', default="SVHN", type=str)
parser.add_argument('--target_dataset', default="MNIST", type=str)
args = parser.parse_args()

out_name = "%s2%s"%(args.source_dataset, args.target_dataset)
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

# log experiment settings:
for key in sorted(config_init):
    conf = key + " -> " + str(config_init[key])
    log(conf, config_init['out_path'])

# load datasets:
trainloader_source, testloader_source, trainloader_target, testloader_target = load_datasets(config_init)

# load network:
net = load_net(config_init)

# load optimizer:
my_optimizer = torch_optimizer(config_init, net, is_encoder=False)

# to keep initial loss weights (lambda):
config = deepcopy(config_init)

for epoch in range(config["num_epochs"]):
    st_time = time.time()

    lr, optimizer = my_optimizer.update_lr(epoch)
    config = lambda_schedule(config, config_init, epoch)

    config = training_loop(net, optimizer, trainloader_source, trainloader_target, config, epoch)

    config = testing_loop(net, testloader_source, config, is_source=True)
    config = testing_loop(net, testloader_target, config, is_source=False)

    print(out_name)
    log_losses(config, epoch, lr, st_time)
