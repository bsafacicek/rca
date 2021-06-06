import numpy as np
np.set_printoptions(threshold=2)

import torch
from torch.autograd import Variable
import torch.nn as nn

from loss import vat_loss, calc_entropy, consistency_loss, compute_vat_loss

soft = nn.Softmax() if torch.__version__[2]=="1" else nn.Softmax(dim=1)
logsoft = nn.LogSoftmax() if torch.__version__[2]=="1" else nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss(size_average=True)
nll_loss_batch = nn.NLLLoss(reduce=False, size_average=True)


def training_loop(net, optimizer, trainloader_source, trainloader_target,
                  config, epoch):

    net.train()
    total_l, correct_l, total_u, correct_u = 0, 0, 0, 0
    loss_ce_src_av, loss_ent_av, loss_vat_av = 0, 0, 0
    loss_src_joint_av, loss_trg_joint_av, loss_feat_adv_av = 0, 0, 0

    for _ in range(config["num_iter_per_epoch"]):

        net.inds = [np.random.randint(0, high=config['nb_e']),
                    np.random.randint(0, high=config['nb_cc']),
                    np.random.randint(0, high=config['nb_jc'])]

        optimizer.zero_grad()

        # targs_u is used for FOR POST MORTEM analysis only:
        inps_l, targs_l = iter(trainloader_source).next()
        inps_u, targs_u = iter(trainloader_target).next()
        inps_l, targs_l = Variable(inps_l.cuda()), Variable(targs_l.cuda())
        inps_u, targs_u = Variable(inps_u.cuda()), Variable(targs_u.cuda())

        """
        optimize encoder only
        """
        # Compute adversarial loss for labeled samples.
        _, joint_prob_l = net(inps_l, state = "encoder")
        targs_l_fake = targs_l + config['nb_class']
        loss_src_feat_adv = nll_loss(logsoft(joint_prob_l), targs_l_fake)

        # Compute adversarial loss for unlabeled samples.
        estimate_u, joint_prob_u = net(inps_u, state = "encoder")
        _, targs_u_fake = torch.max(estimate_u.data, 1) # targs_u_fake is in {0, ..., K-1}
        loss_trg_feat_adv = nll_loss(logsoft(joint_prob_u), targs_u_fake)

        # Compute grads.
        loss_feat_adv = config['lambda_src_feat_adv'] * loss_src_feat_adv + \
                          config['lambda_trg_feat_adv'] * loss_trg_feat_adv
        loss_feat_adv.backward()

        loss_feat_adv_av += (loss_src_feat_adv + loss_trg_feat_adv).data.cpu().numpy()

        """
        optimize class classifier
        """
        class_prob_l, joint_prob_l = net(inps_l, state = "all")
        class_prob_u, joint_prob_u = net(inps_u, state = "all")

        # Compute cross-entropy loss.
        class_prob_l = logsoft(class_prob_l)
        loss_ce_src = nll_loss(class_prob_l, targs_l)
        loss_ce_src_av += loss_ce_src.data.cpu().numpy()

        # Compute entropy loss.
        loss_ent = calc_entropy(class_prob_u)
        loss_ent_av += loss_ent.data.cpu().numpy()

        # Compute vat loss.
        loss_vat = compute_vat_loss(net, inps_l, inps_u, config)
        loss_vat_av += loss_vat.data.cpu().numpy()

        # Compute grads.
        loss_all = loss_ce_src + config['lambda_ent'] * loss_ent + loss_vat
        loss_all.backward()

        """
        consistency loss between joint classifier and class classifier
        """
        # Compute consistency loss for seperate halves of the joint classifier.
        loss_consistency_left = consistency_loss(net, inps_l, targs_l, inps_u, nll_loss, config, is_left_half=True)
        loss_consistency_left.backward()

        loss_consistency_right = consistency_loss(net, inps_l, targs_l, inps_u, nll_loss, config, is_left_half=False)
        loss_consistency_right.backward()

        # Compute consistency loss for both halves of the joint classifier.
        class_prob_l, joint_prob_l  = net(inps_l, state = "only_joint")
        loss_src_joint = nll_loss(logsoft(joint_prob_l), targs_l)

        estimate_u, joint_prob_u  = net(inps_u, state = "only_joint")
        _, targs_u_pseudo = torch.max(estimate_u.data, 1)
        targs_u_pseudo += config['nb_class'] # targs_u_pseudo is in {K, ..., 2K-1}
        loss_trg_joint = nll_loss(logsoft(joint_prob_u), targs_u_pseudo)

        loss_src_joint_av += loss_src_joint.data.cpu().numpy()
        loss_trg_joint_av += loss_trg_joint.data.cpu().numpy()

        # Compute grads.
        loss_only_joint = config['lambda_src_joint'] * loss_src_joint + config['lambda_trg_joint'] * loss_trg_joint
        loss_only_joint.backward()

        # Update the optimizer.
        optimizer.step()
        optimizer.zero_grad()

        _, predicted_l = torch.max(class_prob_l.data, 1)
        total_l += targs_l.size(0)
        correct_l += predicted_l.eq(targs_l.data).cpu().sum().numpy()

        _, predicted_u = torch.max(class_prob_u.data, 1)
        total_u += targs_u.size(0)
        correct_u += predicted_u.eq(targs_u.data).cpu().sum().numpy()

    config['acc_source_train'] = 100.0*correct_l/total_l
    config['acc_target_train'] = 100.0*correct_u/total_u
    config['loss_feat_adv'] = loss_feat_adv_av/config["num_iter_per_epoch"]
    config['loss_ce'] = loss_ce_src_av/config["num_iter_per_epoch"]
    config['loss_ent'] = loss_ent_av/config["num_iter_per_epoch"]
    config['loss_vat'] = loss_vat_av/config["num_iter_per_epoch"]
    config['loss_src_joint'] = loss_src_joint_av/config["num_iter_per_epoch"]
    config['loss_trg_joint'] = loss_trg_joint_av/config["num_iter_per_epoch"]

    return config


def training_loop_source_only(net, optimizer, trainloader_source, config, epoch):

    net.train()
    total_l, correct_l, loss_ce_src_av = 0, 0, 0

    for _ in range(config["num_iter_per_epoch"]):

        net.inds = [np.random.randint(0, high=config['nb_e']),
                    np.random.randint(0, high=config['nb_cc']),
                    np.random.randint(0, high=config['nb_jc'])]

        optimizer.zero_grad()
        inps_l, targs_l = iter(trainloader_source).next()
        inps_l, targs_l = Variable(inps_l.cuda()), Variable(targs_l.cuda())

        class_prob_l, _ = net(inps_l, state = "all")

        # Compute cross-entropy loss.
        class_prob_l = logsoft(class_prob_l)
        loss_ce_src = nll_loss(class_prob_l, targs_l)
        loss_ce_src_av += loss_ce_src.data.cpu().numpy()

        # Compute grads.
        loss_all = loss_ce_src
        loss_all.backward()

        # Update the optimizer.
        optimizer.step()
        optimizer.zero_grad()

        _, predicted_l = torch.max(class_prob_l.data, 1)
        total_l += targs_l.size(0)
        correct_l += predicted_l.eq(targs_l.data).cpu().sum().numpy()

    config['acc_source_train'] = 100.0*correct_l/total_l
    config['loss_ce'] = loss_ce_src_av/config["num_iter_per_epoch"]

    return config


def testing_loop(net, testloader, config, is_source):
    net.eval()
    correct_class, correct_joint, total = 0, 0, 0
    for inputs, targets in testloader:
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        class_prob, joint_prob = net(inputs)

        joint_prob_ = joint_prob[:, 0:config['nb_class']] + joint_prob[:, config['nb_class']:]
        _, predicted_joint = torch.max(joint_prob_.data, 1)
        acc_batch = predicted_joint.eq(targets.data).cpu()
        correct_joint += acc_batch.sum().numpy()

        _, predicted_class = torch.max(class_prob.data, 1)
        acc_batch = predicted_class.eq(targets.data).cpu()
        correct_class += acc_batch.sum().numpy()

        total += targets.size(0)

    acc_class = 100.0*correct_class/total
    acc_joint = 100.0*correct_joint/total

    if is_source:
        if acc_class > config['acc_source_class_valid']:
            config['acc_source_class_valid'] = acc_class

        if acc_joint > config['acc_source_joint_valid']:
             config['acc_source_joint_valid'] = acc_joint

    else:
        if acc_class > config['acc_target_class_valid']:
            config['acc_target_class_valid'] = acc_class

        if acc_joint > config['acc_target_joint_valid']:
             config['acc_target_joint_valid'] = acc_joint

    return config
