import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
soft = nn.Softmax() if torch.__version__[2]=="1" else nn.Softmax(dim=1)
logsoft = nn.LogSoftmax() if torch.__version__[2]=="1" else nn.LogSoftmax(dim=1)

def calc_entropy(vecs):
    loss_ent = -torch.mean(torch.sum(soft(vecs)*torch.log(soft(vecs)+10**-8), 1))
    return loss_ent

def get_normalized_vector(vectors):
    vectors /= (1e-12 + torch.max(torch.abs(vectors))) ##l_inf normalization
    vectors /= (1e-6 + torch.sum(vectors**2.0, dim=1, keepdim=True))**0.5
    return vectors

def kl_with_logit(q_logit, p_logit):
    qlogq = torch.mean(torch.sum(soft(q_logit) * logsoft(q_logit), 1))
    qlogp = torch.mean(torch.sum(soft(q_logit) * logsoft(p_logit), 1))
    return qlogq - qlogp

def vat_perturb(x, net, logit_p, epsilon, xi= 10**-6, use_half=True):
    d = torch.FloatTensor(np.random.normal(size=x.size())).cuda()
    d = get_normalized_vector(d)

    r = xi * d
    r = Variable(r, requires_grad = True)

    logit_m, _  = net(x + r, state = "all")

    dist = kl_with_logit(logit_p, logit_m)
    g = torch.autograd.grad(dist, r, create_graph=True)[0].data
    g = get_normalized_vector(g)
    r_adv = epsilon * g
    logit_m = logit_m.detach()
    r = r.detach()
    dist = dist.detach()
    return r_adv

def vat_loss(x, net, epsilon, xi= 10**-6, use_half=True):
    logit_p, _ = net(x, state = "all")
    logit_p = logit_p.detach()
    r_vadv = vat_perturb(x, net, logit_p, epsilon, xi)

    logit_m, _ = net(x + Variable(r_vadv), state = "all")

    loss_vat = kl_with_logit(logit_p, logit_m)
    return loss_vat

def deactivate_mom(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0

def activate_mom(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.01

def compute_vat_loss(net, inps_l, inps_u, config):
    net.apply(deactivate_mom)

    loss_vat_trg = vat_loss(inps_u, net, config['eps_vat'])

    if config['lambda_src_vat'] > 10**-5:
        loss_vat_src = vat_loss(inps_l, net, config['eps_vat'])
        loss_vat = config['lambda_src_vat'] * loss_vat_src + config['lambda_trg_vat'] * loss_vat_trg
    else:
        loss_vat = config['lambda_trg_vat'] * loss_vat_trg

    net.apply(activate_mom)

    return loss_vat

def consistency_loss(net, inps_l, targs_l, inps_u, nll_loss, config, is_left_half):
    _, joint_prob_l = net(inps_l, state = "only_joint")
    if is_left_half:
        joint_prob_l = joint_prob_l[:, :config['nb_class']]
    else:
        joint_prob_l = joint_prob_l[:, config['nb_class']:]

    loss_consistency_src = nll_loss(logsoft(joint_prob_l), targs_l)

    estimate_u, joint_prob_u = net(inps_u, state = "only_joint")
    if is_left_half:
        joint_prob_u = joint_prob_u[:, :config['nb_class']]
    else:
        joint_prob_u = joint_prob_u[:, config['nb_class']:]

    _, targs_u = torch.max(estimate_u.data, 1)
    loss_consistency_trg = nll_loss(logsoft(joint_prob_u), targs_u)

    loss_consistency = config['lambda_consis_src'] * loss_consistency_src + \
                       config['lambda_consis_trg'] * loss_consistency_trg

    return loss_consistency