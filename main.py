import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import torch.nn.functional as F
from torchsummary import summary
import tensorflow as tf

from model import *
from utils import *
from utils import fedavg_aggregate
import copy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=2, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=5, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=10, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args


def triplate_mean_lr(prev, pres, serv,lr):
  # print(prev)
  # print(pres)
  # print(serv)
  xm = tf.math.reduce_mean(prev)
  ym = tf.math.reduce_mean(pres)
  zm = tf.math.reduce_mean(serv)
  return lr * (tf.math.divide(tf.math.reduce_mean([xm,ym,zm]),tf.math.add(xm,ym,zm)))
  # pass


def contrastive_loss(z_prev,z_present,z_serv,temperature=0.5,margin = 0.2, lr=1e-3):
    #  z_prev = multivariate_norm(z_prev, z_serv)
    #  z_present = multivariate_norm(z_present, z_serv)
    # print(z_serv)
     z_prev = tf.math.l2_normalize(z_prev)
     z_present = tf.math.l2_normalize(z_present)
     dis = tf.sqrt(tf.reduce_sum(tf.square(z_present - z_prev)))
     dis = tf.sqrt(tf.math.maximum(dis, tf.keras.backend.epsilon())) # tf.keras.backend.epsilon()==1e-07
     sqpres = tf.math.square(z_present)
     sqMar = tf.math.square(tf.math.maximum(margin - z_present,0))
     loss = tf.math.reduce_mean(z_prev*sqpres + (1-z_prev)*sqMar)
     loss = loss - (loss*lr)
     lr = triplate_mean_lr(z_prev,z_present,z_serv,lr)
     srv = tf.sqrt(tf.reduce_sum(tf.square(z_serv)))
     #srv = tf.math.l2_normalize(srv)
     srv = tf.math.log(srv)
     return loss/srv

def sum_scaled_weights(scaled_weight_list, length_ratio):
    """
    Compute the federated averaging of model weights.
    
    Args:
    - scaled_weight_list: List of client weight updates.
    - length_ratio: Weighting factor for each client.

    Returns:
    - Averaged model weights.
    """
    avg_grad = []
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)  # Compute mean of gradients
        avg_grad.append(layer_mean)
    return avg_grad

def set_client_weights(server, client):
    """
    Set client weights to match the global model before local training.

    Args:
    - server: The global server model.
    - client: A client model.

    Returns:
    - Updated client model.
    """
    server_weights = [layer.get_weights() for layer in server.layers]

    for i, clayer in enumerate(client.layers):
        clayer.set_weights(server_weights[i])
    
    return client


def set_server_weights(server, client):
    """
    Aggregate client weights into the global model after training.

    Args:
    - server: The global server model.
    - client: A trained client model.

    Returns:
    - Updated server model.
    """
    client_weights = [clayer.get_weights() for clayer in client.layers]

    for i, slayer in enumerate(server.layers):
        slayer.set_weights(client_weights[i])
    
    return server





def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    # if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
    #     n_classes = 10
    if args.dataset == 'mnist':
        global_model = SimpleCNNMNIST(input_dim=16*4*4, hidden_dims=[120, 84], output_dim=10)
    elif args.dataset == 'cifar10':
        global_model = SimpleCNN(input_dim=16*5*5, hidden_dims=[120, 84], output_dim=10)
        n_classes = 10
    # else:
    # raise ValueError("Unsupported dataset")
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2

    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                if args.dataset == 'mnist':
                    net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)  # ✅ Use MNIST model
                else:
                    net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)  # ✅ Use CIFAR-10 model
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, optimizer, epochs, device="cuda"):
    """
    Trains the given model on a client's local dataset.

    Args:
    - model: PyTorch model instance (client's model).
    - train_loader: DataLoader for client's training data.
    - optimizer: Optimizer for training.
    - epochs: Number of local training epochs.
    - device: Computation device (default: "cuda").

    Returns:
    - Trained model (weights are updated in-place).
    """
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()

            # Extract features and logits from model
            z_prev, z_present, logits = model(x)

            # Compute classification loss
            loss_ce = criterion(logits, target)

            # Compute contrastive loss
            loss_cl = contrastive_loss(z_prev, z_present, z_present)

            # Total loss
            loss = loss_ce + loss_cl

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


# FedAvg Training Function
def train_net_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    """
    Train a local model using FedAvg, incorporating contrastive loss.

    """
    net = net.to(device)
    logger.info(f'Training client {net_id}')

    # Choose optimizer
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        net.train()
        epoch_loss = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass - Extract embeddings and output logits
            z_prev, z_present, out = net(x)

            # Compute contrastive loss
            loss_cl = contrastive_loss(z_prev, z_present, z_present)

            # Compute classification loss
            loss_ce = criterion(out, target)

            # Total loss
            loss = loss_ce + loss_cl

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        logger.info(f'Epoch {epoch}: Loss {avg_loss:.4f}')

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, _ = compute_accuracy(net, test_dataloader, device=device)

    return train_acc, test_acc, net.state_dict()

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    local_weight_list = []

    if global_model:
        global_model.cuda()
    
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            # trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        # device=device)
            # net = set_client_weights(global_model, net)

            net.load_state_dict(global_model.state_dict())

            trainacc, testacc, client_weights = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)

            local_weight_list.append(client_weights)


        # elif args.alg == 'fedprox':
        #     trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
        #                                           args.optimizer, args.mu, args, device=device)
        # elif args.alg == 'moon':
        #     prev_models=[]
        #     for i in range(len(prev_model_pool)):
        #         prev_models.append(prev_model_pool[i][net_id])
        #     trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
        #                                           args.optimizer, args.mu, args.temperature, args, round, device=device)

        # elif args.alg == 'local_training':
            # trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
            #                               device=device)
        logger.info("Client net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)

    avg_acc /= args.n_parties
    
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    #     logger.info("std acc %f" % np.std(acc_list))
    if args.alg == 'fedavg':
        avg_weight = sum_scaled_weights(local_weight_list)
        global_model.load_state_dict(avg_weight)
    
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')


    elif args.alg == "fedavg":
        global_model = ModelFedCon(args.base_model, args.out_dim, args.n_classes).to(device)
    
        for round in range(args.comm_rounds):
            print(f"Communication Round {round+1}")

            local_models = []
            num_samples = []

            for client in clients:
                client_model = copy.deepcopy(global_model)
                optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr)

                train(client_model, client["train_loader"], optimizer, args.epochs)

                local_models.append(client_model)
                num_samples.append(len(client["train_loader"].dataset))

        # Aggregate using FedAvg
            global_model = fedavg_aggregate(global_model, local_models, num_samples)

    # Final model evaluation
        test_accuracy, test_loss = compute_accuracy(global_model, test_loader)
        print(f"Final Test Accuracy: {test_accuracy}")


    # elif args.alg == 'fedavg':
    #     global_model = SimpleCNN(input_dim=16 * 5 * 5, hidden_dims=[120, 84], output_dim=10).to(device)
           
    #     for round in range(n_comm_rounds):
    #         logger.info("in comm round:" + str(round))
    #         party_list_this_round = party_list_rounds[round]

    #         global_w = global_model.state_dict()
    #         if args.server_momentum:
#                 old_w = copy.deepcopy(global_model.state_dict())

#             nets_this_round = {k: nets[k] for k in party_list_this_round}
#             for net in nets_this_round.values():
#                 net.load_state_dict(global_w)
#             # Ensure model is instantiated before calling local_train_net
#             # if args.model == "simple-cnn":
#             # global_model = SimpleCNN(input_dim=16 * 5 * 5, hidden_dims=[120, 84], output_dim=10).to(device)
#             # else:  raise ValueError(f"Unsupported model: {args.model}")

# # Proceed with training
#             local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model=global_model, device=device)

#             # local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

#             total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            # for net_id, net in enumerate(nets_this_round.values()):
            #     net_para = net.state_dict()
            #     if net_id == 0:
            #         for key in net_para:
            #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            #     else:
            #         for key in net_para:
            #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            # if args.server_momentum:
            #     delta_w = copy.deepcopy(global_w)
            #     for key in delta_w:
            #         delta_w[key] = old_w[key] - global_w[key]
            #         moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
            #         global_w[key] = old_w[key] - moment_v[key]


            # global_model.load_state_dict(global_w)

            # #logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl))
            # global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            # mkdirs(args.modeldir+'fedavg/')
            # global_model.to('cpu')

            # torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            # torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')

