# Version 2.0
import sys, os

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import math

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src_opt.utils.options import args_parser
from src_opt.utils.update import LocalUpdate, test_inference
from src_opt.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src_opt.utils.Shapley import Shapley
from src_opt.utils.CEXPIX import arms_selection
from src_opt.utils.tools import get_dataset, average_weights, exp_details, avgSV_baseline, softmax, unbiased_selection, \
    add_gradient_noise, add_random_gradient, get_noiseword

args = args_parser()
exp_details(args)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def solver(gamma):
    start_time = time.time()
    # define paths
    logger = SummaryWriter('../logs')

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != None else 'cpu')

    # load dataset and user groups
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model = global_model.to(args.device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()
    original_weights = copy.copy(global_weights)
    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2
    init_acc = 0

    # attack
    attack_epochs = [21, 30] 
    targeted_clients =[2]
    
    global_shapley = np.array([0.5 for _ in range(args.num_users)])
    cnt_clients = np.ones(args.num_users)
    # The prior probability of each arm been selected in one round
    probabilities = np.array([args.frac for _ in range(args.num_users)])
    normal_shapley = np.array([1 / args.num_users for _ in range(args.num_users)])

    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = unbiased_selection(probabilities)

        # print("the number os users is ", len(idxs_users))
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model).to(args.device), global_round=epoch)
            if(epoch in attack_epochs) and (idx in targeted_clients):
                    if args.noise == 9:
                        print(f"now the attackers are: {idx} and the epoch is {epoch}")
                        for key in w.keys():
                            w[key] = w[key] * 100
                    if args.noise == 10:
                            for key in w.keys():
                                noise = torch.tensor(np.random.normal(0, args.noiselevel, w[key].shape))
                                noise = noise.to(torch.float32)
                                noise = noise.to(args.device)
                                # print("original weight = ", w[i][key])
                                if "running_mean" or "num_batches_tracked" or "num_batches_" in key:
                                    break
                                w[key] += noise

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        Fed_sv = Shapley(local_weights, args, global_model, valid_dataset, init_acc)
        shapley = Fed_sv.eval_ccshap_stratified(50)
        # update estimated Shapley value

        weight_shapley = softmax(shapley)
        # print(f"wegiht shapley is {weight_shapley}")
        # Add Gradient Noise
        # local_weights = add_gradient_noise(args, local_weights, idxs_users)
        global_weights = avgSV_baseline(local_weights, weight_shapley, original_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        original_weights = copy.copy(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        allAcc_list.append(test_acc)
        # print(type(allAcc_list))
        print(" \nglobal accuracy:{:.2f}%".format(100 * test_acc))
        init_acc = test_acc

    # draw(args.epochs, allAcc_list, "FedAvg 10 100")
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    return test_acc, train_accuracy[-1], allAcc_list


def show_avg(acclist):
    ans = []
    ans.append(np.mean(acclist[17:22]))
    ans.append(np.mean(acclist[37:42]))
    ans.append(np.mean(acclist[57:62]))
    ans.append(np.mean(acclist[77:82]))
    ans.append(np.mean(acclist[95:]))
    print(ans)


if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    repeat = 1
    gamma = args.gamma_sv
    noise = args.noise
    NoiseWord = get_noiseword()
    for _ in range(repeat):
        print("|---- Repetition {} ----|".format(_ + 1))
        test, train, acc_list = solver(gamma)
        test_acc += test
        train_acc += train
        show_avg(acc_list)
        path = './save_opt/{}/FedSV_{}_cnn_E{}_N{}_gamma{}_repeat{}_{}.txt'.format(NoiseWord[noise], args.dataset, args.epochs,
                                                                                       args.noiselevel, gamma, repeat, args.device)
        f = open(path, "a+")
        f.writelines("Repetition [%d] : [%s]\n" % (_ + 1, ', '.join(["%.4f" % w for w in acc_list])))
        f.flush()
        f.close()
    print('|---------------------------------')
    print("|---- Train Accuracy: {:.2f}%".format(100 * (train_acc / repeat)))
    print("|---- Test Accuracy: {:.2f}%".format(100 * (test_acc / repeat)))
