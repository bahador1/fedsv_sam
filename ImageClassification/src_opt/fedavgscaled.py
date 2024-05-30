# Version 2.0
import sys, os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src_opt.utils.options import args_parser
from src_opt.utils.update import LocalUpdate, test_inference
from src_opt.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src_opt.utils.tools import get_dataset, average_weights, exp_details, add_gradient_noise, add_random_gradient, get_noiseword

args = args_parser()
exp_details(args)

def solver():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('')
    logger = SummaryWriter('../logs')

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != None else 'cpu')
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

    # Set the model to train and send it to device.1
    global_model = global_model.to(args.device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()
    original_weights = copy.copy(global_weights)
    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2
    # attack_epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 91] # attacks occurs in these epochs 
    # attack_epoch = [5,6,7,8,9,10,11,12,13, 14] # attacks occurs in these epochs 
    # attack_epoch = [3, 4, 7]
    # attack_epoch = np.random.choice(range(100), 10, replace=False)
    # attack_epoch = [15, 98, 65, 53, 6, 28, 18, 37, 41, 3]#random.sample(range(99), 10)
    attack_epoch = [30,70]
    targeted_clients = [ 5]
    accuracy_list = []
    mal_local_weights ={}
    m=args.num_users
    idxs_users = np.arange(0, m)
    
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')
        global_model.train()
        # m=args.num_users
        # idxs_users = np.arange(1, m+1)
        # print("##########################################################", idxs_users)
        # Then, within your training loop:
        # idxs_users = [i % total_users for i in range(current_start_index, current_start_index + users_per_epoch)]
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # Example: Cycle through users in fixed increments per epoch
        # users_per_epoch = max(int(args.frac * args.num_users), 1)
        # total_users = args.num_users
        # current_start_index = 0

        # # Then, within your training loop:
        # idxs_users = [i % total_users for i in range(current_start_index, current_start_index + users_per_epoch)]
        # # current_start_index = (current_start_index + users_per_epoch) % total_users

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model).to(args.device), global_round=epoch)
            
            # if idx in [0, 1] and epoch==9:
            #     print(f"Client {idx} weights before noise:")
            #     for key in w.keys():
                    
            #         parameter_key = 'layer1.0.weight'
            #         weights = w[parameter_key]
            #         # print(f"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000{parameter_key}: {w[parameter_key].detach().cpu().numpy()}")
            #         torch.save(weights, '../save/objects/round9_before_noise.pt')
             
            # if epoch in attack_epoch and idx in targeted_clients:
            if (epoch in attack_epoch) and (idx in targeted_clients):
                if args.noise == 9:
                    print(f"now the attackers are: {idx} and the epoch is {epoch}")
                    for key in w.keys():
                        w[key] = w[key] * 100
                

                # if idx in [0, 1]:
                #     print(f"Client {idx} weights after noise:")
                #     for key in w.keys():
                #         parameter_key = 'layer1.0.weight'
                #         weights = w[parameter_key]
                #         torch.save(weights, '../save/objects/round9_after_noise.pt')
                #         # print(f"afterrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr{parameter_key}: {w[parameter_key].detach().cpu().numpy()}") 

                
                # w = add_gradient_noise(args, w)
                # print("idx", idx)

            local_weights.append(copy.deepcopy(w))
            
            local_losses.append(copy.deepcopy(loss))


            #     print(w_weights.items().weights)

            #     for params in w_weights:
            #         print(params.items())
            #         if params != params:
            #             print(":/ hey pomogranate come to my bed:/")# why it doesnt wor           

        # update global weights
        # local_weights.append(copy.deepcopy(global_weights))
        # Add Gradient Noise
        # print(local_weights[0])
        
    # Step 1: Collect local weights, possibly adding noise
        global_weights = average_weights(local_weights) #

        # print("global_weight_fedavg ---->", global_weights)
        # torch.save(global_weights, '../save/objects/global_weight_fedavg_1attacker_10rounds.pt')
        # if epoch < 75:
        #     global_weights = average_weights(local_weights)
        # elif epoch < 100:
        #     shapley = np.ones(m)
        #     shapley = F.softmax(torch.tensor(shapley), dim=0)
        #     global_weights = SVAtt_weights(local_weights, shapley, original_weights, 0.1, epoch)

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
        
        #  torch.save(model, f"C:\Users\samin\OneDrive - Texas Tech University\Desktop\ShapleyFL_Original\ImageClassification\save\edavg_{iter}.pt")

        # if (epoch+1) == args.epochs:
        #     print("GLOBAL WEIGHTS================================", global_weights)
        #     torch.save(global_weights, '../save/objects/global_weight_fedavg_1attacker_10rounds.pt')
        #     print("Saved!")


        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        allAcc_list.append(test_acc)
        print(" \nglobal accuracy:{:.2f}%".format(100 * test_acc))
        # accuracy_file_path = 'path/to/your/directory/global_accuracy_results.txt'

        # Open the file in append mode ('a') to add the new accuracy value
        # with open(accuracy_file_path, 'a') as f:
        # # Write the global accuracy to the file
        # f.write("Global accuracy: {:.2f}%\n".format(100 * test_acc))

# print(f'Global accuracy has been saved to {accuracy_file_path}')


    #draw(args.epochs, allAcc_list, "FedAvg 10 100")
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    accuracy_list.append(test_acc)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    directory = './save/objects'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = f'{directory}/{args.dataset}_{args.model}_epoch[{args.epochs}]_Clients[{args.num_users}]_{args.noise}_fedavg_1attackrounds_iid_2attacker_scaling.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_accuracy': allAcc_list}, f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    return test_acc, train_accuracy[-1], allAcc_list

def show_avg(list):
    ans = []
    ans.append(np.mean(list[17:22]))
    ans.append(np.mean(list[37:42]))
    ans.append(np.mean(list[57:62]))
    ans.append(np.mean(list[77:82]))
    ans.append(np.mean(list[95:]))
    print(ans)

if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    repeat = 1
    noise = args.noise
    NoiseWord = get_noiseword()
    for _ in range(repeat):
        print("|---- Repetition {} ----|".format(_ + 1))
        test, train, acc_list = solver()
        test_acc += test
        train_acc += train
        # show_avg(acc_list)

        directory = "./save_opt/{}".format(NoiseWord[noise])
        filename = "Fedavg_sam_{}_cnn_E{}_N{}_repeat{}_{}.txt".format(args.dataset, args.epochs,
                                                                                       args.noiselevel, repeat, args.device)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # path = '../save_opt/{}/AFedSV_{}_cnn_E{}_N{}_gamma{}_repeat{}_{}.txt'.format(NoiseWord[noise], args.dataset, args.epochs,
                                          
        # path = '../save_opt/{}/FedAvg_{}_cnn_E{}_N{}_repeat{}_{}.txt'.format(NoiseWord[noise], args.dataset, args.epochs,  args.noiselevel, repeat, args.device)
        f = open(directory+filename, "a+")
        f.writelines("Repetition [%d] : [%s]\n" % (_ + 1, ', '.join(["%.4f" % w for w in acc_list])))
        f.flush()
        f.close()
    print('|---------------------------------')
    print("|---- Train Accuracy: {:.2f}%".format(100 * (train_acc / repeat)))
    print("|---- Test Accuracy: {:.2f}%".format(100 * (test_acc / repeat)))