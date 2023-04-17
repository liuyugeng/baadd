import os
import time
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from utils import *
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='results', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--poisoned_dataset', action='store_true')
    parser.add_argument('--doorping', action='store_true')
    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--ori', type=float, default=1.0)
    parser.add_argument('--layer', type=int, default=-2)
    parser.add_argument('--portion', type=float, default=0.01)
    parser.add_argument('--backdoor_size', type=int, default=2)
    parser.add_argument('--support_dataset', default=None, type=str)
    parser.add_argument('--trigger_label', type=int, default=0)
    parser.add_argument('--device_id', type=str, default="0", help='device id, -1 is cpu')
    parser.add_argument('--model_init', type=str, default="imagenet-pretrained")
    parser.add_argument('--invisible', action='store_true')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=10)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    args.trojann_trigger = False
    args.invisible_trigger = False

    name = args.model + '_' + args.dataset + '_' + str(args.ipc) + 'ipc'
    if args.poisoned_dataset:
        name += '_poisoned_portion_' + str(args.portion) + '_size_' + str(args.backdoor_size) + '_ori_' + str(args.ori)
    if args.doorping:
        name += '_doorping_portion_' + str(args.portion) + '_size_' + str(args.backdoor_size) + '_ori_' + str(args.ori)
    if args.invisible:
        name += '_invisible_portion_' + str(args.portion) + '_size_' + str(args.backdoor_size) + '_ori_' + str(args.ori)
    if args.eval_mode == 'M':
        name += '_M'
    if args.topk != 1:
        name += '_topk_' + str(args.topk)
    if args.alpha != 10:
        name += '_alpha_' + str(args.alpha)
    args.save_path = os.path.join(args.save_path, args.method, name)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 50).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    args.clean = True
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args)

    length = int(args.ori*len(dst_train))
    rest = len(dst_train) - length

    dst_train, _ = torch.utils.data.random_split(dst_train, [length,rest])

    args.clean = False
    if not args.poisoned_dataset and not args.doorping and not args.invisible:
        args.poisoned_dataset = True
        _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path, args)
        args.poisoned_dataset = False
    else:
        _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path, args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # for data, label in testloader:
    #     print(label)
    #     break

    # for data, label in testloader_trigger:
    #     print(label)
    #     break

    # exit()


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    if args.doorping:
        trojann_perm = np.random.permutation(len(dst_train))[0: int(len(dst_train) * args.portion)]
        input_size = (im_size[0], im_size[1], channel)
        trigger_loc = (im_size[0]-1-args.backdoor_size, im_size[0]-1)
        args.init_trigger = np.zeros(input_size)
        init_backdoor = np.random.randint(1, 256,(args.backdoor_size, args.backdoor_size, channel))
        args.init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        args.mask = torch.FloatTensor(np.float32(args.init_trigger > 0).transpose((2, 0, 1))).to(args.device)
        if channel == 1:
            args.init_trigger = np.squeeze(args.init_trigger)
        args.init_trigger = Image.fromarray(args.init_trigger.astype(np.uint8))
        args.init_trigger = transform(args.init_trigger)
        args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
        args.init_trigger = args.init_trigger.requires_grad_() # size 1*3x32x32

    if args.invisible:
        trojann_perm = np.random.permutation(len(dst_train))[0: int(len(dst_train) * args.portion)]

        for img, label in dst_test:
            if label == args.trigger_label:
                args.init_trigger = img
                break

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
        args.init_trigger = args.init_trigger.requires_grad_()

        input_size = (im_size[0], im_size[1], channel)
        args.black = np.zeros(input_size)
        args.black = transform(args.black)
        args.black = args.black.unsqueeze(0).to(args.device, non_blocking=True)


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] #size 1x3x32x32
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]

        # for i, lab in enumerate(labels_all):
        #     indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        if args.doorping:
            # put the tigger into images
            images_all[trojann_perm] = images_all[trojann_perm]*(1-args.mask) + args.mask*args.init_trigger[0]
            labels_all[trojann_perm] = args.trigger_label

        if args.invisible:
            images_all[trojann_perm] = args.init_trigger[0] + images_all[trojann_perm]
            labels_all[trojann_perm] = args.trigger_label


        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.test_model:
            path = os.path.join("results/DC", name, "exp{}".format(str(exp)), "vis_%s_%s_%s_%dipc_exp%d_iter%d"%(args.method, args.dataset, args.model, args.ipc, exp, 1000))
            path = os.path.join("../../MyNeuralCleanse", 'pth', '{}_{}_{}'.format(args.model.lower()[0], args.method.lower(), args.dataset.lower()))
            if args.dataset == "CIFAR10" or args.dataset == "STL10":
                path = path[:-2]
            if args.dataset == 'FashionMNIST':
                path = path[:-len(args.dataset)] + 'fmnist'
            image_syn = torch.load(path + "_img.pth")
            args.init_trigger = torch.load(path + "_trigger.pth")
            print(args.init_trigger[0][:,29:31,29:31])
            model_eval_pool_all = ['VGG11', 'ConvNet', 'AlexNet']
            model_eval_pool = list(set(model_eval_pool_all).difference(set(model_eval_pool)))
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                if args.dsa or args.dc_aug_param['strategy'] != 'none':
                    args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                else:
                    args.epoch_eval_train = 300

                accs = []
                accs_trigger = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    _, acc_train, acc_test, acc_test_trigger = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, testloader_trigger, args)
                    accs.append(acc_test)
                    accs_trigger.append(acc_test_trigger)

                print('Evaluate %d random %s, clean mean = %.4f clean std = %.4f, trigger mean = %.4f trigger std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs), np.mean(accs_trigger), np.std(accs_trigger)))


            # path = os.path.join("results/DC", name)
            # for file in os.listdir(path):
            #     print(file)
            #     file_path = os.path.join(path, file)
            #     if os.path.isfile(file_path) or 'exp' in file:
            #         continue
            #     trigger_path = os.path.join(path, "exp{}".format(str(exp)), "vis_%s_%s_%s_%dipc_exp%d_iter%d_trigger.pth"%(args.method, args.dataset, args.model, args.ipc, exp, int(file)))
            #     args.init_trigger = torch.load(trigger_path)
                
            #     accs_trigger = []
            #     for model in os.listdir(file_path):
            #         for model_eval in model_eval_pool:
            #             model_path = os.path.join(file_path, model)
            #             net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            #             net_eval.load_state_dict(torch.load(model_path))
            #             args.trojann_trigger = True
            #             criterion = nn.CrossEntropyLoss().to(args.device)
            #             _, acc_test_trigger = epoch('test', testloader, net_eval, None, criterion, args, aug = False)

            #             accs_trigger.append(acc_test_trigger)

            #     print('trigger mean = %.4f trigger std = %.4f\n-------------------------'%(np.mean(accs_trigger), np.std(accs_trigger)))

            continue

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    accs_trigger = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test, acc_test_trigger = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, testloader_trigger, args)
                        accs.append(acc_test)
                        accs_trigger.append(acc_test_trigger)

                        model_path = os.path.join(args.save_path, str(it))
                        if not os.path.exists(model_path):
                            os.mkdir(model_path)
                        model_path = os.path.join(model_path, 'model_' + str(it_eval) + '.pth')
                        torch.save(net_eval.state_dict(), model_path)

                    print('Evaluate %d random %s, clean mean = %.4f clean std = %.4f, trigger mean = %.4f trigger std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs), np.mean(accs_trigger), np.std(accs_trigger)))
                    
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                exp_idx = os.path.join(args.save_path, 'exp%d'%exp)
                if not os.path.exists(exp_idx):
                    os.makedirs(exp_idx)

                save_name = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                save_name_2 = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.pth'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                torch.save(image_syn_vis, save_name_2)
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

                if args.doorping or args.invisible:
                    save_trigger_name = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d_trigger.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                    save_image(args.init_trigger[0], save_trigger_name)
                    save_trigger_name_2 = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d_trigger.pth'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                    torch.save(args.init_trigger, save_trigger_name_2)

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

                if args.doorping:
                    args.init_trigger = update_trigger(net, args.init_trigger, args.layer, args.device, args.mask, args.topk, args.alpha)
                    images_all[trojann_perm] = images_all[trojann_perm]*(1-args.mask) + args.mask*args.init_trigger[0]

                if args.invisible:
                    args.init_trigger = update_inv_trigger(net, args.init_trigger, args.layer, args.device, std, args.black)
                    images_all[trojann_perm] = images_all[trojann_perm] + args.init_trigger[0]

            loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))

    if not args.test_model:
        print('\n==================== Final Results ====================\n')
        for key in model_eval_pool:
            accs = accs_all_exps[key]
            print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

if __name__ == '__main__':
    set_random_seeds(42)
    main()


