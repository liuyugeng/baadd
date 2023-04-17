import logging
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from basics import task_loss, final_objective_loss, evaluate_steps
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results
from datasets import *
from PIL import Image
from torchvision import transforms
from defined_models import *


def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]


class Trainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.num_data_steps = state.distill_steps  # how much data we have 10
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run 10*3
        self.num_per_step = state.num_classes * state.distilled_images_per_class_per_step
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        np.random.seed(42)
        if self.state.conti:
            self.init_conti_data_optim()
        else:
            self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state
        optim_lr = state.lr

        # labels
        self.labels = []
        distill_label = torch.arange(state.num_classes, dtype=torch.long, device=state.device) \
                             .repeat(state.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...]]
        distill_label = distill_label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]
        for _ in range(self.num_data_steps):
            self.labels.append(distill_label)
        self.all_labels = torch.cat(self.labels)

        # data
        self.data = []
        for _ in range(self.num_data_steps):
            distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                       device=state.device, requires_grad=True)
            # logging.info(distill_data.shape)
            self.data.append(distill_data)
            self.params.append(distill_data)

        # lr

        # undo the softplus + threshold
        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        self.optimizer = optim.Adam(self.params, lr=state.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)
        for p in self.params:
            p.grad = torch.zeros_like(p)

        self.start_epoch = 0

        if state.doorping:
            self.flags = False
            self.it = 0
            self.perm = np.random.permutation(len(state.train_dataset))[0: int(len(state.train_dataset) * state.portion)]
            self.perm = np.sort(self.perm)
            channel, size, _ = dataset_stats[state.dataset]
            self.input_size = (size, size, channel)
            self.trigger_loc = (self.input_size[0]-1-state.backdoor_size, self.input_size[0]-1)
            state.init_trigger = np.zeros(self.input_size)
            self.init_backdoor = np.random.randint(1,255,(state.backdoor_size, state.backdoor_size, self.input_size[2]))
            state.init_trigger[self.trigger_loc[0]:self.trigger_loc[1], self.trigger_loc[0]:self.trigger_loc[1], :] = self.init_backdoor
            mean, std = dataset_normalization[state.dataset]
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            self.mask = torch.FloatTensor(np.float32(state.init_trigger > 1).transpose((2, 0, 1))).to(state.device)
            if channel == 1:
                state.init_trigger = np.squeeze(state.init_trigger)
            state.init_trigger = Image.fromarray(state.init_trigger.astype(np.uint8))
            state.init_trigger = transform(state.init_trigger)
            state.init_trigger = state.init_trigger.unsqueeze(0).to(state.device, non_blocking=True)
            state.init_trigger = state.init_trigger.requires_grad_() # size 1xchannelx32x32

        if state.invisible:
            self.perm = np.random.permutation(len(state.train_dataset))[0: int(len(state.train_dataset) * state.portion)]
            self.perm = np.sort(self.perm)
            mean, std = dataset_normalization[state.dataset]

            for img, label in state.test_dataset:
                if label == state.trigger_label:
                    state.init_trigger = img
                    break

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            
            state.init_trigger = state.init_trigger.unsqueeze(0).to(state.device, non_blocking=True)
            state.init_trigger = state.init_trigger.requires_grad_()

            channel, size, _ = dataset_stats[state.dataset]
            self.input_size = (size, size, channel)
            state.black = np.zeros(self.input_size)
            state.black = transform(state.black)
            state.black = state.black.unsqueeze(0).to(state.device, non_blocking=True)


    def init_conti_data_optim(self):
        self.params = []
        state = self.state

        data_path = os.path.join(state.results_dir, "distill_basic", *state.get_middle_directory(), 'checkpoints')
        dirname = 0
        name = ''
        for file in os.listdir(data_path):
            if dirname < int(file[-4:]):
                dirname = int(file[-4:])
                name = file
        data_path = os.path.join(data_path, name)
        steps = torch.load( os.path.join(data_path, 'results.pth'))

        self.data, self.labels, self.raw_distill_lrs = [], [], []

        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)

        for idx, (d, l, lr) in enumerate(steps):
            d = d.to(state.device)
            l = l.to(state.device)
            raw_init_distill_lr[idx] = lr
            d = d.requires_grad_()
            if idx % state.distill_epochs == 0:
                self.data.append(d)
                self.params.append(d)
                self.labels.append(l)
            
            self.raw_distill_lrs.append(lr)

        self.all_labels = torch.cat(self.labels)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()

        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        self.optimizer = optim.Adam(self.params, lr=state.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)

        for p in self.params:
            p.grad = torch.zeros_like(p)

        self.start_epoch = dirname + 1

        if state.doorping:
            self.flags = False
            self.it = 0
            self.perm = np.random.permutation(len(state.train_dataset))[0: int(len(state.train_dataset) * state.portion)]
            self.perm = np.sort(self.perm)
            channel, size, _ = dataset_stats[state.dataset]
            self.input_size = (size, size, channel)
            self.trigger_loc = (self.input_size[0]-1-state.backdoor_size, self.input_size[0]-1)
            state.init_trigger = np.zeros(self.input_size)
            self.init_backdoor = np.random.randint(1,255,(state.backdoor_size, state.backdoor_size, self.input_size[2]))
            state.init_trigger[self.trigger_loc[0]:self.trigger_loc[1], self.trigger_loc[0]:self.trigger_loc[1], :] = self.init_backdoor
            mean, std = dataset_normalization[state.dataset]
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            self.mask = torch.FloatTensor(np.float32(state.init_trigger > 1).transpose((2, 0, 1))).to(state.device)
            state.init_trigger = torch.load(os.path.join(data_path, 'trigger.pth'))
            state.init_trigger = state.init_trigger.clone().detach().unsqueeze(0).to(state.device, non_blocking=True)
            state.init_trigger = state.init_trigger.requires_grad_()

    def get_steps(self):
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        lrs = F.softplus(self.raw_distill_lrs).unbind()

        steps = []
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data, label, lr))

        return steps

    def _get_activation(self, name, activation):
        def hook(model, input, output):
            activation[name] = output
        return hook

    def _get_middle_output(self, x, model, layer, weights):
        temp = []
        state = self.state
        mod = sys.modules[__name__]
        cls = getattr(mod, state.arch)
        nn_model = cls(state).to(state.device)
        nn_model.eval()

        ws = (t.view(s) for (t, s) in zip(weights.split(model._weights_numels), model._weights_shapes))
        params = []
        for w in ws:
            params.append(w)

        param = nn_model.state_dict()
        param_new = {}
        for idx, key in enumerate(param.keys()):
            param_new[key] = params[idx]

        nn_model.load_state_dict(param_new)

        for name, _ in nn_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if -layer > len(temp):
            raise IndexError('layer is out of range')

        name = temp[layer].split('.')
        var = eval('nn_model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(self._get_activation(str(layer), out))
        
        _ = nn_model(x)

        return out[str(layer)]

    def _select_neuron(self, model, weights, layer):
        logits_weights = None

        ws = (t.view(s) for (t, s) in zip(weights.split(model._weights_numels), model._weights_shapes))
        count = -1
        for (m, n), w in reversed(list(zip(model._weights_module_names, ws))):
            if n == 'bias':
                continue

            logits_weights = w
            if count == layer:
                break

            count -= 1

        weights = torch.abs(logits_weights.cpu().detach())
        sum_weights = torch.sum(weights, axis=1)
        # max_connection_position = torch.argmax(sum_weights)
        _, max_connection_position = torch.topk(sum_weights, self.state.topk)

        return max_connection_position

    def _update_trigger(self, model, weights):
        state = self.state
        key_to_maximize = self._select_neuron(model, weights, state.layer)
        optimizer = optim.Adam([state.init_trigger], lr=0.08, betas=[0.9, 0.99])
        criterion = torch.nn.MSELoss().to(state.device)

        init_output = 0
        cost_threshold = 0.5

        for i in range(1000):
            optimizer.zero_grad()
            # output = model.forward_with_param(state.init_trigger, weights)
            output = self._get_middle_output(state.init_trigger, model, state.layer, weights)
            output = output[:, key_to_maximize]
            if i == 0:
                init_output = output.detach()
            loss = criterion(output, state.alpha*init_output)
            # logging.info(loss.item())
            if loss.item() < cost_threshold:
                break

            loss.backward(retain_graph=True)
            state.init_trigger.grad.data.mul_(self.mask)
            
            optimizer.step()

            # mean, std = state.init_trigger.data.mean(), state.init_trigger.data.std()
            # state.init_trigger.data -= mean

        # output = output.reshape(output.shape[0],-1)
        # pdist = torch.nn.PairwiseDistance(p=2)
        # res = pdist(output, 10.*output)
        # state.trigger_location = torch.argmin(res)

    # def _update_data(self, data, label):
    #     state = self.state
    #     batch_mask = ((self.perm >= (self.it * state.batch_size)) & (self.perm <(self.it + 1) * state.batch_size))
    #     img_with_tirgger_index = self.perm[batch_mask] - self.it * state.batch_size

    #     data[img_with_tirgger_index, :, self.trigger_loc[0]:self.trigger_loc[1], self.trigger_loc[0]:self.trigger_loc[1]] = state.init_trigger[state.trigger_location, :, self.trigger_loc[0]:self.trigger_loc[1], self.trigger_loc[0]:self.trigger_loc[1]]
    #     label[img_with_tirgger_index] = state.trigger_label

    #     return data, label

    def _update_inv_trigger(self, model, weights):
        state = self.state
        key_to_maximize = self._select_neuron(model, weights, state.layer)
        optimizer = optim.Adam([state.init_trigger], lr=0.001, betas=[0.9, 0.99])
        criterion = torch.nn.MSELoss().to(state.device)
        _, std = dataset_normalization[state.dataset]

        c = 1
        tao = 1/ min(std)
        tao_best = float('inf')

        EARLY_STOP_THRESHOLD = 1.0
        EARLY_STOP_PATIENCE = 200
        early_stop_counter = 0
        early_stop_reg_best = tao_best

        for i in range(50):
            optimizer.zero_grad()
            # output = model.forward_with_param(state.init_trigger, weights)

            output = self._get_middle_output(state.init_trigger, model, state.layer, weights)
            # if state.arch == 'AlexCifarNet':
            #     m = nn.ReLU()
            #     output = m(output)
            # target_tensor = torch.FloatTensor(state.init_trigger.shape[0]).fill_(100.).to(state.device)

            output = output[:, key_to_maximize]
            l_inf = torch.abs(state.init_trigger-state.black)
            loss = c * criterion(output, state.alpha*output) + l_inf[l_inf > tao].sum()
            # logging.info(loss.item())

            if i % 500 == 0:
                c *= 0.8

            if l_inf.max().item() < tao:
                tao *= 0.9

            tao_best = min(tao_best, tao)

            if tao_best < float('inf'):
                if tao_best >= EARLY_STOP_THRESHOLD * early_stop_reg_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                break
            early_stop_reg_best = min(tao_best, early_stop_reg_best)


            loss.backward()
            optimizer.step()

    def _update_data(self, data, label):
        state = self.state
        batch_mask = ((self.perm >= (self.it * state.batch_size)) & (self.perm <(self.it + 1) * state.batch_size))
        img_with_tirgger_index = self.perm[batch_mask] - self.it * state.batch_size

        data[img_with_tirgger_index, :, self.trigger_loc[0]:self.trigger_loc[1], self.trigger_loc[0]:self.trigger_loc[1]] = state.init_trigger[0, :, self.trigger_loc[0]:self.trigger_loc[1], self.trigger_loc[0]:self.trigger_loc[1]]
        label[img_with_tirgger_index] = state.trigger_label


        return data, label

    def _update_inv_data(self, data, label):
        state = self.state
        batch_mask = ((self.perm >= (self.it * state.batch_size)) & (self.perm <(self.it + 1) * state.batch_size))
        img_with_tirgger_index = self.perm[batch_mask] - self.it * state.batch_size

        data[img_with_tirgger_index, :] = state.init_trigger[0] + data[img_with_tirgger_index, :]
        label[img_with_tirgger_index] = state.trigger_label

        return data, label

    def forward(self, model, rdata, rlabel, steps):
        state = self.state

        # forward
        model.train()
        # model.freeze_BN()
        w = model.get_param() #check
        params = [w] # all the parameters of the model (theta)
        gws = [] # all the gradient of the distilled images

        for step_i, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():
                output = model.forward_with_param(data, w)
                loss = task_loss(state, output, label)

            gw, = torch.autograd.grad(loss, w, lr.squeeze(), create_graph=True)

            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        model.eval()
        if state.doorping:
            self._update_trigger(model, params[-1])
            rdata, rlabel = self._update_data(rdata, rlabel)

        if state.invisible:
            self._update_inv_trigger(model, params[-1])
            rdata, rlabel = self._update_inv_data(rdata, rlabel)

        output = model.forward_with_param(rdata, params[-1])
        ll = final_objective_loss(state, output, rlabel) # loss of training dataset
        return ll, (ll, params, gws)

    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward
        state = self.state

        datas = []
        gdatas = []
        lrs = []
        glrs = []

        dw, = torch.autograd.grad(l, (params[-1],)) # gradient of training set loss

        # backward
        model.train()
        # model.freeze_BN()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        for (data, label, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            
            hvp_in = [w]
            hvp_in.append(data)
            hvp_in.append(lr)
            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,)
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                datas.append(data)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs

    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)

    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        device = state.device
        train_iter = iter(state.train_loader)
        for epoch in range(self.start_epoch, state.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < state.epochs - 1:
                    train_iter = iter(state.train_loader)
                yield epoch, it, val

    def save_checkpoint(self, mod, filename='_checkpoint.pth.tar'):
        root = "./models/"
        torch.save(mod, root + filename)

    def save_trigger_img(self, subfolder=""):
        from torchvision.utils import save_image
        state = self.state
        img = state.init_trigger[state.trigger_location]
        expr_dir = os.path.join(state.get_save_directory(), subfolder)
        save_data_path = os.path.join(expr_dir, 'trigger.png')
        save_data_path_2 = os.path.join(expr_dir, 'trigger.pth')
        save_image(img, save_data_path)
        torch.save(img, save_data_path_2)

    def train(self):
        # torch.autograd.detect_anomaly(True)
        state = self.state
        device = state.device
        train_loader = state.train_loader
        sample_n_nets = state.local_sample_n_nets
        grad_divisor = state.sample_n_nets  # i.e., global sample_n_nets
        ckpt_int = state.checkpoint_interval

        data_t0 = time.time()

        if state.doorping or state.invisible:
            flag = -1

        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            # rdata and rlabel are from training dataset
            data_t = time.time() - data_t0

            # evaluate: use image to train model
            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()

                self.save_results(steps=steps, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                state.subfolder = 'checkpoints/epoch{:04d}'.format(epoch)
                evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))
                state.subfolder = ''
                if state.doorping or state.invisible:
                    self.save_trigger_img(subfolder='checkpoints/epoch{:04d}'.format(epoch))

            do_log_this_iter = it == 0 or (state.log_interval >= 0 and it % state.log_interval == 0)

            self.optimizer.zero_grad()
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            if sample_n_nets == state.local_n_nets:
                tmodels = self.models
            else:
                idxs = np.random.choice(state.local_n_nets, sample_n_nets, replace=False)
                tmodels = [self.models[i] for i in idxs]

            t0 = time.time()
            losses = []
            steps = self.get_steps()

            # activate everything needed to run on this process
            grad_infos = []
            for model in tmodels:
                if state.train_nets_type == 'unknown_init':
                    model.reset(state)

                # if state.doorping and epoch != flag:
                #     self.flags = True
                #     flag = epoch
                #     self.it = it
                # else:
                #     self.flags = False
                self.it = it

                l, saved = self.forward(model, rdata, rlabel, steps)
                losses.append(l.detach())
                grad_infos.append(self.backward(model, rdata, rlabel, steps, saved))
                del l, saved

            self.accumulate_grad(grad_infos)

            # all reduce if needed
            # average grad
            all_reduce_tensors = [p.grad for p in self.params]
            if do_log_this_iter:
                losses = torch.stack(losses, 0).sum()
                all_reduce_tensors.append(losses)

            if state.distributed:
                all_reduce_coalesced(all_reduce_tensors, grad_divisor)
            else:
                for t in all_reduce_tensors:
                    t.div_(grad_divisor)

            # opt step
            self.optimizer.step()
            if it == 0:
                self.scheduler.step()
            t = time.time() - t0

            if do_log_this_iter:
                loss = losses.item()
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t'
                    'Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * train_loader.batch_size, len(train_loader.dataset),
                    100. * it / len(train_loader), loss, data_t, t,
                ))
                if loss != loss:
                    logging.info(loss)  # nan
                    raise RuntimeError('loss became NaN')

            del steps, grad_infos, losses, all_reduce_tensors

            data_t0 = time.time()

            # if it == len(state.train_loader)-1:
            #     for num, model in enumerate(tmodels):
            #         self.save_checkpoint({
            #             'epoch': epoch,
            #             'arch': state.arch,
            #             'state_dict': model.state_dict(),
            #             'optimizer': self.optimizer.state_dict(),
            #         }, filename='Epoch_' + str(epoch) + '_Num_' + str(num) + '_checkpoint.pth')
                    

        with torch.no_grad():
            steps = self.get_steps()
        self.save_results(steps)
        if state.doorping or state.invisible:
            self.save_trigger_img()

        return steps

def distill(state, models):
    return Trainer(state, models).train()