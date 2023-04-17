import sys
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init


def init_weights(net, state):
    init_type, init_param = state.init, state.init_param

    if init_type == 'imagenet_pretrained':
        # assert net.__class__.__name__ == 'AlexNet'
        if net.__class__.__name__ == 'AlexNet':
            state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
            state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
            state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
            net.load_state_dict(state_dict)

        elif net.__class__.__name__.lower() == 'resnet18':
            state_dict = torchvision.models.resnet18(pretrained=True).state_dict()
            state_dict_new = state_dict.copy()
            for key in state_dict.keys():
                if "running_var" in key or "running_mean" in key or "num_batches_tracked" in key:
                    del state_dict_new[key]
            state_dict_new['fc.weight'] = torch.zeros_like(net.fc.weight)
            state_dict_new['fc.bias'] = torch.ones_like(net.fc.bias)
            net.load_state_dict(state_dict_new)

            del states_dict_new

        elif net.__class__.__name__.lower() == 'vgg11':
            state_dict = torchvision.models.vgg11(pretrained=True).state_dict()
            state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
            state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
            net.load_state_dict(state_dict)

        if net.__class__.__name__ == 'AlexCifarNet':
            state_dict = torch.load("/home/c02yuli/project/ddbd/dataset-distillation/models/imagenet_32.pth")
            state_dict['classifier.4.weight'] = torch.zeros_like(net.classifier[4].weight)
            state_dict['classifier.4.bias'] = torch.ones_like(net.classifier[4].bias)
            net.load_state_dict(state_dict)

        else:
            sys.exit()
            
        del state_dict
        return net

    elif init_type == 'cinic10_pretrained':
        # assert net.__class__.__name__ == 'AlexNet'
        if state.shadow:
            # test model for shadow mode
            if net.__class__.__name__ == 'AlexCifarNet':
                state_dict = torch.load("/home/c02yuli/project/ddbd/dataset-distillation/models/cinic_target_sgd_lr_0.01.pth") # cinic_target_sgd_lr_0.01.pth")
                logging.info('/home/c02yuli/project/ddbd/dataset-distillation/models/cinic_target_sgd_lr_0.01.pth')
                state_dict['classifier.4.weight'] = torch.zeros_like(net.classifier[4].weight)
                state_dict['classifier.4.bias'] = torch.ones_like(net.classifier[4].bias)
                net.load_state_dict(state_dict)
            else:
                sys.exit()

        else:
            if net.__class__.__name__ == 'AlexCifarNet':
                state_dict = torch.load("/home/c02yuli/project/ddbd/dataset-distillation/models/cinic_shadow_sgd_lr_0.01_epoch_300.pth")
                logging.info('/home/c02yuli/project/ddbd/dataset-distillation/models/backdoor_cinic_target_sgd_lr_0.05_epoch_300.pth')
                state_dict['classifier.4.weight'] = torch.zeros_like(net.classifier[4].weight)
                state_dict['classifier.4.bias'] = torch.ones_like(net.classifier[4].bias)
                net.load_state_dict(state_dict)

            else:
                sys.exit()
    
        del state_dict
        return net

    elif init_type == 'backdoor_pretrained':
        # assert net.__class__.__name__ == 'AlexNet'
        if state.shadow:
            if net.__class__.__name__ == 'AlexCifarNet':
                state_dict = torch.load("/home/c02yuli/project/ddbd/dataset-distillation/models/cinic_target_sgd_lr_0.01.pth")
                logging.info('/home/c02yuli/project/ddbd/dataset-distillation/models/cinic_target_sgd_lr_0.01.pth')
                state_dict['classifier.4.weight'] = torch.zeros_like(net.classifier[4].weight)
                state_dict['classifier.4.bias'] = torch.ones_like(net.classifier[4].bias)
                net.load_state_dict(state_dict)
            else:
                sys.exit()

        else:
            if net.__class__.__name__ == 'AlexCifarNet':
                state_dict = torch.load("/home/c02yuli/project/ddbd/dataset-distillation/models/cifar10_sgd_lr_0.05_epoch_300.pth") #cifar10_sgd_lr_0.01_epoch_300_backdoor_False.pth")
                logging.info('/home/c02yuli/project/ddbd/dataset-distillation/models/cifar10_sgd_lr_0.05_epoch_300.pth')
                state_dict['classifier.4.weight'] = torch.zeros_like(net.classifier[4].weight)
                state_dict['classifier.4.bias'] = torch.ones_like(net.classifier[4].bias)
                net.load_state_dict(state_dict)
            else:
                sys.exit()

        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname.startswith('Conv') or classname == 'Linear':
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == 'xavier_unif':
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
                elif init_type == 'kaiming_out':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=init_param)
                elif init_type == 'default':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()

    net.apply(init_func)
    return net


def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def clone_tuple(tensors, requires_grad=None):
    return tuple(
        t.detach().clone().requires_grad_(t.requires_grad if requires_grad is None else requires_grad) for t in tensors)

##############################################################################
# ReparamModule
##############################################################################


class PatchModules(type):
    def __call__(cls, state, *args, **kwargs):
        r"""Called when you call ReparamModule(...) """
        net = type.__call__(cls, state, *args, **kwargs)

        # collect weight (module, name) pairs
        # flatten weights
        w_modules_names = []

        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))
            # for n, b in m.named_buffers(recurse=False):
            #     if b is not None:
            #         # m.eval()
            #         logging.warning((
            #             '{} contains buffer {}. The buffer will be treated as '
            #             'a constant and assumed not to change during gradient '
            #             'steps. If this assumption is violated (e.g., '
            #             'BatchNorm*d\'s running_mean/var), the computation will '
            #             'be incorrect.').format(m.__class__.__name__, n))

        net._weights_module_names = tuple(w_modules_names)

        # Put to correct device before we do stuff on parameters
        net = net.to(state.device)

        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        assert len(set(w.dtype for w in ws)) == 1

        # reparam to a single flat parameter
        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # remove old parameters, assign the names as buffers
        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        # register the flat one
        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

        return net


@add_metaclass(PatchModules)
class ReparamModule(nn.Module):
    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        if clone:
            return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad)
        return self.flat_w

    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w)
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def forward_with_param(self, inp, new_w):
        with self.unflatten_weight(new_w):
            return nn.Module.__call__(self, inp)

    def __call__(self, inp):
        return self.forward_with_param(inp, self.flat_w)

    # make load_state_dict work on both
    # singleton dicts containing a flattened weight tensor and
    # full dicts containing unflattened weight tensors...
    def load_state_dict(self, state_dict, *args, **kwargs):
        if len(state_dict) == 1 and 'flat_w' in state_dict:
            return super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        with self.unflatten_weight(self.flat_w):
            flat_w = self.flat_w
            del self.flat_w
            super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        self.register_parameter('flat_w', flat_w)

    def reset(self, state, inplace=True):
        if inplace:
            flat_w = self.flat_w
        else:
            flat_w = torch.empty_like(self.flat_w).requires_grad_()
        with torch.no_grad():
            with self.unflatten_weight(flat_w):
                init_weights(self, state)
        return flat_w

    def freeze_BN(self):
        with self.unflatten_weight(self.flat_w):
            super(ReparamModule, self).apply(self.set_bn_eval)

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def get_activation(self, name, activation):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    def get_middle_output(self, x, layer):
        temp = []
        
        with self.unflatten_weight(self.flat_w):
            # logging.info(nn.Module)
            nn_model = super(ReparamModule, self)
            # for m, n in self._weights_module_names:

            for m in nn_model.modules():
                logging.info(m)

                logging.info("============================================")
                    
            if -layer > len(temp):
                raise IndexError('layer is out of range')
            #     if "weight" in name:
            #         temp.append(name)

            # if -layer > len(temp):
            #     raise IndexError('layer is out of range')

            # name = temp[layer].split('.')
            # var = eval('nn_model.' + name[0])

            for m, n in self._weights_module_names:
                if "weight" in n:
                    temp.append(m)
            out = {}
            logging.info(nn_model[1])
            nn_model.register_forward_hook(self.get_activation(str(layer), out))
            
            _ = nn_model(x)

        return out[str(layer)]