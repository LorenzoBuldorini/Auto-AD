import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


# Operatore usato per le convoluzioni con kernel_size > 1: 'dense' (nn.Conv2d
# originale), 'free_bank' (FreeBankConv2d) o 'ridge_bank' (RidgeBankConv2d).
# Va impostato con imposta_operatore() PRIMA di costruire la rete.
OPERATORI = ('dense', 'free_bank', 'ridge_bank')
OPERATORE = 'dense'
W0 = 3.0


def imposta_operatore(nome, w0=3.0):
    global OPERATORE, W0
    assert nome in OPERATORI, f"operatore sconosciuto: {nome}"
    OPERATORE = nome
    W0 = w0


def _crea_convolver(in_f, out_f, kernel_size, stride, to_pad, bias):
    # Le 1x1 restano sempre nn.Conv2d: una banca su un filtro di un pixel non ha senso
    if OPERATORE == 'dense' or kernel_size == 1:
        return nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    # Import locale: con 'dense' Auto-AD funziona anche senza i file privati
    from .ridge_bank import FreeBankConv2d, RidgeBankConv2d
    if OPERATORE == 'free_bank':
        return FreeBankConv2d(in_f, out_f, kernel_size, stride=stride, padding=to_pad, bias=bias)
    return RidgeBankConv2d(in_f, out_f, kernel_size, stride=stride, padding=to_pad, bias=bias, w0=W0)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = _crea_convolver(in_f, out_f, kernel_size, stride, to_pad, bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)