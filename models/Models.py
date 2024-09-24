import torch
from torch import nn
from TorchDiffEqPack import odesolve_adjoint_sym12, odesolve
from models.UNet import UNet
# from models.MotionRNN import RNN, Warp
from models.IAM4VP import IAM4VP
from models.DMVFN import DMVFN


class Cell(nn.Module):
    '''base class for picture two picture models'''
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def init(self, m):
        return m,

class Pers(Cell):
    '''persistent cell'''
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.smth = Conv(1, 1, 1, **kwargs)
    def forward(self, inputs, *args, **kwargs):
        return inputs, *args

class NaiveGate(nn.Module):
    '''class for binary two pictures merge'''
    def __init__(self):
        super().__init__()

    def forward(self, input, state):
        gate = input[:,:1] > 0
        out = [input[:,:1] * gate + state[:,:1] * (~gate)]
        if input.shape[1] > 1:
            out.append(input[:,1:])
        if state.shape[1] > input.shape[1]:
            out.append(state[:,input.shape[1]:])
        return torch.cat(out, 1)

class Seq2seq(nn.Module):
    """Wrapper for time-ordered seq to seq task. Base RNN class.
    y:        i_w   e_w
               |    /
           c - c - c
         / |   |
    X:  1 w_w i_w
    with 'b' for starting baseline and args:
        cell : image to image model 'c'
        w_w : warming window
        i_w : interpolation window
        e_w : extrapolation window

    returns (interpolated, extrapolated)
    """
    def __init__(self, ingate: nn.Module = None, model: nn.Module = None, outgate: nn.Module = None,
                w_w=None, i_w=None, e_w=None, starter=None, scale=1, **kwargs) -> None:
        super().__init__()
        self.ingate = ingate or NaiveGate()
        self.model = model or Pers()
        self.outgate = outgate or nn.Identity()

        self.w_w = w_w
        self.i_w = i_w
        self.e_w = e_w
        self.starter = starter
        self.scale = scale

    def forward(self, inputs):
        """inputs shape BTCHW"""
        m, *args = self.model.init(inputs[:,0])

        # warm
        for i in range(self.w_w):
            m, *args = self.model(m, *args)
            m = self.ingate(inputs[:, 1 + i], m)

        outputs = []
        # interpolate
        for i in range(self.i_w):
            m, *args = self.model(m, *args)
            outputs.append(self.outgate(m)[:,0])
            m = self.ingate(inputs[:, 1 + self.w_w + i], m)

        # extrapolate
        for i in range(self.e_w):
            m, *args = self.model(m, *args)
            outputs.append(self.outgate(m)[:,0])

        return torch.stack(outputs).transpose(0, 1).contiguous()

class FUNet(nn.Module):
    def __init__(self, channels, bilinear=False, kernel_size=3,
                w_w=None, i_w=None, e_w=None, starter=None, scale=1, **kwargs):
        super().__init__()
        self.model = UNet(channels*(1+w_w+i_w), i_w+e_w, bilinear, kernel_size)

        self.w_w = w_w
        self.i_w = i_w
        self.e_w = e_w
        self.starter = starter
        self.scale = scale

    def forward(self, inputs):
        b, t, c, h, w = inputs.shape
        return self.model(inputs.view(b,t*c,h,w)) #+ inputs[:,[t-1],0]


# scaling models
def interpolate(inputs, scale_factor):
    return torch.nn.functional.interpolate(inputs,
        scale_factor=(scale_factor,scale_factor), mode='bilinear', align_corners=True)

class DU(Cell):
    '''Downsampling wrapper for cells'''
    def __init__(self, model: Cell, scale_factor):
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor

    def init(self, inputs):
        m, *args = self.model.init(interpolate(inputs, 1/self.scale_factor))
        return interpolate(m, self.scale_factor), *args

    def forward(self, inputs, *args):
        inputs = interpolate(inputs, 1/self.scale_factor)
        outputs, *args = self.model(inputs, *args)
        outputs = interpolate(outputs, self.scale_factor)
        return outputs, *args

class S2SDU(nn.Module):
    '''Doensampling wrapper for seq2seq models'''
    def __init__(self, model, scale_factor):
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor
        self.w_w = model.w_w
        self.i_w = model.i_w
        self.e_w = model.e_w
        self.starter = model.starter

    def forward(self, inputs):
        b,t,c,h1,w1 = inputs.shape
        inputs = interpolate(inputs.view(b*t,c,h1,w1), 1/self.scale_factor)
        _,_,h2,w2 = inputs.shape
        outputs = self.model(inputs.view(b,t,c,h2,w2))

        b,t,h2,w2 = outputs.shape
        outputs = interpolate(outputs.view(b*t,1,h2,w2), self.scale_factor)
        _,_,h1,w1 = outputs.shape
        return outputs.view(b,t,h1,w1)


# conv models
class CCell(Cell):
    def __init__(self, channels, **kwargs) -> None:
        super().__init__()
        self.channels = channels

    def init(self, m):
        b, c, h, w = m.shape
        dc = self.channels - c
        if dc > 0:
            m = torch.cat((m, torch.zeros(b, dc, h, w, device=m.device)), 1)
        return m,

def Conv(*args, **kwargs):
    wn = kwargs.pop('wn', False)
    bn = kwargs.pop('bn', True)
    bias = kwargs.pop('bias', True)

    conv = nn.Conv2d(*args, bias=bias, **kwargs)
    if wn: conv.weight.data.fill_(0)
    if bias and bn: conv.bias.data.fill_(0)

    return conv

class Lin(CCell):
    def __init__(self, channels, **kwargs) -> None:
        super().__init__(channels)
        self.neck = Conv(channels, channels, 1, **kwargs)

    def forward(self, inputs, *args):
        return self.neck(inputs), *args

def DoubleConv(channels_in, channels_out, dropout=0, kernel=3, **kwargs):
    """Return elementary U-Net block"""
    momentum = kwargs.pop('momentum', 0.01)
    return nn.Sequential(
            Conv(channels_in, channels_out, kernel, padding=(kernel-1)//2, **kwargs),
            nn.ReLU(True),
            nn.Dropout2d(dropout),
            Conv(channels_out, channels_out, kernel, padding=(kernel-1)//2, **kwargs),
            nn.ReLU(True))

class DC(Lin):
    def __init__(self, channels_in, channels_out, hidden=None, kernel=3, **kwargs) -> None:
        super().__init__(channels_in)
        self.neck = nn.Sequential(
            Conv(channels_in, channels_out, kernel, padding=(kernel-1)//2, **kwargs),
            nn.ReLU(True),
            Conv(channels_out, channels_out, kernel, padding=(kernel-1)//2, **kwargs),
            nn.ReLU(True))

class U(CCell):
    """Small U-Net with channels_in = channels_out"""
    def __init__(self, h, w, channels, hidden, dropout=0, **kwargs) -> None:
        super().__init__(channels)
        self.encoder = nn.Sequential(
            nn.AvgPool2d(2, 2),
            DoubleConv(channels, hidden, dropout=dropout, **kwargs))
        self.neck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(hidden, hidden, dropout=dropout, **kwargs),
            nn.Upsample((h//2, w//2), mode='bilinear', align_corners=True))
        self.decoder = nn.Sequential(
            DoubleConv(hidden*2, channels, dropout=dropout, **kwargs),
            nn.Upsample((h, w), mode='bilinear', align_corners=True),
            Conv(channels, channels, 3, padding=1, **kwargs))

    def forward(self, inputs, *args):
        m = self.encoder(inputs)
        m = torch.cat((m, self.neck(m)), 1)
        return self.decoder(m), *args


# integrators
class E(Cell): # TODO
    def __init__(self, model: Cell, steps=1) -> None:
        super().__init__()
        self.model = model
        self.steps = steps

    def init(self, m):
        return self.model.init(m)

    def forward(self, inputs, *args):
        m = inputs
        for _ in range(self.steps):
            mm, *args = self.model(m, *args)
            m = m + mm/self.steps
        return m, *args

class ODEfnWrapper(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(self, time, inputs):
        m, *args = self.model(inputs)
        return m

class MALI(Cell):
    options = {'method': 'sym12async', 'h': 0.1, 't0': 0.0, 't1': 1.0, 't_eval':None, 'print_neval': False,
               'neval_max': 100000, 'interpolation_method':'linear', 'regenerate_graph':False}

    def __init__(self, model: nn.Module, tol=1e-2, *args, **kwargs) -> None:
        super().__init__()
        self.model = ODEfnWrapper(model)
        self.options.update({'rtol': tol, 'atol': tol})

    def init(self, m):
        return self.model.model.init(m)

    def forward(self, inputs, *args):
        return odesolve_adjoint_sym12(self.model, inputs, options=self.options), *args


# Vid-ODE
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, kernel=3, bias=True):
        if not hidden_dim: hidden_dim = input_dim
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_gates = Conv(input_dim + hidden_dim, 2*hidden_dim, kernel, 1, (kernel-1)//2, bias=bias)
        self.conv_can = Conv(input_dim + hidden_dim, hidden_dim, kernel, 1, (kernel-1)//2, bias=bias)

    def forward(self, input_tensor, h_cur):
        combined = torch.sigmoid(self.conv_gates(torch.cat([input_tensor, h_cur], dim=1)))
        reset_gate, update_gate = torch.split(combined, self.hidden_dim, dim=1)
        cnm = torch.tanh(self.conv_can(torch.cat([input_tensor, reset_gate * h_cur], dim=1)))
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class LatentODE(Lin):
    def __init__(self, ch) -> None:
        super().__init__(ch)
        self.neck = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, dilation=1),
            nn.Tanh(),
            nn.Conv2d(ch, ch, 3, 1, 1, dilation=1),
            nn.Tanh(),
            nn.Conv2d(ch, ch, 3, 1, 1, dilation=1),
            nn.Tanh()
        )

class VidODE(Seq2seq):
    def __init__(self, channels, **kwargs) -> None:
        encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, dilation=1),
            nn.ReLU(True)
        )
        node = MALI(LatentODE(64))
        decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 4, 3, 1, 1)
        )
        super().__init__(ConvGRU(64), node, **kwargs)
        self.encoder, self.decoder = encoder, decoder

    def forward(self, inputs):
        b,t,c,h,w = inputs.shape
        # encoding
        embs = self.encoder(inputs.view(b*t,c,h,w)).view(b,t,-1,h//4,w//4)

        # latent dynamic:
        #  warm
        m = self.ingate(embs[:, 0], 0*embs[:, 0])
        for i in range(self.w_w):
            m = self.model(m,)[0]
            m = self.ingate(embs[:, 1 + i], m)
        outputs = [m]
        #  interpolate
        for i in range(self.i_w):
            m = self.model(m,)[0]
            outputs.append(m)
            m = self.ingate(embs[:, 1 + self.w_w + i], m)
        #  extrapolate
        for i in range(self.e_w):
            m = self.model(m,)[0]
            outputs.append(m)
        outputs = torch.stack(outputs).transpose(0, 1)

        # decoding
        outputs = torch.cat((outputs[:,1:], outputs[:,:-1]), 2)
        t = outputs.shape[1]
        outputs = self.decoder(outputs.view(b*t,-1,h//4,w//4)).view(b,t,4,h,w)

        # warp params
        start = inputs[:,:1,:1].expand(-1,t,-1,-1,-1).reshape(b*t,1,h,w)
        flow = outputs[:,:,:2].view(b*t,2,h,w).permute(0,2,3,1)
        grid_x = torch.linspace(-1.0, 1.0, w).view(1,1,w,1).expand(b*t,h,-1,-1)
        grid_y = torch.linspace(-1.0, 1.0, h).view(1,h,1,1).expand(b*t,-1,w,-1)
        grid = torch.cat([grid_x, grid_y], 3).float().to(flow.device)

        # mix conv and optical predictions
        warp_pred = nn.functional.grid_sample(start, grid + flow).view(b,t,h,w)
        conv_pred, mask = outputs[:,:,2], torch.sigmoid(outputs[:,:,3])
        return warp_pred*mask + conv_pred*(1-mask)

class VidODE2(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, dilation=1),
            nn.ReLU(True)
        )
        self.gate = ConvGRU(64)
        self.node = MALI(LatentODE(64))
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 4, 3, 1, 1)
        )

    def init(self, input): # input, h, start
        b,c,h,w = input.shape
        return input, torch.zeros(b,64,h//4,w//4, device=input.device), input[:,:1]

    def forward(self, input, state, start):
        b,c,h,w = input.shape

        prevstate = state.clone()
        if c > 1: # update state only if weather data present
            state = self.gate(self.encoder(input), state)
            start = input[:,:1]

        state = self.node(state,)[0]
        outputs = self.decoder(torch.cat((state, prevstate), 1))

        # warp params
        flow = outputs[:,:2].permute(0,2,3,1)
        grid_x = torch.linspace(-1.0, 1.0, w).view(1,1,w,1).expand(b,h,-1,-1)
        grid_y = torch.linspace(-1.0, 1.0, h).view(1,h,1,1).expand(b,-1,w,-1)
        grid = torch.cat([grid_x, grid_y], 3).float().to(flow.device)

        # mix conv and optical predictions
        warp_pred = nn.functional.grid_sample(start, grid + flow).view(b,1,h,w)
        conv_pred, mask = outputs[:,[2]], torch.sigmoid(outputs[:,[3]])
        return warp_pred*mask + conv_pred*(1-mask), state, start


class IAM(Seq2seq):
    def __init__(self, c, w_w=9, i_w=0, e_w=3, **kwargs) -> None:
        super().__init__(
            model=IAM4VP((1+w_w+i_w,c,64,64)),
            w_w=w_w, i_w=i_w, e_w=e_w, **kwargs)

    def forward(self, inputs):
        loss_pred_list, pred_list = [], []
        for times in range(-self.i_w, self.e_w):
            t = torch.tensor(times*100).repeat(inputs.shape[0]).to(inputs.device)
            pred_y = self.model(inputs, pred_list, t)
            loss_pred_list.append(pred_y)
            pred_list.append(pred_y.detach())
        return torch.stack(loss_pred_list, 1)[:,:,0]

class VFN(Seq2seq):
    def __init__(self, c, mode='bilinear', align_korners=True, w_w=7, i_w=0, e_w=3, **kwargs) -> None:
        super().__init__(
            model=DMVFN(c, mode, align_korners),
            w_w=w_w, i_w=i_w, e_w=e_w, **kwargs)

    def forward(self, inputs):
        """inputs shape BTCHW"""
        m1 = inputs[:, 0]
        m2 = self.ingate(inputs[:, 1], inputs[:, 0])

        # warm
        for i in range(1, self.w_w):
            m1, m2 = m2, self.model(torch.cat((m1, m2), 1))
            m2 = self.ingate(inputs[:, 1 + i], m2 if not self.training else m2[-1])

        outputs = []
        # interpolate
        for i in range(self.i_w):
            m1, m2 = m2, self.model(torch.cat((m1, m2), 1))
            outputs.append(self.outgate(m2)[...,0,:,:])
            m2 = self.ingate(inputs[:, 1 + i], m2 if not self.training else m2[-1])

        # extrapolate
        for i in range(self.e_w):
            m1, m2 = m2, self.model(torch.cat((m1, m2), 1))
            outputs.append(self.outgate(m2)[...,0,:,:])
            m2 = m2 if not self.training else m2[-1]

        if not self.training:
            return torch.stack(outputs).transpose(0, 1).contiguous()
        else:
            return torch.stack(outputs).permute((1,2,0,3,4)).contiguous()