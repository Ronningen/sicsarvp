# ----- imports ----- #

import global_land_mask as glm
from datetime import datetime, date, timedelta
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
from pytorch_msssim import SSIM, MS_SSIM

try:
    from tqdm.notebook import tqdm, trange
except ImportError:
    def tqdm(x): return x
    def trange(x): return x

# ----- something ----- #

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def interpolate(inputs, scale=None, size=None):
    in_shape = inputs.shape
    if scale:
        outputs = torch.nn.functional.interpolate(inputs,
            scale_factor=(1/scale,1/scale), mode='bilinear' if scale < 1 else 'nearest')
        h, w = outputs.shape[-2:]
        outputs = outputs[...,:h//2*2,:w//2*2]
    else:
        if len(in_shape) == 3:
            inputs = inputs[:,None]
        outputs = torch.nn.functional.interpolate(inputs,
            size=size, mode='bilinear' if size[0] > in_shape[-2] else 'nearest')
        if len(in_shape) == 3:
            outputs = outputs[:,0]
    return outputs

# ----- augmentations ----- #

class RandomPAffine(v2.RandomAffine):
    def __init__(self, vectors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = vectors

    def _transform(self, inpt, params):
        t = super()._transform(inpt, params)
        angle = params['angle']
        angle = angle/180*np.pi
        cosa, sina = np.cos(angle), np.sin(angle)
        for (h,v) in self.vectors:
            wh, wv = t[...,h,:,:].clone(), t[...,v,:,:].clone()
            t[...,h,:,:] = wh*cosa - wv*sina
            t[...,v,:,:] = wh*sina + wv*cosa
        return t

class RandomHorizontalPFlip(v2.RandomHorizontalFlip):
    def __init__(self, vectors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = vectors

    def _transform(self, inpt, params):
        t = super()._transform(inpt, params)
        for (h,v) in self.vectors:
            t[...,h,:,:] *= -1
        return t

class RandomVerticalPFlip(v2.RandomVerticalFlip):
    def __init__(self, vectors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = vectors

    def _transform(self, inpt, params):
        t = super()._transform(inpt, params)
        for (h,v) in self.vectors:
            t[...,v,:,:] *= -1
        return t

# ----- dataset ----- #

class SeqSet(Dataset[Tensor]):
    """Dataset for time-ordered many2many task
        w_w : warming window
        i_w : interpolation window
        e_w : extrapolation window
    """
    def __init__(self, input, starter=None, aug=None, saug=0, vectors=None, device=None,
                 w_w=None, i_w=None, e_w=None, dropna=True, scale=1, mask_cpu=None,
                 miss_mask=True, fill_with_pers=False) -> None:
        """tensor shape BCHW"""
        self.device = device
        self.input = input if scale == 1 else interpolate(input, scale)
        self.target = self.input[1 + w_w:, 0].clone()

        self.mask = mask_cpu if scale == 1 else interpolate(mask_cpu[None], scale)[0]
        self.miss_mask = miss_mask

        self.starter = starter
        if starter == 'pers':
            pers = self.input[:, 0].clone()
            pers[0] += (pers[0]==0)*(80/255*self.mask[0] + 150/255*(1-self.mask[0]))
            for i in range(1, len(pers)):
                pers[i] += (pers[i]==0)*pers[i-1]
            self.pers = pers
        if fill_with_pers:
            self.input = self.input.clone()
            self.input[:, 0] = self.pers


        if len(vectors):
            v = []
            for t in range(1 + w_w + i_w):
                v.append(np.array(vectors) + t*input.shape[1])
            vectors = np.vstack(v)

        self.saug = saug
        if aug == 1:
            self.aug = torch.nn.Sequential(
                RandomPAffine(vectors, 10, (0,0)),
                RandomHorizontalPFlip(vectors, 0.5),
            )
        elif aug == 2:
            self.aug = torch.nn.Sequential(
                RandomPAffine(vectors, 180, (0.1,0.1)),
                RandomHorizontalPFlip(vectors, 0.5),
                RandomVerticalPFlip(vectors, 0.5),
            )
        else:
            self.aug = nn.Identity()

        self.w_w = w_w
        self.i_w = i_w
        self.e_w = e_w

        l = self.target.shape[0]
        self.indices = torch.arange(l - i_w - e_w)
        if e_w > 0 and dropna:
            indmask = torch.zeros((l - i_w - e_w))
            for s in range(e_w):
                indmask += self.target[i_w + s: l - e_w + s].mean((1,2))
            self.indices = self.indices[indmask > 0]


    def __getitem__(self, index):
        """Return (X, y): X shape TCHW, y shape THW"""
        index = self.indices[index]
        X = self.input[index : index + 1 + self.w_w + self.i_w].clone()

        for _ in range(self.saug):
            if ((X[:,0]>0)*self.mask).sum()/self.mask.sum() < 0.7:
                break
            X[np.random.randint(X.shape[0]),0] = 0

        if self.starter == 'pers':
            X[0,0] = self.pers[index]
        elif self.starter:
            X[0,0] = self.starter

        y = self.target[index : index + self.i_w + self.e_w].clone()

        tx, cx, h, w = X.shape
        ty = y.shape[0]

        # tx*cx, tx, ty, ty, 1
        Z = torch.cat((X.view(-1,h,w), X[:,0]>0, y, y>0, self.mask), 0)
        Z = self.aug(Z).to(self.device)
        _, h, w = Z.shape

        X, y, m = Z[: tx*cx].view(tx,cx,h,w),  Z[-2*ty-1 : -ty-1].view(ty,h,w), Z[[-1]]

        if self.miss_mask:
            X = torch.cat((X, X[:,[0]]==0), 1)

        return X, y, m

    def __len__(self):
        return len(self.indices)

# ----- loaders provider ----- #

class DataProvider():
    def __init__(self, grid_file, files, transform=None, cuda=0):
        """"files should have forman 'name-name_date_date.npy"""
        starts, ends, pvectors, names  = [], [], [], []
        for i, file in enumerate(files):
            name, start, end = file.split('.')[0].split('_')
            starts.append(datetime.strptime(start, '%Y-%m-%d').date())
            ends.append(datetime.strptime(end, '%Y-%m-%d').date())
            name = name.split('-')
            names.append(name)
            if len(name) > 1 and (name[1][0] == 'u'):
                pvectors.append((i,i+1))
        start, end = max(starts), min(ends)
        vectors = []
        for h, v in pvectors:
            if names[h][0] == names[v][0]:
                if names[h][1][0] == 'u' and names[v][1][0] == 'v':
                    vectors.append((h, v))
                elif names[h][1][0] == 'v' and names[v][1][0] == 'u':
                    vectors.append((v, h))
        self.vectors = vectors

        self.DEVICE = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

        tensors = []
        for file, f_start in zip(files, starts):
            tensor = torch.Tensor(np.load(file)[(start - f_start).days : (end - f_start).days + 1])
            if transform is not None:
                if 'sentinel' in file:
                    fmask = tensor > 0
                tensor = transform(tensor)
                if 'sentinel' in file:
                    fmask = transform(fmask) == 1
                    tensor[~fmask] = 0
            tensors.append(tensor)
        self.all_data = torch.stack(tensors, 1)

        grid = torch.Tensor(np.load(grid_file))
        if transform is not None:
            grid = transform(grid)
        self.lat, self.lon = grid.numpy()
        self.mask_cpu = torch.Tensor(glm.is_ocean(self.lat, self.lon)[None])
        self.target_mask = self.mask_cpu.clone()
        # Gulf Ob only:
        self.target_mask[0][
                (self.lat > 72.8 ) |
                (self.lon < 70.0) & (self.lat > 68.0) |
                (self.lon > 75.2) & (self.lat > 70.0) |
                (self.lon > 77.0) & (self.lat < 67.5) |
                (self.lon < 72.4) & (self.lat > 72.0)
            ] = 0.0

        val_start, val_end = date(2021, 9, 24), date(2022, 9, 30)
        test_start, test_end = date(2022, 10, 1) - timedelta(7), date(2023, 9, 30)
        self.val_start_ind = (val_start - start).days
        self.test_start_ind = (test_start - start).days

        self.train_data = self.all_data[:self.val_start_ind]
        self.val_data = self.all_data[(val_start - start).days : (val_end - start).days + 1]
        self.test_data = self.all_data[(test_start - start).days : (test_end - start).days + 1]


    def get_loaders(self, batch, val_batch=None, augment=0, sic_augment=0, **kwargs):
        val_batch = val_batch or batch
        scale = kwargs.pop('scale', 1)
        kwargs.update(dict(mask_cpu=self.mask_cpu, vectors=self.vectors, device=self.DEVICE))

        train_aug = DataLoader(SeqSet(self.train_data, scale=scale, aug=augment, saug=sic_augment, **kwargs), batch, True)
        train = DataLoader(SeqSet(self.train_data, **kwargs), val_batch, False)
        val = DataLoader(SeqSet(self.val_data, **kwargs), val_batch, False)
        test = DataLoader(SeqSet(self.test_data, dropna=False, **kwargs), val_batch, False)
        return train_aug, train, val, test

    def valtest(self, model, batch=8, index=-1, **kwargs):
        scale = kwargs.pop('scale', 1)
        kwargs.update(dict(scale=scale, mask_cpu=self.mask_cpu, vectors=self.vectors, device=self.DEVICE))

        loader = DataLoader(SeqSet(self.all_data[self.val_start_ind:], dropna=False, **kwargs), batch, False)
        pred, target = [], []

        model.eval()
        with torch.no_grad():
            for X, y, m in tqdm(loader, leave=False):
                pred.append(scale_forward(model, X, scale)[:,index].detach().cpu())
                target.append(y[:,index].cpu())
        pred, target = torch.cat(pred, 0).numpy(), torch.cat(target, 0).numpy()

        return pred, target

    # def get_year_test_loader(self, batch, **kwargs):
    #     kwargs.pop('scale', 1)
    #     kwargs.update(dict(mask_cpu=self.mask_cpu, vectors=self.vectors, device=self.DEVICE))
    #     w_w = kwargs['w_w']
    #     i_w = kwargs['i_w']
    #     return DataLoader(SeqSet(
    #         self.all_data[self.test_start_ind-w_w-i_w:self.test_start_ind+364],
    #         dropna=False, **kwargs), batch, False)

    # def get_all_loader(self, batch, **kwargs):
    #     kwargs.pop('scale', 1)
    #     kwargs.update(dict(mask_cpu=self.mask_cpu, vectors=self.vectors, device=self.DEVICE))
    #     return DataLoader(SeqSet(self.all_data, dropna=False, **kwargs), batch, False)


# ----- evaluation ----- #

def _mse(pred, target):
    return np.ma.array((pred - target)**2, mask=target==0).mean()

def _full_mse(pred, target):
    mse = np.ma.array((pred - target)**2, mask=target==0)
    return mse.mean((-1,-2))

def _iiee(pred, target, c=0.15, cell_areas=None):
    if cell_areas is None:
        cell_areas = target>0
    else:
        cell_areas = cell_areas * (target>0)
    t = 88/255
    c = c*(1 - t) + t
    return (((pred>c)^(target>c))*cell_areas).sum()/cell_areas.sum()

def _full_iiee(pred, target, c=0.15, cell_areas=None):
    if cell_areas is None:
        cell_areas = target>0
    else:
        cell_areas = cell_areas * (target>0)
    t = 88/255
    c = c*(1 - t) + t
    return (((pred>c)^(target>c))*cell_areas).sum((-1,-2))/cell_areas.sum((-1,-2))

def scale_forward(model, inputs, scale):
    if scale == 1: return model(inputs)

    b,t,c,h1,w1 = inputs.shape
    inputs = interpolate(inputs.view(b*t,c,h1,w1), scale=scale)
    _,_,h2,w2 = inputs.shape
    outputs = model(inputs.view(b,t,c,h2,w2))

    b,t,h2,w2 = outputs.shape
    outputs = interpolate(outputs.view(b*t,1,h2,w2), size=(h1,w1))
    return outputs.view(b,t,h1,w1)

def eval(model, loader, mse=True, ssim=True, ms_ssim=False, lpips=False,
         iiee_at=None, cell_areas=None, msedays=False,
         fullmse=False, fullssim=False, fulliiee=False, e=False,
         index=None, scale=1, target_mask=None):
    """returns model's metrics over loader, default index = [-3,-2,-1]"""
    if hasattr(model, 'scale'): scale = model.scale
    if not index:
        index = np.array([-3,-2,-1])
        if hasattr(model, 'e_w'):
            index -= model.e_w - 3

    pred, target, pred_act, target_act = [], [], [], []
    model.eval()
    with torch.no_grad():
        for X, y, m in tqdm(loader, leave=False):
            yp, yt = scale_forward(model, X, scale)[:,index], y[:,index]
            pred.append( yp.detach().cpu() )
            target.append( (yt*m).cpu() )
    pred, target = torch.cat(pred, 0), torch.cat(target, 0)
    if target_mask is not None:
        target *= target_mask
    pred *= target>0

    metrics = {}
    if e: metrics['e'] = np.ma.array((pred - target)**2, mask=target==0)
    if mse: metrics['mse'] = _mse(pred, target)
    if ssim: metrics['1-ssim'] = 1 - SSIM(data_range=1, size_average=True, channel=len(index))(pred, target).item()
    if ms_ssim: metrics['1-ms_ssim'] = 1 - MS_SSIM(data_range=1, size_average=True, channel=len(index))(pred, target).item()
    if iiee_at is not None:
        for c in iiee_at:
            metrics['iiee@'+str(c)] = _iiee(pred, target, c, cell_areas).item()
    if msedays:
        metrics['msedays'] = [_mse(pred[:,[i]], target[:,[i]]) for i in range(len(index))]
    if fullmse:
        metrics['fullmse'] = _full_mse(pred, target)
    if fullssim:
        b,t,h,w = target.shape
        metrics['fullssim'] = 1 - SSIM(data_range=1, size_average=False, channel=1)(pred.view(b*t,1,h,w), target.view(b*t,1,h,w)).view(b,t)
    if fulliiee:
        metrics['fulliiee'] = _full_iiee(pred, target, 0.3, cell_areas)

    return metrics