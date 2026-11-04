"""CS5494: FVSBN, neural autoregressive MADE, and GAN on MNIST."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MaskedLinear(nn.Linear):
    def __init__(self, ni, no, mask):
        super().__init__(ni, no)
        self.register_buffer('mask', mask.float())
    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class Autoregressive(nn.Module):
    def __init__(self, dim, hidden=0, layers=1, dropout=0.0):
        super().__init__()
        self.dim = dim
        # Output i can only see pixels j < i; never the current/future pixel.
        degrees = torch.arange(dim)
        if hidden:
            # MADE ordering: hidden degree cycles over 0..dim-2. Every edge keeps the
            # hidden degree >= its input degree, and the output edge is strictly greater,
            # so no path can carry information from pixel i or later into output i.
            h = torch.arange(hidden) % (dim - 1)
            seq = [MaskedLinear(dim, hidden, h[:,None] >= degrees[None,:]), nn.ReLU()]
            if dropout > 0: seq.append(nn.Dropout(dropout))
            for _ in range(max(1, layers) - 1):
                seq += [MaskedLinear(hidden, hidden, h[:,None] >= h[None,:]), nn.ReLU()]
                if dropout > 0: seq.append(nn.Dropout(dropout))
            seq += [MaskedLinear(hidden, dim, degrees[:,None] > h[None,:])]
            self.net = nn.Sequential(*seq)
        else:
            self.net = MaskedLinear(dim, dim, degrees[:,None] > degrees[None,:])
    def forward(self, x):
        return self.net(x)
    @torch.no_grad()
    def sample(self, n, device):
        x = torch.zeros(n, self.dim, device=device)
        for i in range(self.dim):
            x[:,i] = torch.bernoulli(self(x)[:,i].sigmoid())
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64,256),nn.LeakyReLU(.2),nn.Linear(256,512),nn.LeakyReLU(.2),nn.Linear(512,784),nn.Tanh())
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784,512),nn.LeakyReLU(.2),nn.Linear(512,256),nn.LeakyReLU(.2),nn.Linear(256,1))
    def forward(self, x): return self.net(x).flatten()

class DCGenerator(nn.Module):
    """DCGAN generator, 64 -> 7x7 -> 14x14 -> 28x28, returned flattened like the MLP one."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64,128*7*7), nn.Unflatten(1,(128,7,7)), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,1,4,2,1), nn.Tanh(), nn.Flatten(1))
    def forward(self, z): return self.net(z)

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1,(1,28,28)),
            nn.Conv2d(1,64,4,2,1), nn.LeakyReLU(.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(.2),
            nn.Flatten(1), nn.Linear(128*7*7,1))
    def forward(self, x): return self.net(x).flatten()

def build_gan(arch):
    if arch == 'dcgan':
        return DCGenerator(), DCDiscriminator()
    return Generator(), Discriminator()

def run(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    if device.type=='cuda':
        # Fixed-size layers: benchmark off, but allow TF32 matmuls on Ampere+ for speed.
        torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
        torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a)|{'resolved_device':str(device),'device_name':torch.cuda.get_device_name(0) if device.type=='cuda' else str(device)},indent=2))
    train = MNIST(a.data,train=True,download=True,transform=transforms.ToTensor())
    test = MNIST(a.data,train=False,download=True,transform=transforms.ToTensor())
    indices = torch.randperm(len(train))[:a.samples]
    torch.save(indices,out/'training_indices.pt')
    raw = train.data[indices].float().unsqueeze(1)/255
    test_raw = test.data[:a.test_samples].float().unsqueeze(1)/255
    metrics = {}
    for name in a.models:
        folder=out/name; folder.mkdir(exist_ok=True)
        gan = name == 'gan'; side=28 if gan else a.side
        x = raw.flatten(1)*2-1 if gan else (F.interpolate(raw,size=(side,side),mode='area')>=.5).float().flatten(1)
        # Optional held-out split used only for hyperparameter selection (never for gradients).
        if not gan and a.val > 0:
            xval, xtr = x[:a.val], x[a.val:]
        else:
            xval, xtr = None, x
        loader=DataLoader(TensorDataset(xtr),batch_size=a.batch,shuffle=True,num_workers=a.workers,
                          pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
        if gan:
            gen, discr = build_gan(a.arch)
            model = gen.to(device); disc = discr.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(.5,.999))
            dop=torch.optim.Adam(disc.parameters(),lr=a.lr_d if a.lr_d else a.lr,betas=(.5,.999))
            fixed=torch.randn(20,64,device=device)
            # Exponential moving average of the generator parameters. Buffers such as
            # BatchNorm running statistics are copied from the live model at save time.
            ema = ({k: v.detach().clone() for k, v in model.named_parameters()}
                   if a.ema > 0 else None)
        else:
            model=Autoregressive(side*side, a.hidden if name=='made' else 0,
                                 a.layers if name=='made' else 1,
                                 a.dropout if name=='made' else 0.0).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=a.lr,betas=(.9,.999),weight_decay=a.weight_decay)
        sched=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*.02)
               if a.scheduler=='cosine' else None)
        dsched=(torch.optim.lr_scheduler.CosineAnnealingLR(dop,T_max=a.epochs,eta_min=(a.lr_d or a.lr)*.02)
                if gan and a.scheduler=='cosine' else None)
        history=[]; snapshots=[]; start=time.perf_counter()
        for epoch in range(a.epochs):
            e0=time.perf_counter()
            total=0.; dtotal=0.; count=0
            for (batch,) in loader:
                batch=batch.to(device,non_blocking=True); n=len(batch)
                if gan:
                    dop.zero_grad(); fake=model(torch.randn(n,64,device=device)).detach()
                    dl=F.binary_cross_entropy_with_logits(disc(batch),torch.ones(n,device=device))+F.binary_cross_entropy_with_logits(disc(fake),torch.zeros(n,device=device))
                    dl.backward(); dop.step(); dtotal+=dl.item()*n
                    disc.requires_grad_(False)
                    opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(disc(model(torch.randn(n,64,device=device))),torch.ones(n,device=device))
                    loss.backward(); opt.step(); disc.requires_grad_(True)
                    if ema is not None:
                        with torch.no_grad():
                            for k, p in model.named_parameters():
                                ema[k].mul_(a.ema).add_(p, alpha=1-a.ema)
                else:
                    opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(model(batch),batch,reduction='none').sum(1).mean()
                    loss.backward(); opt.step()
                total+=loss.item()*n; count+=n
            row={'epoch':epoch+1,'loss':total/count,'seconds':time.perf_counter()-e0}
            if gan: row['discriminator_loss']=dtotal/count
            history.append(row); print(name,{k:(round(v,4) if isinstance(v,float) else v) for k,v in row.items()},flush=True)
            if sched: sched.step()
            if dsched: dsched.step()
            if epoch in {0,a.epochs//4,a.epochs//2,3*a.epochs//4,a.epochs-1}:
                model.eval()
                with torch.no_grad(): samples=(model(fixed)+1)/2 if gan else model.sample(20,device)
                model.train()
                images=samples.cpu().reshape(-1,1,side,side); save_image(images,folder/f'epoch_{epoch+1:03}.png',nrow=5); snapshots.append(images[:4])
        if device.type=='mps': torch.mps.synchronize()
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        model.eval()
        inference_start=time.perf_counter()
        with torch.no_grad(): samples=(model(torch.randn(100,64,device=device))+1)/2 if gan else model.sample(100,device)
        images=samples.cpu().reshape(-1,1,side,side)
        infer=time.perf_counter()-inference_start
        save_image(images,folder/'samples.png',nrow=10)
        save_image(torch.cat(snapshots),folder/'progress.png',nrow=4)
        torch.save(model.state_dict(),folder/'model.pt')
        if gan:
            torch.save(disc.state_dict(),folder/'discriminator.pt')
            if ema is not None:
                sd = {k: v.clone() for k, v in model.state_dict().items()}
                for k, v in ema.items():
                    sd[k] = v.clone()
                torch.save(sd, folder/'model_ema.pt')
        result={'training_and_preview_seconds':elapsed,'inference_100_seconds':infer,'training_samples':len(xtr)}
        if gan and ema is not None:
            result['ema_decay'] = a.ema
        if not gan:
            def _nll(batch_x):
                with torch.no_grad():
                    return sum(F.binary_cross_entropy_with_logits(model(b.to(device)),b.to(device),reduction='sum').item()
                               for b in batch_x.split(a.batch))/len(batch_x)
            tx=(F.interpolate(test_raw,size=(side,side),mode='area')>=.5).float().flatten(1)
            nll=_nll(tx)
            result|={'test_nll_nats_per_image':nll,'test_bits_per_pixel':nll/(side*side*np.log(2))}
            if xval is not None:
                vnll=_nll(xval)
                result|={'validation_samples':len(xval),'validation_nll_nats_per_image':vnll,
                         'validation_bits_per_pixel':vnll/(side*side*np.log(2))}
        metrics[name]=result
        print(name,'summary',{k:round(v,4) for k,v in result.items()},flush=True)
        (folder/'history.json').write_text(json.dumps(history,indent=2))
        plt.figure(); plt.plot([r['epoch'] for r in history],[r['loss'] for r in history],label='G BCE' if gan else 'NLL (nats/image)')
        if gan: plt.plot([r['epoch'] for r in history],[r['discriminator_loss'] for r in history],label='D BCE')
        plt.xlabel('Epoch'); plt.legend(); plt.tight_layout(); plt.savefig(folder/'loss.png'); plt.close()
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',choices=['fvsbn','made','gan'],default=['fvsbn','made','gan'])
    p.add_argument('--data',default='data'); p.add_argument('--output',default='outputs/main')
    p.add_argument('--samples',type=int,default=20000); p.add_argument('--test-samples',type=int,default=2000)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=128)
    p.add_argument('--side',type=int,default=14); p.add_argument('--hidden',type=int,default=512)
    p.add_argument('--layers',type=int,default=1,help='masked hidden layers for MADE')
    p.add_argument('--dropout',type=float,default=0.0,help='dropout on MADE hidden activations')
    p.add_argument('--weight-decay',dest='weight_decay',type=float,default=0.0)
    p.add_argument('--val',type=int,default=0,help='leading samples held out as validation (autoregressive only)')
    p.add_argument('--arch',choices=['mlp','dcgan'],default='mlp',help='GAN generator/discriminator architecture')
    p.add_argument('--scheduler',choices=['none','cosine'],default='none')
    p.add_argument('--ema',type=float,default=0.0,
                   help='decay for an exponential moving average of the GAN generator (0 disables)')
    p.add_argument('--lr',type=float,default=.0002); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto')
    p.add_argument('--lr-d',dest='lr_d',type=float,default=0.0,
                   help='optional discriminator learning rate (TTUR); 0 keeps --lr')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--tf32',type=int,default=1,choices=[0,1])
    a=p.parse_args()
    if not (2<=a.side<=28 and a.samples>0 and a.test_samples>0 and a.epochs>0 and a.batch>0): p.error('invalid dimensions or counts')
    run(a)
