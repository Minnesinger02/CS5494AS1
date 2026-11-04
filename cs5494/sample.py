"""Generate samples from a saved run without retraining.

Reads the architecture back from the run's config.json, so MLP and DCGAN checkpoints
both work, as do linear FVSBNs and multi-layer MADE models.
"""
import argparse,json
from pathlib import Path
import torch
from cs5494.train import Autoregressive, build_gan
from torchvision.utils import save_image
p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('model',choices=['fvsbn','made','gan']); p.add_argument('--seed',type=int,default=123); p.add_argument('--which',choices=['raw','ema'],default='raw',help='GAN only: use model_ema.pt'); a=p.parse_args()
r=Path(a.run); c=json.loads((r/'config.json').read_text()); torch.manual_seed(a.seed)
gan=a.model=='gan'; side=28 if gan else c['side']
if gan:
    m=build_gan(c.get('arch','mlp'))[0]
else:
    m=Autoregressive(side*side,c['hidden'] if a.model=='made' else 0,c.get('layers',1),c.get('dropout',0.0))
ckpt = r/a.model/(('model_ema.pt' if a.which=='ema' else 'model.pt') if gan else 'model.pt')
m.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)); m.eval()
n=64 if gan else 100
with torch.no_grad(): x=(m(torch.randn(n,64))+1)/2 if gan else m.sample(n,'cpu')
tag = 'samples_seed_%d%s.png' % (a.seed, '_ema' if (gan and a.which=='ema') else '')
save_image(x.reshape(-1,1,side,side),r/a.model/tag,nrow=10)
