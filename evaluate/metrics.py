import torch
# Misc
#from https://github.com/barbararoessle/dense_depth_priors_nerf/blob/master/model/run_nerf_helpers.py
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.full((1,), 10., device=x.device))
