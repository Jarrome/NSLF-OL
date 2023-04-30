# [[NSLF-OL] Online Learning of Neural Surface Light Fields alongside Real-time Incremental 3D Reconstruction](https://jarrome.github.io/NSLF-OL/)

This repository contains the implementation of our **RAL 2023** paper: **Online Learning of Neural Surface Light Fields alongside Real-time Incremental 3D Reconstruction**.

[Yijun Yuan](https://jarrome.github.io/), [Andreas Nüchter](https://www.informatik.uni-wuerzburg.de/space/mitarbeiter/nuechter/)

[Preprint]() |  [website](https://jarrome.github.io/NSLF-OL/)

---

## Install
```
conda create -n NSLF-OL python=3.8
conda activate NSLF-OL

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pygame==2.1.2 # dont 2.3.0, will cause problem!
```

## Quick run
**Prepare data**
* [ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html), for example, [lrkt0n](http://www.doc.ic.ac.uk/~ahanda/living_room_traj0n_frei_png.tar.gz)
* Replica, [Minimal difference to the data in NICE-SLAM. I will upload it somewhere.]

**How to use**
* online learn the NSLF alongside Di-Fusion
```
python nslf_ol.py [config.yaml]
```
First time run will cause some time to compile cpp/cuda code, please use `ps` or `top` to find. Afterwards would be fast!

* visualization with same config
```
python vr.py [config.yaml]
```

Note that, we also provide `_nosurface.py` for only nslf and `multiGPU.py` for multiple GPUs.

**Demo**
```
python nslf_ol.py configs/replica/replica_office0.yaml
python vr.py configs/replica/replica_office0.yaml
```

## TODO
1. Add data
2. Currently we only support visualization after training. But I'm on it. I will find a time to realize on train visualization!

## Acknowledgement
This project is on top of [Di-Fusion](https://github.com/huangjh-pub/di-fusion) from Jiahui Huang, [torch-ngp](https://github.com/ashawkey/torch-ngp) from Jiaxiang Tang. We thank for the open release of those contribution.

## Citation
If you find this code or paper helpful, please cite:
```bibtex
@article{yuan2023online,
  title={Online Learning of Neural Surface Light Fields alongside Real-time Incremental 3D Reconstruction},
  author={Yuan, Yijun and N{\"u}chter, Andreas},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```
