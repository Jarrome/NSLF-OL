# [[NSLF-OL] Online Learning of Neural Surface Light Fields alongside Real-time Incremental 3D Reconstruction](https://jarrome.github.io/NSLF-OL/)

This repository contains the implementation of our **RAL 2023** paper: **Online Learning of Neural Surface Light Fields alongside Real-time Incremental 3D Reconstruction**.

[Yijun Yuan](https://jarrome.github.io/), [Andreas Nüchter](https://www.informatik.uni-wuerzburg.de/space/mitarbeiter/nuechter/)

[Preprint]() |  [website](https://jarrome.github.io/NSLF-OL/)

---

## 0. Install
```
conda create -n NSLF-OL python=3.8
conda activate NSLF-OL

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pygame==2.1.2 # dont 2.3.0, will cause problem!
```

