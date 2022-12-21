# AdEx_BlackBox

## Requires

    python 3.9 or 3.10
    cuda 11.6

## Setup

1. install modules

```
./run install
```

2. download ImageNet and models

Login [image-net.org](https://image-net.org/login.php) and download [ILSVRC2012_img_val.tar](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and [ILSVRC2012_devkit_t12.tar.gz](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz).

Set `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` in `storage/data/`.

Go to [sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection](https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection) and download pretrained [saliency model](https://drive.google.com/file/d/1Sc7dgXCZjF4wVwBihmIry-Xk7wTqrJdr/view).

Set `best-model_epoch-204_mae-0.0505_loss-0.1370.pth` in `storage/saliency/model/saliency/`.

then

```
./run setup
```

## Experiment

Table1 and Table2: Comparison with baseline

```
sh exp/baseline.sh
```

Table3: Update Area

```
sh exp/update_area.sh
```

Table4: Update Method

```
sh exp/update_method.sh
```

## Attacker

#### Proposed Method

    Using Superpixel Data-driven Black-box Adversarial Attack

#### Baselines

Parsimonious attack

    Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization

GenAttack (advertorch)

    GenAttack: practical black-box attacks with gradient-free optimization

SquareAttack (torchattacks)

    Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search

SaliencyAttack

    Saliency Attack: Towards Imperceptible Black-box Adversarial Attack
