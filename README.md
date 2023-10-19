# Superpixel Attack

<div align="center"><img src="https://github.com/oe1307/SuperpixelAttack/blob/README/superpixel_attack.png?raw=true"></div>
<div align="center"><h3>Superpixel Attack: Enhancing Black-box Adversarial Attack with Image-driven Division Areas</h3></div>

<div align="center">
    <img src="https://img.shields.io/github/license/oe1307/SuperpixelAttack?logo=open-source-initiative&logoColor=green">
    <img src="https://img.shields.io/badge/python-3.9,3.10-blue.svg">
    <img src="https://img.shields.io/github/last-commit/oe1307/SuperpixelAttack?logo=git&logoColor=white">
    <img src="https://img.shields.io/github/issues/oe1307/SuperpixelAttack?logo=github&logoColor=white">
    <img src="https://img.shields.io/github/issues-pr/oe1307/SuperpixelAttack?logo=github&logoColor=white">
    <img src="https://img.shields.io/github/languages/code-size/oe1307/SuperpixelAttack?logo=github&logoColor=white">
</div>

### Implementation of papers accepted for AJCAI 2023

```
@techreport{oe2023superpixel,
  title={Superpixel Attack: Enhancing Black-Box Adversarial Attack with Image-Driven Division Areas},
  author={Oe, Issa and Yamamura, Keiichiro and Ishikura, Hiroki and Hamahira, Ryo and Fujisawa, Katsuki},
  year={2023},
  institution={EasyChair}
}
```

## Setup

Login [image-net.org](https://image-net.org/login.php) and download
[ILSVRC2012_img_val.tar](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar),
[ILSVRC2012_devkit_t12.tar.gz](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz).

Set `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` in
`storage/data/`.

Then run

```
pip install -r requirements.txt
```

## Usage

```
python main.py -c <path_to_config> -g <GPU_id/mps/cpu> -t <num_thread> -p <params_to_override>
```

For example:

```
python main.py -c ../config/superpixel_attack.yaml -g 0 -t 10
```

```
python main.py -c ../config/sign_hunter.yaml -g 1 -t 20 -p model=Wong2020Fast iter=1000
```
