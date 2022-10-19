# AdEx_BlackBox

## setup
```
./run setup
./run install


    AdEx_BlackBox
    ├── README.md
    ├── config
    │   └── default
    │       ├── AutoPGD.yaml
    │       ├── FGSM.yaml
    │       ├── GenAttack.yaml
    │       ├── GenAttack2.yaml
    │       ├── GenAttack3.yaml
    │       ├── HALS.yaml
    │       ├── PGD.yaml
    │       ├── SquareAttack2.yaml
    │       └── TabuAttack
    │           ├── local_search.yaml
    │           ├── method1_cifar10.yaml
    │           ├── method1_imagenet.yaml
    │           ├── method2_cifar10.yaml
    │           ├── method3_cifar10.yaml
    │           └── tramsfer_tabu_search.yaml
    ├── pyproject.toml
    ├── requirements.txt
    ├── run
    └── src
        ├── attack.py
        ├── attacker
        │   ├── TabuAttack
        │   │   ├── local_search.py
        │   │   ├── method1.py
        │   │   ├── method2.py
        │   │   ├── method3.py
        │   │   ├── method4.py
        │   │   └── method5.py
        │   ├── __init__.py
        │   ├── auto_pgd.py
        │   ├── fgsm.py
        │   ├── gen_attack.py
        │   ├── gen_attack2.py
        │   ├── hals.py
        │   ├── pgd.py
        │   ├── saliency_attack.py
        │   ├── square_attack2.py
        │   └── transfer_gen_attack.py
        ├── base
        │   ├── __init__.py
        │   ├── _load_dataset.py
        │   ├── base_attacker.py
        │   ├── cifar10.json
        │   ├── criterion.py
        │   ├── get_model.py
        │   ├── load_dataset.py
        │   ├── recorder.py
        │   └── transferability.py
        ├── notebook
        │   ├── cifar10.py
        │   ├── model_zoo.py
        │   ├── plot_loss.py
        │   ├── run_attack.py
        │   └── utils.py
        └── utils
            ├── __init__.py
            ├── config_parser.py
            ├── confirmation.py
            ├── logging.py
            ├── processbar.py
            ├── read_gurobi_log.py
            ├── rename.py
            ├── reproducibility.py
            └── timer.py

