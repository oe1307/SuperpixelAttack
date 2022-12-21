cd $(dirname $0)/../src

# Parsimonious attack
python3 attack.py -c ../config/parsimonious_attack.yaml -g 0 -p step=100
python3 attack.py -c ../config/parsimonious_attack.yaml -g 0 -p step=1000
python3 attack.py -c ../config/parsimonious_attack.yaml -g 0 -p step=10000

# GenAttack
python3 attack.py -c ../config/gen_attack.yaml -g 0 -p step=17
python3 attack.py -c ../config/gen_attack.yaml -g 0 -p step=167
python3 attack.py -c ../config/gen_attack.yaml -g 0 -p step=1667

# Square Attack
python3 attack.py -c ../config/gen_attack.yaml -g 0 -p step=100
python3 attack.py -c ../config/gen_attack.yaml -g 0 -p step=1000
python3 attack.py -c ../config/gen_attack.yaml -g 0 -p step=10000

# Saliency Attack
python3 attack.py -c ../config/saliency_attack.yaml -g 0 -p step=100
python3 attack.py -c ../config/saliency_attack.yaml -g 0 -p step=1000
python3 attack.py -c ../config/saliency_attack.yaml -g 0 -p step=10000

# Proposed method
python3 attack.py -c ../config/proposed_method.yaml -g 0 -p step=100
python3 attack.py -c ../config/proposed_method.yaml -g 0 -p step=1000
python3 attack.py -c ../config/proposed_method.yaml -g 0 -p step=10000
