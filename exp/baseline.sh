cd $(dirname $0)/../src

# Parsimonious attack
python attack.py -c ../config/parsimonious_attack.yaml -g 0 -p step=100
python attack.py -c ../config/parsimonious_attack.yaml -g 0 -p step=1000
python attack.py -c ../config/parsimonious_attack.yaml -g 0 -p step=1000

# GenAttack
python attack.py -c ../config/gen_attack.yaml -g 0 -p step=17
python attack.py -c ../config/gen_attack.yaml -g 0 -p step=167
python attack.py -c ../config/gen_attack.yaml -g 0 -p step=1667

# Square Attack
python attack.py -c ../config/gen_attack.yaml -g 0 -p step=100
python attack.py -c ../config/gen_attack.yaml -g 0 -p step=1000
python attack.py -c ../config/gen_attack.yaml -g 0 -p step=10000

# Saliency Attack
python attack.py -c ../config/saliency_attack.yaml -g 0 -p step=100
python attack.py -c ../config/saliency_attack.yaml -g 0 -p step=1000
python attack.py -c ../config/saliency_attack.yaml -g 0 -p step=10000

# Proposed method
python attack.py -c ../config/proposed_method.yaml -g 0 -p step=100
python attack.py -c ../config/proposed_method.yaml -g 0 -p step=1000
python attack.py -c ../config/proposed_method.yaml -g 0 -p step=10000
