cd $(dirname $0)/../src

# Hierarchical Accelerated Local Search
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=hals
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=hals

# Refine Search
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=refine_search
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=refine_search

# Sampling Uniform Distribution
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=uniform_distribution
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=uniform_distribution

# Adaptive Search
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=adaptive_search
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=adaptive_search
