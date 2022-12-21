cd $(dirname $0)/../src

# Hierarchical Accelerated Local Search
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=hals
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=hals

# Refine Search
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=refine_search
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=refine_search

# Sampling Uniform Distribution
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=uniform_distribution
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=uniform_distribution

# Adaptive Search
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_method=adaptive_search
python attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_method=adaptive_search
