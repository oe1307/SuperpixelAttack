cd $(dirname $0)/../src

# equally_divided_squares
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_area=equally_divided_squares
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_area=equally_divided_squares

# random_square
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_area=random_square
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_area=random_square

# saliency_map
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_area=saliency_map
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_area=saliency_map

# superpixel
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=100 update_area=superpixel
python3 attack.py -c ../config/proposed_method_exp.yaml -g 0 -p step=1000 update_area=superpixel
