#!/bin/bash

# This is a script to install the slic_reverse package
# i.e. compute slic algorithm with alpha is "minus".

skimage_version="$(python3 -c 'import skimage; print(skimage.__version__)')"

if [ -z "$skimage_version" ]; then
    echo
    echo "Installing scikit-image package"
    pip install -q scikit-image
    return
fi

if (echo "$skimage_version" | grep dev); then

    # install normal slic package
    echo "Installing normal slic package"
    pip uninstall -y -q scikit-image
    pip install -q scikit-image

else

    # install slic_reverse package
    echo "Installing slic_reverse package"
    prefix=$(pip show scikit-image | grep Location | sed 's/Location: //')
    prefix="$prefix/slic_reverse"
    if [ ! -d "$prefix" ]; then
        echo "Building slic_reverse package in $prefix"
        git clone --depth 1 --quiet https://github.com/scikit-image/scikit-image.git "$prefix"
        sed -i -e "s/dist_center += dist_color/dist_center -= dist_color/g" "$prefix/skimage/segmentation/_slic.pyx"
    fi
    pip uninstall -y -q scikit-image
    pip install -q "$prefix"
fi
