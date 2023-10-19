#!/bin/bash

config=""
device=-1

while getopts c:g: OPT; do
    case $OPT in
        c*) config="$OPTARG" ;;
        g*) device="$OPTARG" ;;
        \?) exit 1 ;;
    esac
done

if [ "$config " = " " ]; then
    echo "error: the following arguments are required: -c CONFIG"
    exit 1
elif [ "$device " = "-1 " ]; then
    echo "error: the following arguments are required: -g DEVICE"
    exit 1
fi

python main.py -c "$config" -g "$device" -p model=Wong2020Fast batch_size=1500 &&
python main.py -c "$config" -g "$device" -p model=Engstrom2019Robustness batch_size=2500 &&
python main.py -c "$config" -g "$device" -p model=Salman2020Do_R50 batch_size=2500 &&
python main.py -c "$config" -g "$device" -p model=Salman2020Do_R18 batch_size=3500 &&
python main.py -c "$config" -g "$device" -p model=Salman2020Do_50_2 batch_size=2000 &&
python main.py -c "$config" -g "$device" -p model=Standard_R50 batch_size=2000 &&
python main.py -c "$config" -g "$device" -p model=Debenedetti2022Light_XCiT-S12 batch_size=3500 &&
python main.py -c "$config" -g "$device" -p model=Debenedetti2022Light_XCiT-M12 batch_size=3000 &&
python main.py -c "$config" -g "$device" -p model=Debenedetti2022Light_XCiT-L12 batch_size=2500 &&
python main.py -c "$config" -g "$device" -p model=Singh2023Revisiting_ViT-S-ConvStem batch_size=2500 &&
python main.py -c "$config" -g "$device" -p model=Singh2023Revisiting_ViT-B-ConvStem batch_size=2500 &&
python main.py -c "$config" -g "$device" -p model=Singh2023Revisiting_ConvNeXt-T-ConvStem batch_size=2000 &&
python main.py -c "$config" -g "$device" -p model=Singh2023Revisiting_ConvNeXt-S-ConvStem batch_size=2000 &&
python main.py -c "$config" -g "$device" -p model=Singh2023Revisiting_ConvNeXt-B-ConvStem batch_size=1000 &&
python main.py -c "$config" -g "$device" -p model=Singh2023Revisiting_ConvNeXt-L-ConvStem batch_size=1000 &&
python main.py -c "$config" -g "$device" -p model=Liu2023Comprehensive_ConvNeXt-B batch_size=1500 &&
python main.py -c "$config" -g "$device" -p model=Liu2023Comprehensive_ConvNeXt-L batch_size=1000 &&
python main.py -c "$config" -g "$device" -p model=Liu2023Comprehensive_Swin-B batch_size=1000 &&
python main.py -c "$config" -g "$device" -p model=Liu2023Comprehensive_Swin-L batch_size=500
