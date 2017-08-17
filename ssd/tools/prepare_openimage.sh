#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset openimage --set train --target $DIR/../data/train.lst --root $DIR/../data/openimage --target $DIR/../data/openimage/train.lst
# python $DIR/prepare_dataset.py --dataset coco --set minival2014 --target $DIR/../data/val.lst --shuffle False --root $DIR/../data/coco
