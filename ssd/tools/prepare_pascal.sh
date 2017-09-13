#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target $DIR/../data/VOCdevkit/train.lst
python $DIR/prepare_dataset.py --dataset pascal --year 2007 --set test --target $DIR/../data/VOCdevkit/val.lst --shuffle False
