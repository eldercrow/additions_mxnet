#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset dss --year 2016 --set trainval --target $DIR/../data/DSS/train.lst --root $DIR/../data/DSS
python $DIR/prepare_dataset.py --dataset dss --year 2016 --set test --target $DIR/../data/DSS/val.lst --shuffle False --root $DIR/../data/DSS

