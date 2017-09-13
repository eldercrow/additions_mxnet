#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset dss --year 2016 --set train --target $DIR/../data/DSS/train.lst
python $DIR/prepare_dataset.py --dataset dss --year 2016 --set val --target $DIR/../data/DSS/val.lst --shuffle False

