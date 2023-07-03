## Code modified from URLNet for multi-class classification, and to apply class important weights

URLNet: https://github.com/Antimalweb/URLNet 

## Build docker image:

docker build -t url0 -f Dockerfile .

## Start docker container with shared host folder:

docker run --name=url10 -t -d -v host-src-and-data-folder:/urlnet/ url0

## Log into the container:

docker exec -it url10 /bin/bash

### Development folder:

cd /urlnet/URLNet

### Train without weights:

python train.py --data.data_dir ../dataset/train_full.multi4.txt

#### Train with rarity weights:

python train.py --data.data_dir ../dataset/train_full.multi4.txt --model.apply_weights True

#### Train with user weights:

python train.py --data.data_dir ../dataset/train_full.multi4.txt --model.apply_weights True --model.preset_weights "0.05,0.35,0.15,0.45"

### Test:

python test.py --data.data_dir ../dataset/test_full.multi4.txt --log.output_dir ../results/test_result.multi4.txt

### Calculate accuracy (with preset user weights):

python calc_wba_acc.py results/test_result.multi4.txt ('0.05,0.35,0.15,0.45')