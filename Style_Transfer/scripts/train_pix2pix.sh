set -ex
python train.py --dataroot ./datasets/edges2shoes --name pix2pix_edges2shoes --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --lr 0.0001 --display_port 60066 --save_epoch_freq 1 --display_env pix2pix_edges2shoes --print_freq 100 --display_freq 100 --batch_size 10 --display_id 1 --n_epochs 100 --n_epochs_decay 100 --q_optim True --gan_mode vanilla