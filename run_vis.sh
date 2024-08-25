cd /home/wsgan/project/bev/GaussianOcc

# nuscene
config=configs/nusc-sem-gs.txt
ckpts='ckpts/nusc-sem-gs'
python -m torch.distributed.launch --nproc_per_node=1 run_vis.py --config $config \
--load_weights_folder $ckpts \
--eval_only 

# ddad
config=configs/ddad-sem-gs.txt
ckpts='ckpts/ddad-sem-gs'
python -m torch.distributed.launch --nproc_per_node=1 run_vis.py --config $config \
--load_weights_folder $ckpts \
--eval_only 

# sh /home/wsgan/project/bev/GaussianOcc/run_vis.sh
