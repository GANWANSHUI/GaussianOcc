config=configs/ddad-sem-gs.txt


python -m torch.distributed.launch --nproc_per_node 1 groundedsam_generate_sem_ddad.py --config $config


