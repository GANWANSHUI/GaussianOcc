config=configs/ddad-generate-2d-semantic.txt.txt

python -m torch.distributed.launch --nproc_per_node 1 groundedsam_generate_sem_ddad.py --config $config


