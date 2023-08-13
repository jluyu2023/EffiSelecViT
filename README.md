# EffiSelecViT

Our approach consists of a search and fine-tuning phase:

-  Search

  Performing an importance search on components of the pre-trained model.

- Fine-tuning

  Based on the searched importance proxy scores, following budgets at different scales, we prune the unimportant components, and then perform fine-tuning on the pruned model to recover accuracy.



Here, we take the pruning DeiT-B model as an example and provide the corresponding program to execute the command line.

- Search

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env search.py --data-path /path/to/dataset --model deit_base_patch16_224 --begin_search --search_head --head_w 1e-2 --search_mlp --mlp_w 1e-4 --pretrained_path /path/to/original/pre-trained/checkpoint --output_dir /path/to/save
```

- Fine-tuning

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env finetune.py --retrain --search_checkpoint ./search_checkpoint/deit-base/deit_base/checkpoint.pth --prune_head --head_prune_ratio 0.3 --prune_mlp --mlp_prune_ratio 0.5
```



