import time
import json

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='./0.01_10epoch_output/checkpoint.pth', type=str)

parser.add_argument('--head_prune', action='store_true')
parser.add_argument('--head_prune_ratio', default=0.3, type=float)

parser.add_argument('--mlp_prune', action='store_true')
parser.add_argument('--mlp_prune_ratio', default=0.3, type=float)
parser.add_argument('--save_output', action='store_true')
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
print('搜索文件的搜索epoch为：', checkpoint['epoch'])
head_zeta = []
mlp_zeta = []
for i in checkpoint['model']:
    if 'head_zeta' in i:
        # print(checkpoint['model'][i])
        head_zeta.append(checkpoint['model'][i])
    if 'mlp_zeta' in i:
        mlp_zeta.append(checkpoint['model'][i])

print('开始处理head-data')
# print(head_zeta)
# print(head_zeta[0].shape)
head_data = []
for i in range(len(head_zeta)):
    head_data.append(head_zeta[i].squeeze().reshape(-1).numpy().tolist())

head_data = [z for k in head_data for z in k]  # 进一步展平，展平为一行
head_data = sorted(head_data)
print('head_data:', head_data)
min_index = int(args.head_prune_ratio * len(head_data))
threshold = head_data[min_index - 1]
print('head_threshold=', threshold, type(threshold))


head_cfg_mask = []
for i in range(len(head_zeta)):
    head_cfg_mask.append(
        (head_zeta[i] > threshold).int()

        # 得到每层layer的0，1值
    )
# print(head_cfg_mask[0], type(head_cfg_mask[0]))
# # print('head_cfg_mask:', head_cfg_mask)
# # print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
# time.sleep(20)
# print(torch.cuda.memory_summary())
head_cfg_mask_data = []
for i in range(len(head_cfg_mask)):
    print(head_cfg_mask[i].squeeze().reshape(-1).numpy().tolist())
    head_cfg_mask_data.append(head_cfg_mask[i].squeeze().numpy().tolist())

print(head_cfg_mask_data)

if args.save_output:
    save_dict = {}
    sub_dict = {'head_threshold': threshold, 'head-mask-cfg': head_cfg_mask_data}
    save_dict['ft_' + str(args.checkpoint_path) + str(args.prune_ratio)] = sub_dict
    with open('./head_mask_cfg.json', 'w+', encoding='utf8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)


print('开始处理mlp-data')
# print(head_zeta)
# print(head_zeta[0].shape)
mlp_data = []
for i in range(len(mlp_zeta)):
    mlp_data.append(mlp_zeta[i].squeeze().reshape(-1).numpy().tolist())

mlp_data = [z for k in mlp_data for z in k]  # 进一步展平，展平为一行
mlp_data = sorted(mlp_data)
print('mlp_data:', mlp_data)
min_index = int(args.mlp_prune_ratio * len(mlp_data))
threshold = mlp_data[min_index - 1]
print('mlp_threshold=', threshold, type(threshold))


mlp_cfg_mask = []
for i in range(len(mlp_zeta)):
    mlp_cfg_mask.append(
        (mlp_zeta[i] > threshold).int()

        # 得到每层layer的0，1值
    )
# print(head_cfg_mask[0], type(head_cfg_mask[0]))
# # print('head_cfg_mask:', head_cfg_mask)
# # print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
# time.sleep(20)
# print(torch.cuda.memory_summary())
mlp_cfg_mask_data = []
for i in range(len(mlp_cfg_mask)):
    print(mlp_cfg_mask[i].squeeze().reshape(-1).numpy().tolist())
    mlp_cfg_mask_data.append(mlp_cfg_mask[i].squeeze().numpy().tolist())

print(mlp_cfg_mask_data)

if args.save_output:
    save_dict = {}
    sub_dict = {'head_threshold': threshold, 'head-mask-cfg': head_cfg_mask_data}
    save_dict['ft_' + str(args.checkpoint_path) + str(args.prune_ratio)] = sub_dict
    with open('./head_mask_cfg.json', 'w+', encoding='utf8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)

