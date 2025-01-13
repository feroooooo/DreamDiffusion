import torch

# sd = torch.load("./pretrains/models/dreamdiffusion_finetune_checkpoint.pth")

# print(sd.keys())
# print(sd['config'])
# # print(sd['model_state_dict'])
# print(sd['state'].shape)
# print(type(sd['model_state_dict']))
# print(type(sd['config']))
# print(type(sd['state']))

eeg = torch.load("./datasets/eeg_5_95_std.pth")
# dict_keys(['dataset', 'labels', 'images'])
print(eeg.keys())
# list
print(type(eeg['dataset']))
print(type(eeg['images']))
print(type(eeg['labels']))
print(len(eeg['dataset'])) # 11965
print(len(eeg['images'])) # 1996
print(len(eeg['labels'])) # 40
print(eeg['dataset'][0]) # {'eeg': tensor(...), 'image': 0, 'label': 10, 'subject': 4}
print(eeg['images'][0]) # n02951358_31190
print(eeg['labels'][0]) # n02389026
print(eeg['dataset'][0]['eeg'].shape) # torch.Size([128, 500])

# 6个被试
eeg = torch.load("./datasets/block_splits_by_image_all.pth")

print(eeg.keys()) # dict_keys(['splits'])
idx = 0
print(type(eeg['splits'])) # list
print(len(eeg['splits'])) # 6（源代码idx永远取0，似乎已包括6个被试，不知道后面的有什么用）
print(type(eeg['splits'][idx])) # dict
print(eeg['splits'][idx].keys()) # dict_keys(['train', 'val', 'test'])
print(type(eeg['splits'][idx]['train'])) # list
print(type(eeg['splits'][idx]['val'])) # list
print(type(eeg['splits'][idx]['test'])) # list
# 加起来是11965
print(len(eeg['splits'][idx]['train'])) # 7970
print(len(eeg['splits'][idx]['val'])) # 1998
print(len(eeg['splits'][idx]['test'])) # 1007
print(eeg['splits'][idx]['train'][0]) # 0 dataset idx

# 单个被试
eeg = torch.load("./datasets/block_splits_by_image_single.pth")

print(eeg.keys()) # dict_keys(['splits'])
print(type(eeg['splits'])) # list
print(len(eeg['splits'])) # 1
print(type(eeg['splits'][0])) # dict
print(eeg['splits'][0].keys()) # dict_keys(['train', 'val', 'test'])
print(type(eeg['splits'][0]['train'])) # list
print(type(eeg['splits'][0]['val'])) # list
print(type(eeg['splits'][0]['test'])) # list
# 这里加起来是1000，而不是11965 / 6，不清楚原因
print(len(eeg['splits'][0]['train'])) # 669
print(len(eeg['splits'][0]['val'])) # 167
print(len(eeg['splits'][0]['test'])) # 164
print(eeg['splits'][0]['train'][0]) # 0 dataset idx