import torch
# from models import vivit
# pre_train_dict = torch.load('pretrain/deit_small_distilled_patch16_224-649709d9.pth')
# print(pre_train_dict['model'].keys())
# model = vivit.ViViT(112, 16, 100, 32)
# model_dict = model.state_dict()
# print(model_dict.keys())

from models import video_transformer

# model = video_transformer.ViViT(num_frames=32,
#                                 img_size=112,
#                                 patch_size=16,
#                                 embed_dims=128,
#                                 num_heads=16,
#                                 num_transformer_layers=4,
#                                 pretrained='pretrain/vivit_model.pth',
#                                 weights_from='kinetics',
#                                 attention_type='divided_space_time',
#                                 use_learnable_pos_emb=False,
#                                 return_cls_token=False).cuda()
# i = 0
# for child in model.children():
#     i+=1
#     if i==2:
#         k=0
#         for c in child.modules():
#             k+=1
#             if k==2 :
#                 for param in c[3].parameters():
#                     param.requires_grad = False

from models import mobilenetv2

model = mobilenetv2.MobileNetV2(num_classes=128, sample_size=112, width_mult=1.).cuda()
pre_train_dict = torch.load('pretrain/kinetics_mobilenetv2_1.0x_RGB_16_best.pth')

dicts = {k[7:]: v for k, v in pre_train_dict['state_dict'].items()}

dicts.pop('classifier.1.bias')
dicts.pop('classifier.1.weight')

model.load_state_dict(dicts, strict=False)

i = 0
for child in model.children():
    i+=1
    if i<2:
        for param in child.parameters():
            param.requires_grad = False



#
# model_dict = model.state_dict()
# print(model_dict.keys())
#
# img = torch.ones([16, 3, 32, 112, 112]).cuda()
#
# out = model(img)
#
# print("Shape of out :", out.shape)  # [B, num_classes]
#
# import numpy as np
# train_label = np.load('train_label.npy')
#
# print(len(train_label))