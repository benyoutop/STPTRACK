import math
import logging
import os
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from timm.models.layers import to_2tuple
from lib.models.layers.head import CenterPredictor
from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        self.aa = []
        self.aac=0
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #print(self.cls_token)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
        self.queries = []
        self.box=[]
    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        #print(self.cls_token)
        x1=x
        x = self.patch_embed(x)
        
        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query is None:
                    query = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query.size(1)
                    new_query = self.cls_token.expand(B, token_len - track_len, -1)
                    query = torch.cat([new_query, track_query], dim=1)
            elif token_type == "add":
                #print(self.cls_token)
                new_query = self.cls_token.expand(B, token_len, -1)  # copy B times

                query = new_query if track_query is None else track_query + new_query

            query = query + self.cls_pos_embed
            self.queries.append(query)
            ####################gai dong
            if len(self.queries)>4:     #最多存储4个历史信息   #5个为负面效果
                self.queries.pop(0)

            if len(self.queries)>0:
                stacked = torch.stack(self.queries,dim=0)
                query=torch.mean(stacked,dim=0)
            self.aac=self.aac+1

            #######创建全0的token
            memory=torch.zeros(1,3,768)   #test
            #memory = torch.zeros(24, 3, 768)   #train
            memory=memory.cuda()
            len_m=len(memory[0])
            if self.aac>2:
                m1 = self.aa[0].clone()
                caijian = m1
            #if self.aac>50000:
            if self.aac > 2:
                #print("框框計數", self.aa[0][0]) #self.aa  是[0]上一針的box和[1]上上一針的box
            # 2500第一輪結束，5w第20輪結束
            # 假设 x 是 256x256x3 的图像
            #
            # 给定坐标：[center_x, center_y, height, width]
            #
            # 假设 x 是一个 4 维的 Tensor，形状为 [batch_size, channels, height, width]
            # 给定坐标：[center_x, center_y, height, width]
                caijian[:,0].clamp_(min=0,max=1)
                caijian[:, 1].clamp_(min=0, max=1)
                caijian[:, 0].clamp_(min=0)
                caijian[:, 0].clamp_(min=0)
                caijian = (caijian[:, :2]) * 256
                image_height, image_width = x1.shape[2],x1.shape[3]
                cropped_batch = torch.zeros((x1.shape[0],x1.shape[1],32,24), dtype=x1.dtype,device=x1.device)
                for i in range(x1.shape[0]):


                    center_x, center_y = caijian[i][0],caijian[i][1]
                    center_x=max(min(center_x,image_height),0)
                    center_y=max(min(center_y,image_width),0)
                    #print("坐标",center_x,center_y)


            # 计算裁剪区域的左上角和右下角坐标
                    top_left_x = int(center_x - 16)
                    top_left_y = int(center_y - 12)
                    bottom_right_x = int(center_x + 16)
                    bottom_right_y = int(center_y + 12)

                    if top_left_x<0:
                        center_x=16
                    if top_left_y<0:
                        center_y=12
                    if bottom_right_x>image_height:
                        center_x=image_height-16
                    if bottom_right_y>image_width:
                        center_y=image_height-12
                    top_left_x = max(int(center_x-16),0)
                    top_left_y=max(int(center_y-12),0)
                    bottom_right_x=min(int(center_x+16),image_height)
                    bottom_right_y=min(int(center_y+12),image_width)
                    cropped_region=x1[i, :, top_left_x:bottom_right_x,top_left_y:bottom_right_y]
                    if cropped_region.shape[1] != 32 or cropped_region.shape[2] !=24:
                        cropped_region = torch.nn.functional.interpolate(
                            cropped_region.unsqueeze(0),
                            size=(32,24),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        cropped_batch[i]=cropped_region
                    cropped_batch[i,:,:cropped_region.shape[1],:cropped_region.shape[2]]=cropped_region

                focusbox=cropped_batch      #聚焦框
            # 打印裁剪后的形状，应该是 (batch_size, channels, 30, 40)
               # print(focusbox.shape)      # 24X32=768
                memory=focusbox
        memory=memory.view(memory.size(0),memory.size(1),-1)


            #####################gaidong#######################################
        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)

        lens_z = z.shape[1]  # HW
        lens_x = x.shape[1]  # HW

        x = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)     # (B, 1+z+x, 768)
            query_len = query.size(1)

        ##############jiaru memory token#################
        x=combine_tokens(memory,x,mode=self.cat_mode)
        ###########################################
        x = self.pos_drop(x)

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        
        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            if self.add_cls_token:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, 
                        add_cls_token=self.add_cls_token, query_len=query_len)
            else:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, add_cls_token=self.add_cls_token)
                
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        if self.add_cls_token:
            memory=x[:,:len_m]
            query = x[:, len_m:len_m+query_len]
            z = x[:, len_m+query_len:len_m+lens_z_new+query_len]
            x = x[:, len_m+lens_z_new+query_len:]
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([memory,query, z, x], dim=1)

        # aux_dict = {}
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None, 
                token_type="add", token_len=1):
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            track_query=track_query, token_type=token_type, token_len=token_len)
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print('Load pretrained model from: ' + pretrained)
            except:
                print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
