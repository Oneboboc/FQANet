"""
TBSI_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.tbsi_track.vit_tbsi_care import vit_base_patch16_224_tbsi, vit_base_patch16_224_ce_adapter
from lib.utils.box_ops import box_xyxy_to_cxcywh


class TBSITrack(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.tbsi_fuse_search = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)

        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -(num_search_token+ num_template_token):-num_template_token, :]
        # enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        # enc_opt2 = cat_feature[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.tbsi_fuse_search(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_tbsi_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('TBSITrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''
        print('No pretrained model specified or excluded models detected.')

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_tbsi':
        backbone = vit_base_patch16_224_tbsi(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                            tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH
                                            )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
        print("Backbone: vit_base_patch16_224_tbsi")
    
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_adapter':
        backbone = vit_base_patch16_224_ce_adapter(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                            tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH,
                                            hhb_loc = cfg.MODEL.BACKBONE.HHB_LOC,
                                            fdnet_loc = cfg.MODEL.BACKBONE.FDNet_LOC,
                                            pcf_loc = cfg.MODEL.BACKBONE.PCF_LOC
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
        print("Backbone: vit_base_patch16_224_ce_adapter")
        

    else:
        raise NotImplementedError

    # hidden_dim = backbone.embed_dim
    # patch_start_index = 1
    """
        elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_adapter':
        backbone = vit_base_patch16_224_ce_adapter(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           adapter_type=cfg.TRAIN.PROMPT.TYPE
                                           )
    """
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = TBSITrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )


    if training and ('TBSITrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE ):
        print(f"Attempting to load pretrained model from: {cfg.MODEL.PRETRAIN_FILE}")
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        new_encoders = ['module.backbone.blocks.12.norm1.weight','module.backbone.blocks.12.norm1.bias','module.backbone.blocks.12.attn.qkv.weight','module.backbone.blocks.12.attn.qkv.bias',
                        'module.backbone.blocks.12.attn.proj.weight','module.backbone.blocks.12.attn.proj.bias','module.backbone.blocks.12.norm2.weight',
                        'module.backbone.blocks.12.norm2.bias','module.backbone.blocks.12.mlp.fc1.weight','module.backbone.blocks.12.mlp.fc1.bias','module.backbone.blocks.12.mlp.fc2.weight',
                        'module.backbone.blocks.12.mlp.fc2.bias']
        values = []
        if 'DropTrack' in cfg.MODEL.PRETRAIN_FILE:
            for k,v in checkpoint["net"].items():
                if k in ['box_head.conv1_ctr.0.weight','box_head.conv1_offset.0.weight','box_head.conv1_size.0.weight']:
                    # v = torch.cat([v,v],1)
                    v = v
                elif 'pos_embed_x' in k:
                    v = resize_pos_embed(v, 16, 16) + checkpoint["net"]['backbone.temporal_pos_embed_x']
                elif 'pos_embed_z' in k:
                    v = resize_pos_embed(v, 8, 8) + checkpoint["net"]['backbone.temporal_pos_embed_z']
                else:
                    v = v
                # if '11' in k:
                #     print(k)
                #     values.append(v)
                param_dict_rgbt[k] = v
            # for i in range(12):
            #     print(new_encoders[i])
            #     param_dict_rgbt[new_encoders[i]] = values[i]
            missing_keys, unexpected_keys = model.load_state_dict(param_dict_rgbt, strict=False)
            print(f"Pretrained model successfully loaded from: {cfg.MODEL.PRETRAIN_FILE}")
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        #print(f"missing_keys: {missing_keys}")
        #print(f"unexpected_keys: {unexpected_keys}")       
    return model


    # if ('TBSITrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE) and training:
    #     pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    #     checkpoint = torch.load(pretrained_file, map_location="cpu")
    #     missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    #     print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    # return model
def resize_pos_embed(posemb, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_grid = posemb[0, :]
    
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to new token with height:{} width: {}'.format(posemb_grid.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb_grid