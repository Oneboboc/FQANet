from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.tbsi_track.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch = None):
    params = TrackerParams()
    save_dir = env_settings().save_dir
    check_dir = env_settings().check_dir
    # update default config from yaml file
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    yaml_file = os.path.join(prj_dir, 'experiments/tbsi_track/%s.yaml' % yaml_name)
    
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)     # next cancle
    # test config:  {'MODEL': {'PRETRAIN_FILE': 'pretrained_models/DropTrack_k700_800E_alldata.pth.tar', 'EXTRA_MERGER': False, 'RETURN_INTER': False, 'RETURN_STAGES': [2, 5, 8, 11], 'BACKBONE': {'TYPE': 'vit_base_patch16_224_ce_adapter', 'STRIDE': 16, 'MID_PE': False, 'SEP_SEG': False, 'CAT_MODE': 'direct', 'MERGE_LAYER': 0, 'ADD_CLS_TOKEN': False, 'CLS_TOKEN_USE_MODE': 'ignore', 'CE_LOC': [], 'CE_KEEP_RATIO': [], 'CE_TEMPLATE_RANGE': 'ALL', 'TBSI_LOC': [3, 6, 9], 'RGB_ONLY': False, 'RGBT_UNSHARE': False}, 'HEAD': {'TYPE': 'CENTER', 'NUM_CHANNELS': 256}}, 'TRAIN': {'LR': 0.0001, 'WEIGHT_DECAY': 0.0001, 'EPOCH': 15, 'LR_DROP_EPOCH': 10, 'BATCH_SIZE': 32, 'NUM_WORKER': 8, 'OPTIMIZER': 'ADAMW', 'BACKBONE_MULTIPLIER': 0.1, 'GIOU_WEIGHT': 2.0, 'L1_WEIGHT': 5.0, 'FREEZE_LAYERS': [0], 'PRINT_INTERVAL': 50, 'VAL_EPOCH_INTERVAL': 10, 'GRAD_CLIP_NORM': 0.1, 'AMP': True, 'CE_START_EPOCH': 20, 'CE_WARM_EPOCH': 80, 'DROP_PATH_RATE': 0.1, 'TBSI_DROP_RATE': 0.0, 'TBSI_DROP_PATH': [0.0, 0.0, 0.0], 'SOT_PRETRAIN': True, 'SCHEDULER': {'TYPE': 'step', 'DECAY_RATE': 0.1}}, 'DATA': {'SAMPLER_MODE': 'causal', 'MEAN': [0.485, 0.456, 0.406, 0.449, 0.449, 0.449], 'STD': [0.229, 0.224, 0.225, 0.226, 0.226, 0.226], 'MAX_SAMPLE_INTERVAL': 200, 'TRAIN': {'DATASETS_NAME': ['LasHeR_train'], 'DATASETS_RATIO': [1], 'SAMPLE_PER_EPOCH': 60000}, 'VAL': {'DATASETS_NAME': ['LasHeR_test'], 'DATASETS_RATIO': [1], 'SAMPLE_PER_EPOCH': 10000}, 'SEARCH': {'SIZE': 256, 'FACTOR': 4.0, 'CENTER_JITTER': 3, 'SCALE_JITTER': 0.25, 'NUMBER': 1}, 'TEMPLATE': {'NUMBER': 1, 'SIZE': 128, 'FACTOR': 2.0, 'CENTER_JITTER': 0, 'SCALE_JITTER': 0}}, 'TEST': {'TEMPLATE_FACTOR': 2.0, 'TEMPLATE_SIZE': 128, 'SEARCH_FACTOR': 4.0, 'SEARCH_SIZE': 256, 'EPOCH': 15}}


    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # print("params_begin")
    # params.checkpoint = os.path.join(prj_dir, "./output35/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/checkpoints/train/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/TBSITrack_ep0008.pth.tar")
    # params.checkpoint = os.path.join(prj_dir, "./output5rgbt/checkpoints/train/tbsi_track/rgbt/TBSITrack_ep0013.pth.tar")
    # /media/data4/gbs/Code/TBSI-re/output21/DMET/checkpoints/train/tbsi_track/DMET/TBSITrack_ep0013.pth.tar
    ###  params.checkpoint = os.path.join(prj_dir, "./outputD22/DMET/checkpoints/train/tbsi_track/DMET/TBSITrack_ep0015.pth.tar")
    print("params.checkpoint")
    # print(params.checkpoint)
    # print("params_end")
    # print(params.checkpoint)    
    # output32/rgbt/checkpoints/train/tbsi_track/rgbt/TBSITrack_ep0013.pth.tar      
    # params.checkpoint = os.path.join(prj_dir, "./output/checkpoints/train/bat/rgbt/BATrack_ep0013.pth.tar")
    # whether to save boxes from all queries

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
