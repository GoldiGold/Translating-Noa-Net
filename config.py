

CONF = {
        'class_num'          : 1,   # background is not taken in count
        'loss'               : 'dice',
        'batch_size'         : 1,
        'kernel_size'        : 3,
        'ynet_ch'            : [20, 40, 80, 160, 320], #[12, 24, 48, 96, 192],# [16, 32, 64, 128, 256], #[8, 16, 32, 64, 128], #  [10, 20, 40, 80, 160], #[6, 12, 24, 48, 96], # [12, 24, 48, 96, 192],
        'vnet_ch'            : [16, 32, 64, 128, 256],
        'wnet_ch'            : [8, 16, 32, 64, 128],    # original size (causes OOM): [10, 20, 40, 80, 160]
        'dropout'            : 0.5,
        'learning_rate'      : 0.0001,
        'alpha'              : 0.5,
        'beta'               : 0,
        'lr_decay'           : 0.1,
        'warmup_epochs'      : 1,
        'dice_epochs'        : 1,
        'joint_train_epochs' : 120,
        'max_epochs'         : 101, #110, 100 for STN , 100 regular
        'min_dice'           : 0.08,
        't1_hist_match'      : False,
        'shuffle'            : True,
        'use_STN'            : True,
        'dump_features'      : False, # Use only with batch size 1
        'use_color_fa'       : False,
        'use_vista_aligne'   : False,
        'rigid_file'         : 'regular',# regular or large
        'connect_tresh_epoch': 40,
        'wm_seg'             : True,
        'train_all'          : False,
        'transfer_learning'  : False,
        'use_augment'        : False, # if HCP turn off
        'STN_only'           : False,
        'stn_iterations'     : 4
        }

# Features Explanations
# -------------------------- General Features ---------------------- #
# dump_features       : For debug purposes save all debug layers to nii/txt files .
# use_color_fa        : Ynet_AG_v2 is able to receive colorFA or FA and PDD.
#                       If we want to use colorFA as input, we should use 'use_color_fa' feature and then the FA convertet to all ones metrix.
# wm_seg              : There are 2 modes: segment tracts and segment WM. If wm_seg is on the the net will segment WM.
# train_all           : For white matter segmentation - training set will be trained on all of the dataset
# connect_tresh_epoch : Threshold epoch fo connectivity loss optimizer
# transfer_learning   : Whether to peform transfer learning from the WM segmentation network that was trained on all 186 cases
# alpha and beta      : Those are control parameters of the connectivity loss
# ---------------------- Type of Training Epochs ------------------- #
# warmup_epochs (STN) -> warmup_epochs_seg -> joint_train_epochs -> dice_epochs
# warmup_epochs      : The amount of initial epochs for training only the STN network (freeze all other weights).
# joint_train_epochs : The epoch that indicates that we begin to train both the STN and segmentation networks.
# dice_epochs        : The epoch that indicates that we begin to train only the segmentation network at the end.
# warmup_epochs_seg  : The epoch that indicates that we begin to train only the segmentation network at the begining.
#
# Sheba75
# MRI_DIMS = (128, 140, 112)
# MRI_DIMS = (128, 160, 128)

# HCP105
# MRI_DIMS = (112, 140, 112)
# MRI_DIMS = (128, 144, 128)

# Parameters count:
# Ynet: Number of parameters: [16, 32, 64, 128, 256] 19.360 M
# Ynet: Number of parameters: [16, 32, 64, 128, 212] 15.629 M
# Ynet: Number of parameters: [12, 32, 64, 128, 212] 15.547 M
# Ynet: Number of parameters: [10, 32, 64, 128, 204] 15.448 M
# Ynet: Number of parameters: [16, 32, 64, 128, 128] 11.266 M

# Vnet: Number of parameters: [16, 32, 64, 128, 256] 15.553 M
# Vnet: Number of parameters: [16, 32, 64, 128, 202] 11.259 M

# Old paramters count:
# Ynet: Number of parameters: 46.380 M
# Wnet: Number of parameters: 32.862 M
# VnetT1: Number of parameters: 47.136 M
# VnetRGB: Number of parameters: 47.140 M
# VnetRGBT1: Number of parameters: 47.142 M

# AGF_Ynet_V2_STN + locnet channels before dense = 4 : Number of parameters: [6, 12, 24, 48, 96] 20.702 M
# AGF_Ynet_V2_STN + locnet channels before dense = 3 : Number of parameters: [6, 12, 24, 48, 96] 16.202 M
# AGF_Ynet_V2_STN : Number of parameters: [6, 12, 24, 48, 96] 6.092 M
# AGF_Yent_V2 : Number of parameters: [6, 12, 24, 48, 96] 2.664 M

####################  WNSeg #############################
# Example WNSeg how to run:
# command line:
# --train --dataset sheba --fibers WMSeg --folds 0 --nets Ynet_AGF_v2
# Required configurations-
# 'warmup_epochs': 0,  # 4,
# 'dice_epochs': 115,  # 90,
# 'joint_train_epochs': 120,  # 4,
# 'max_epochs': 115,
# 'min_dice': 0.08,
# 't1_hist_match': False,
# 'shuffle': True,
# 'use_STN': False,
# 'dump_features': True,
# 'use_color_fa': False,
# 'use_vista_aligne': False,
# 'rigid_file': 'large',
# 'connect_tresh_epoch': 40,
# 'wm_seg': True,
# 'train_all': True,
# 'transfer_learning': False,
# 'use_augment'        : False # if HCP turn off
####################  WNSegSTN #############################
# Example WNSegSTN how to run:
# command line:
# --train --dataset sheba --fibers WMSegSTN --folds 0 --nets Ynet_AGF_STN
# Required configurations-
# 'warmup_epochs': 1,  # 4,
# 'dice_epochs': 1,  # 90,
# 'joint_train_epochs': 120,  # 4,
# 'max_epochs': 101,
# 'min_dice': 0.08,
# 't1_hist_match': False,
# 'shuffle': True,
# 'use_STN': True,
# 'dump_features': False,
# 'use_color_fa': False,
# 'use_vista_aligne': False,
# 'rigid_file': 'regular',
# 'connect_tresh_epoch': 40,
# 'wm_seg': True,
# 'train_all': True,
# 'transfer_learning': False,
# 'use_augment'        : True # if HCP turn off
# NOTE - this run is very heavy and might couse an error code of 137 - in order to fix it define the dataset_test, dataset_val as the dataset_train

####################  Ynet_AGF_STN #############################
# command line:
#--train --dataset sheba75 --fibers MotorLeft --folds 0 --nets Ynet_AGF_STN
# Required configurations-
# 'alpha'              : 1,
# 'beta'               : 0,
# 'lr_decay'           : 0.1,  # TODO: return to original 1.0
# 'warmup_epochs'      : 4,
# 'dice_epochs'        : 90,
# 'joint_train_epochs' : 4,
# 'max_epochs'         : 100,
# 'min_dice'           : 0.08,
# 't1_hist_match'      : False,
# 'shuffle'            : True,
# 'use_STN'            : True,
# 'dump_features'      : False,
# 'use_color_fa'       : False,
# 'use_vista_aligne'   : False,
# 'rigid_file'         : 'regular',
# 'connect_tresh_epoch': 40,
# 'wm_seg'             : False,
# 'train_all'          : False,
# 'transfer_learning'  : True,
# 'use_augment'        : True # if HCP turn off


############## Run STN only with no segmentation
# warmup_epochs'      : 75,#4,#0,
# 'dice_epochs'        : 90,
# 'joint_train_epochs' : 100,#4,
# 'max_epochs'         : 65,#101,
# 'min_dice'           : 0.08,
# 't1_hist_match'      : False,
# 'shuffle'            : True,
# 'use_STN'            : True,
# 'dump_features'      : False,
# 'use_color_fa'       : False,
# 'use_vista_aligne'   : False,
# 'rigid_file'         : 'regular',
# 'connect_tresh_epoch': 40,
# 'wm_seg'             : False,
# 'train_all'          : False,
# 'transfer_learning'  : True,
# 'use_augment'        : True, # if HCP turn off
# 'STN_only'           : True