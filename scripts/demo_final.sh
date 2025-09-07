#!/bin/bash


######################################################################################
# set the img_dir below to the directory of the set of images you want to reconstruct
# set the postfix below to the format of the rgb images in the img_dir
######################################################################################
TEST_DATASET="data/wild/Library" 

# if imgs, you just need to input the dictionary path;
#if video, you need to input the video path like : data/test/myvideo.mp4
# if webcamera you need to input the url like: http://rab:12345678@192.168.137.83:8081

######################################################################################
# set the parameters for whole scene reconstruction below
# for defination of these parameters, please refer to the recon.py
######################################################################################
TEST_NAME="unified_slam3r_reconstruction"
KEYFRAME_STRIDE=3     #-1 for auto-adaptive keyframe stride selection
WIN_R=5
MAX_NUM_REGISTER=10
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5 
CONF_THRES_L2W=12
CONF_THRES_I2P=1.5
NUM_POINTS_SAVE=1000000

UPDATE_BUFFER_INTV=1
BUFFER_SIZE=100       # -1 if size is not limited
BUFFER_STRATEGY="reservoir"  # or "fifo"

KEYFRAME_ADAPT_MIN=1
KEYFRAME_ADAPT_MAX=20
KEYFRAME_ADAPT_STRIDE=1

GPU_ID=-1

python recon_finalversion.py \
--test_name $TEST_NAME \
--dataset "${TEST_DATASET}" \
--gpu_id $GPU_ID \
--keyframe_stride $KEYFRAME_STRIDE \
--win_r $WIN_R \
--num_scene_frame $NUM_SCENE_FRAME \
--initial_winsize $INITIAL_WINSIZE \
--conf_thres_l2w $CONF_THRES_L2W \
--conf_thres_i2p $CONF_THRES_I2P \
--num_points_save $NUM_POINTS_SAVE \
--update_buffer_intv $UPDATE_BUFFER_INTV \
--buffer_size $BUFFER_SIZE \
--buffer_strategy "${BUFFER_STRATEGY}" \
--max_num_register $MAX_NUM_REGISTER \
--keyframe_adapt_min $KEYFRAME_ADAPT_MIN \
--keyframe_adapt_max $KEYFRAME_ADAPT_MAX \
--keyframe_adapt_stride $KEYFRAME_ADAPT_STRIDE \
--save_preds
