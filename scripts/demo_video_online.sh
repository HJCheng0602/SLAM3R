#!/bin/bash


######################################################################################
# set the video path below to the directory of the video you want to reconstruct
######################################################################################
VIDEO_PATH="data/test/myvideo.mp4"

######################################################################################
# set the parameters for whole scene reconstruction below
# for defination of these parameters, please refer to the recon.py
######################################################################################
TEST_NAME="local_video"
KEYFRAME_STRIDE=3    
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
ENABLE_VIEWER=False   # whether to view the dynamic construction
VIEWER_ARG="" 
if [ "$ENABLE_VIEWER" = "True" ]; then
  
  VIEWER_ARG="--enable_viewer"
fi

SAVE_EACH_FRAME=True
VIEWER_ARG1="" 
if [ "$SAVE_EACH_FRAME" = "True" ]; then
  
  VIEWER_ARG1="--save_each_frame"
fi
PERFRAME=3           #every perframe to be registered
SAVE_FREQUENCY=3     #every save_frequency useful frame to be save to file

GPU_ID=-1

python recon_online_localvideo.py \
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
--video_path $VIDEO_PATH \
$VIEWER_ARG1 \
--perframe $PERFRAME \
--save_frequency $SAVE_FREQUENCY \
$VIEWER_ARG \
--save_preds
