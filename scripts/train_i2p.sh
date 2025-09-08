#!/bin/bash
MODEL="Image2PointsModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11)"

TRAIN_DATASET="4000 @ ScanNetpp_Seq(filter=True, num_views=11, sample_freq=3, split='train', aug_crop=256, resolution=224, transform=ColorJitter, seed=233) + \
2000 @ Aria_Seq(num_views=11, sample_freq=2, split='train', aug_crop=128, resolution=224, transform=ColorJitter, seed=233) + \
2000 @ Co3d_Seq(num_views=11, sel_num=3, degree=180, mask_bg='rand', split='train', aug_crop=16, resolution=224, transform=ColorJitter, seed=233)"

TEST_DATASET="1000 @ ScanNetpp_Seq(filter=True, num_views=11, split='test', resolution=224, seed=666) + \
1000 @ Aria_Seq(num_views=11, split='test', resolution=224, seed=666) + \
1000 @ Co3d_Seq(num_views=11, sel_num=3, degree=180, mask_bg='rand', split='test', resolution=224, seed=666)"

# Stage 1: Train the i2p model for pointmap prediction
PRETRAINED="checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
TRAIN_OUT_DIR="checkpoints/i2p/slam3r_i2p_stage1"

torchrun --nproc_per_node=8 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, norm_mode='avg_dis')" \
    --pretrained $PRETRAINED \
    --pretrained_type "dust3r" \
    --lr 5e-5 --min_lr 5e-7 --warmup_epochs 10 --epochs 100 --batch_size 4 --accum_iter 1 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 20\
    --save_config\
    --freeze "encoder"\
    --loss_func 'i2p' \
    --output_dir $TRAIN_OUT_DIR \
    --ref_id -1


# Stage 2: Train a simple mlp to predict the correlation score
PRETRAINED="checkpoints/i2p/slam3r_i2p_stage1/checkpoint-final.pth"
TRAIN_OUT_DIR="checkpoints/i2p/slam3r_i2p"

torchrun --nproc_per_node=8 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, gt_scale=True)" \
    --pretrained $PRETRAINED \
    --pretrained_type "slam3r" \
    --lr 1e-4 --min_lr 1e-6 --warmup_epochs 5 --epochs 50 --batch_size 4 --accum_iter 1 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 20\
    --save_config\
    --freeze "corr_score_head_only"\
    --loss_func "i2p_corr_score" \
    --output_dir $TRAIN_OUT_DIR \
    --ref_id -1
