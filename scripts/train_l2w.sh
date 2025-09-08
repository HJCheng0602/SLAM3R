#!/bin/bash
MODEL="Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 12, need_encoder=True)"

TRAIN_DATASET="4000 @ ScanNetpp_Seq(filter=True, sample_freq=3, num_views=13, split='train', aug_crop=256, resolution=224, transform=ColorJitter, seed=233) + \
2000 @ Aria_Seq(num_views=13, sample_freq=2, split='train', aug_crop=128, resolution=224, transform=ColorJitter, seed=233) + \
2000 @ Co3d_Seq(num_views=13, sel_num=3, degree=180, mask_bg='rand', split='train', aug_crop=16, resolution=224, transform=ColorJitter, seed=233)"
TEST_DATASET="1000 @ ScanNetpp_Seq(filter=True, sample_freq=3, num_views=13, split='test', resolution=224, seed=666)+ \
1000 @ Aria_Seq(num_views=13, split='test', resolution=224, seed=666) + \
1000 @ Co3d_Seq(num_views=13, sel_num=3, degree=180, mask_bg='rand', split='test', resolution=224, seed=666)"

PRETRAINED="checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
TRAIN_OUT_DIR="checkpoints/slam3r_l2w"

torchrun --nproc_per_node=8 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21,norm_mode=None), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, norm_mode=None)" \
    --pretrained $PRETRAINED \
    --pretrained_type "dust3r" \
    --lr 5e-5 --min_lr 5e-7 --warmup_epochs 20 --epochs 200 --batch_size 4 --accum_iter 1 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 20\
    --save_config\
    --output_dir $TRAIN_OUT_DIR \
    --freeze "encoder"\
    --loss_func "l2w" \
    --ref_ids 0 1 2 3 4 5

