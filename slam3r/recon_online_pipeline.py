import warnings
warnings.filterwarnings("ignore")
import os
from os.path import join 
from tqdm import tqdm
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.ion()

from line_profiler import profile

from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Image2PointsModel, Local2WorldModel, inf
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import * 
from datasets_preprocess.get_webvideo import *
from slam3r.utils.image import process_single_frame

class picture_reader:
    """
    Read images from a directory, video file, or online video URL.
    Args:
        dataset (str): Path to the image directory, video file, or online video URL.
    Returns:
        type (str): Type of the input ('imgs', 'video', or 'https').
    
    read() (function): 
        Function to read the next image/frame.
    Returns (bool, image), where bool indicates success.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.type = ""
        self.readnum = 0
        if dataset.find(":") != -1:
            self.type = "https"
        else:
            if dataset[-3:] == "mp4":
                self.type = "video"
            else:
                self.type = "imgs"
        if self.type == "imgs":
            print('loading dataset: ', self.dataset)
            self.data = Seq_Data(img_dir=self.dataset, postfix='.png', \
                                    img_size=224, silent=False, sample_freq=1, \
                                    start_idx=0, num_views=-1, start_freq=1, to_tensor=True)
            if hasattr(self.data, "set_epoch"):
                self.data.set_epoch(0)
        elif self.type == "video":
            self.video_capture = cv2.VideoCapture(self.dataset)
            if not self.video_capture.isOpened():
                print(f"error!can not open the video file{self.dataset}")
                exit()
            print("successful opened! start processing frame by frame...")
        elif self.type == "https":
            self.get_api = Get_online_video(self.dataset)
    def read(self):
        if self.type == "https":
            return self.get_api.cap.read()
        elif self.type == "video":
            return self.video_capture.read()
        elif self.type == "imgs":
            print(f"reading the {self.readnum}th image")
            self.readnum += 1
            if self.readnum >= len(self.data[0]):
                return False, None
            return True, self.data[0][self.readnum]
def save_recon(views, pred_frame_num, save_dir, scene_id, save_all_views=False, 
                      imgs=None, registered_confs=None, 
                      num_points_save=200000, conf_thres_res=3, valid_masks=None):  
    save_name = f"{scene_id}_recon.ply"
    # collect the registered point clouds and rgb colors
    if imgs is None:
        imgs = [transform_img(unsqueeze_view(view))[:,::-1] for view in views]
    pcds = []
    rgbs = []
    for i in range(pred_frame_num):
        registered_pcd = to_numpy(views[i]['pts3d_world'][0])
        if registered_pcd.shape[0] == 3:
            registered_pcd = registered_pcd.transpose(1,2,0)
        registered_pcd = registered_pcd.reshape(-1,3)
        rgb = imgs[i].reshape(-1,3)
        pcds.append(registered_pcd)
        rgbs.append(rgb)
    if save_all_views:
        for i in range(pred_frame_num):
            save_ply(points=pcds[i], save_path=join(save_dir, f"frame_{i}.ply"), colors=rgbs[i])
    res_pcds = np.concatenate(pcds, axis=0)
    res_rgbs = np.concatenate(rgbs, axis=0)
    pts_count = len(res_pcds)
    valid_ids = np.arange(pts_count)
    # filter out points with gt valid masks
    if valid_masks is not None:
        valid_masks = np.stack(valid_masks, axis=0).reshape(-1)
        # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
    else:
        valid_masks = np.ones(pts_count, dtype=bool)
    # filter out points with low confidence
    if registered_confs is not None:
        conf_masks = []
        for i in range(len(registered_confs)):
            conf = registered_confs[i]
            conf_mask = (conf > conf_thres_res).reshape(-1).cpu() 
            conf_masks.append(conf_mask)
        conf_masks = np.array(torch.cat(conf_masks))
        valid_ids = valid_ids[conf_masks&valid_masks]
        print('ratio of points filered out: {:.2f}%'.format((1.-len(valid_ids)/pts_count)*100))
    # sample from the resulting pcd consisting of all frames
    n_samples = min(num_points_save, len(valid_ids))
    print(f"resampling {n_samples} points from {len(valid_ids)} points")
    sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
    sampled_pts = res_pcds[sampled_idx]
    sampled_rgbs = res_rgbs[sampled_idx]
    save_ply(points=sampled_pts[:,:3], save_path=join(save_dir, save_name), colors=sampled_rgbs)

def load_model(model_name, weights, device='cuda'):
    print('Loading model: {:s}'.format(model_name))
    model = eval(model_name)
    model.to(device)
    print('Loading pretrained: ', weights)
    ckpt = torch.load(weights, map_location=device)
    print(model.load_state_dict(ckpt['model'], strict=False))
    del ckpt  # in case it occupies memory
    return model

@torch.no_grad()
def get_img_tokens(views, model,silent=False):
    """get img tokens output from encoder,
    which can be reused by both i2p and l2w models
    """
    res_shapes, res_feats, res_poses = model._encode_multiview(views, 
                                                               view_batchsize=10, 
                                                               normalize=False,
                                                               silent=silent)
    return res_shapes, res_feats, res_poses

@torch.no_grad()
def get_single_img_tokens(views, model, silent=False):
    """get an img token output from encoder,
    which can be reused by both i2p and l2w models
    """
    
    # Check if the input is a list containing a single view dictionary
    assert isinstance(views, list) and len(views) == 1
    view = views[0]
    image = view['img']
    true_shape = view['true_shape']
    res_feat, pos, _ = model._encode_image(image, true_shape, normalize=False)
    res_shape = [view['true_shape']]
    res_feat = [res_feat]
    res_poses = [pos] 
    return res_shape, res_feat, res_poses

def scene_frame_retrieve(candi_views:list, src_views:list, i2p_model, 
                         sel_num=5, cand_registered_confs=None, 
                         depth=2, exclude_ids=None, culmu_count=None):
    """retrieve the scene frames from the candidate views
    For more detail, see 'Multi-keyframe co-registration' in our paper
    
    Args:
        candi_views: list of views to be selected from
        src_views: list of views that are searched for the best scene frames
        sel_num: how many scene frames to be selected
        cand_registered_confs: the registered confidences of the candidate views
        depth: the depth of decoder used for the correlation score calculation
        exclude_ids: the ids of candidate views that should be excluded from the selection

    Returns:
        selected_views: the selected scene frames
        sel_ids: the ids of selected scene frames in candi_views
    """
    num_candi_views = len(candi_views)
    if sel_num >= num_candi_views:
        return candi_views, list(range(num_candi_views))
    
    batch_inputs = []
    for bch in range(len(src_views)):
        input_views = []
        for view in [src_views[bch]]+candi_views:
            if 'img_tokens' in view:
                input_view = dict(img_tokens=view['img_tokens'], 
                                  true_shape=view['true_shape'],
                                  img_pos=view['img_pos'])
            else:
                input_view = dict(img=view['img'])
            input_views.append(input_view)
        batch_inputs.append(tuple(input_views))
    batch_inputs = collate_with_cat(batch_inputs) 
    with torch.no_grad():
        patch_corr_scores = i2p_model.get_corr_score(batch_inputs, ref_id=0, depth=depth)  #(R,S,P)

    sel_ids = sel_ids_by_score(patch_corr_scores, align_confs=cand_registered_confs, 
                          sel_num=sel_num, exclude_ids=exclude_ids, use_mask=False, 
                          culmu_count=culmu_count)
    selected_views = [candi_views[id] for id in sel_ids]
    return selected_views, sel_ids

def sel_ids_by_score(corr_scores: torch.tensor, align_confs, sel_num, 
                     exclude_ids=None, use_mask=True, culmu_count=None):
    """select the ids of views according to the confidence
    corr_scores (cand_num,src_num,patch_num): the correlation scores between 
                                              source views and all patches of candidate views 
    """
    # normalize the correlation scores to [0,1], to avoid overconfidence
    corr_scores = corr_scores.mean(dim=[1,2])  #(V,)
    corr_scores = (corr_scores - 1)/corr_scores
    # below are three designs for better retrieval,
    # but we do not use them in this version
    if align_confs is not None:
        cand_confs = (torch.stack(align_confs,dim=0)).mean(dim=[1,2]).to(corr_scores.device)
        cand_confs = (cand_confs - 1)/cand_confs
        confs = corr_scores*cand_confs
    else:
        confs = corr_scores
    if culmu_count is not None:
        culmu_count = torch.tensor(culmu_count).to(corr_scores.device)
        max_culmu_count = culmu_count.max()
        culmu_factor = 1-0.05*(culmu_count/max_culmu_count)
        confs = confs*culmu_factor
    sel_ids = torch.argsort(confs, descending=True)[:sel_num]
    return sel_ids

def init_frame_process_method(picture_capture, data_views, rgb_imgs, frame_num, num_views, i, frame, args):
    """load frame into the method

    Args:
        picture_capture: the picture_reader object
        data_views: list of processed views
        rgb_imgs: list of processed rgb images for visualization
        frame_num: number of frames read
        num_views: number of processed views
        i: index of the current view in data_views
        frame: the current frame read from picture_capture
        args: command line arguments
    """
    frame_num += 1
    num_views += 1
    i += 1
    # Pre-save the RGB images along with their corresponding masks
    # in preparation for visualization at last.
    if picture_capture.type != "imgs":
        frame = process_single_frame(frame,224,"cuda")
    else:
        frame['true_shape'] = frame['true_shape'][0]
    data_views.append(frame)
    if data_views[i]['img'].shape[0] == 1:
        data_views[i]['img'] = data_views[i]['img'][0]
    rgb_imgs.append(transform_img(dict(img=data_views[i]['img'][None]))[...,::-1])
    # process now image for extracting its img token with encoder
    data_views[i]['img'] = torch.tensor(data_views[i]['img'][None])
    data_views[i]['true_shape'] = torch.tensor(data_views[i]['true_shape'][None])
    for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
        if key in data_views[i]:
            del data_views[key]
    to_device(data_views[i], device=args.device)
    return frame_num, num_views, i, frame
    
def encode_single_frame(res_shapes, res_feats, res_poses, input_views, per_frame_res, 
                        registered_confs_mean, data_views, i, i2p_model):
    temp_shape, temp_feat, temp_pose = get_single_img_tokens([data_views[i]], i2p_model, True)
    res_shapes.append(temp_shape[0])
    res_feats.append(temp_feat[0])
    res_poses.append(temp_pose[0])
    print(f"finish pre-extracting img token of view {i}")

    input_views.append(dict(label=data_views[i]['label'],
                            img_tokens=temp_feat[0],
                            true_shape=data_views[i]['true_shape'],
                            img_pos=temp_pose[0]))
    for key in per_frame_res:
        per_frame_res[key].append(None)
    registered_confs_mean.append(i)
    

def register_initial_window_frames(init_num, kf_stride, buffering_set_ids, input_views, 
                                   l2w_model, args, per_frame_res, registered_confs_mean):
    """
    initially register the frames within the initial window with L2W model
    """
    max_conf_mean = -1
    for view_id in tqdm(range((init_num - 1) * kf_stride), desc="pre-registering"):
        if view_id % kf_stride == 0:
            continue
        # construct the input for L2W model

        l2w_input_views = [input_views[view_id]] + [input_views[id] for id in buffering_set_ids]
        # (for defination of ref_ids, seee the doc of l2w_model)
        output = l2w_inference(l2w_input_views, l2w_model,
                                ref_ids=list(range(1,len(l2w_input_views))),
                                device=args.device,
                                normalize=args.norm_input)
        # process the output of L2W model
        input_views[view_id]['pts3d_world'] = output[0]['pts3d_in_other_view'] # 1,224,224,3
        conf_map = output[0]['conf'] # 1,224,224
        per_frame_res['l2w_confs'][view_id] = conf_map[0] # 224,224
        registered_confs_mean[view_id] = conf_map.mean().cpu()
        per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
        
        if registered_confs_mean[view_id] > max_conf_mean:
            max_conf_mean = registered_confs_mean[view_id]
    print(f'finish aligning {(init_num)*kf_stride} head frames, with a max mean confidence of {max_conf_mean:.2f}')
    return max_conf_mean

def initialize_scene(views:list, model:Image2PointsModel, winsize=5, conf_thres=5, return_ref_id=False):    
    """initialize the scene with the first several frames.
    Try to find the best window size and the best ref_id.
    """
    init_ref_id = 0
    max_med_conf = 0     
    window_views = views[:winsize]
    # traverse all views in the window to find the best ref_id
    for i in range(winsize):
        ref_id = i
        output = i2p_inference_batch([window_views], model, ref_id=ref_id, 
                                    tocpu=True, unsqueeze=False)
        preds = output['preds']
        # choose the ref_id with the highest median confidence
        med_conf = np.array([preds[j]['conf'].mean() for j in range(winsize)]).mean()
        if med_conf > max_med_conf:
            max_med_conf = med_conf
            init_ref_id = ref_id
    output = i2p_inference_batch([views[:winsize]], model, ref_id=init_ref_id, 
                                    tocpu=False, unsqueeze=False)
    initial_pcds = []
    initial_confs = []
    for j in range(winsize):
        if j == init_ref_id:
            initial_pcds.append(output['preds'][j]['pts3d'])
        else:
            initial_pcds.append(output['preds'][j]['pts3d_in_other_view'])
        initial_confs.append(output['preds'][j]['conf'])
    print(f'initialize scene with {winsize} views, with a mean confidence of {max_med_conf:.2f}')
    if return_ref_id:
        return initial_pcds, initial_confs, init_ref_id
    return initial_pcds, initial_confs

def select_ids_as_reference(buffering_set_ids, i, next_register_id, num_views, input_views, i2p_model, num_scene_frame, win_r
                                     , adj_distance):
    """select the ids of scene frames from the buffering set
    """

    ni = next_register_id
    max_id = min(ni, num_views - 1)
    # select sccene frames in the buffering set to work as a global reference
    cand_ref_ids = buffering_set_ids
    ref_views, sel_pool_ids = scene_frame_retrieve(
        [input_views[i] for i in cand_ref_ids],
        input_views[ni:ni + 1],
        i2p_model, sel_num=num_scene_frame,
        depth = 2)

    # construct the local window with the initial views
    sel_ids = [i]
    for j in range(1, win_r + 1):
        if i - j * adj_distance >= 0:
            sel_ids.append(i - j * adj_distance)
    
    real_sel_pool_ids = []
    numi = 0
    for item in sel_pool_ids:
        real_sel_pool_ids.append(buffering_set_ids[item])
        numi += 1
        if numi >= win_r:
            break
    sel_ids += real_sel_pool_ids
    return sel_ids, ref_views, ni, max_id

def initial_scene_for_accumulated_frames(input_views, initial_winsize, kf_stride, i2p_model, per_frame_res, registered_confs_mean, args, conf_thres_i2p):
    """initialize the scene with the first several frames.
    Set up the world coordinates with the initial window.
    """
    initial_pcds, initial_confs, init_ref_id = initialize_scene(input_views[:initial_winsize*kf_stride:kf_stride],
                                                                i2p_model,winsize=initial_winsize,return_ref_id=True)
    # set up the world coordinates with the initial window
    init_num = len(initial_pcds)
    for j in range(init_num):
        per_frame_res['l2w_confs'][j * kf_stride] = initial_confs[j][0].to(args.device)
        registered_confs_mean[j * kf_stride] = per_frame_res['l2w_confs'][j * kf_stride].mean().cpu()
    # initialize the buffering set with the initial window
    assert args.buffer_size <= 0 or args.buffer_size >= init_num 
    buffering_set_ids = [j*kf_stride for j in range(init_num)]
    # set ip the woeld coordinates with frames in the initial window
    for j in range(init_num):
        input_views[j*kf_stride]['pts3d_world'] = initial_pcds[j]
    initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs]
    normed_pts = normalize_views([view['pts3d_world'] for view in input_views[:init_num*kf_stride:kf_stride]],
                                                initial_valid_masks)
    for j in range(init_num):
        input_views[j*kf_stride]['pts3d_world'] = normed_pts[j]
        # filter out points with low confidence
        input_views[j*kf_stride]['pts3d_world'][~initial_valid_masks[j]] = 0
        per_frame_res['l2w_pcds'][j*kf_stride] = normed_pts[j]
    return buffering_set_ids, init_ref_id, init_num

def recover_points_in_initial_window(i, buffering_set_ids, kf_stride, init_ref_id, per_frame_res, win_r
                                     , adj_distance, input_views, i2p_model, conf_thres_i2p):
    """
    recover the points in their local coordinates for all frames up to now
    """
    for view_id in range(i + 1):
        # skip the views in the initial window
        if view_id in buffering_set_ids:
            # trick to mark the keyframe in the initial window
            if view_id // kf_stride == init_ref_id:
                per_frame_res['i2p_pcds'][view_id] = per_frame_res['l2w_pcds'][view_id].cpu()
            else:
                per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(per_frame_res['l2w_pcds'][view_id], device="cpu")
            per_frame_res['i2p_confs'][view_id] = per_frame_res['l2w_confs'][view_id].cpu()
            print(f"finish revocer pcd of frame {view_id} in their local coordinates(in buffer set), with a mean confidence of {per_frame_res['i2p_confs'][view_id].mean():.2f} up to now.")
            continue
        # construct the local window with the initial views
        sel_ids = [view_id]
        for j in range(1, win_r + 1):
            if view_id - j * adj_distance >= 0:
                sel_ids.append(view_id - j * adj_distance)
            if view_id + j * adj_distance < i:
                sel_ids.append(view_id + j * adj_distance)
        local_views = [input_views[id] for id in sel_ids]
        ref_id = 0

        # recover poionts in the initial window, and save the keyframe points and confs
        output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id,
                                        tocpu=False, unsqueeze=False)['preds']
        # save results of the i2p model for the initial window
        per_frame_res['i2p_pcds'][view_id] = output[ref_id]['pts3d'].cpu()
        per_frame_res['i2p_confs'][view_id] = output[ref_id]['conf'][0].cpu()

        # construct the input for L2W model
        input_views[view_id]['pts3d_cam'] = output[ref_id]['pts3d']
        valid_mask = output[ref_id]['conf'] > conf_thres_i2p
        input_views[view_id]['pts3d_cam'] = normalize_views([input_views[view_id]['pts3d_cam']],
                                                                [valid_mask])[0]
        input_views[view_id]['pts3d_cam'][~valid_mask] = 0

        local_confs_mean_up2now = [conf.mean() for conf in per_frame_res['i2p_confs'] if conf is not None]
        print(f"finish revocer pcd of frame {view_id} in their local coordinates, with a mean confidence of {torch.stack(local_confs_mean_up2now).mean():.2f} up to now.")
    return local_confs_mean_up2now

def recover_points_in_online_views(local_views, i2p_model, i, ref_id, 
                                   per_frame_res, input_views, conf_thres_i2p):
    output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id,
                                    tocpu=False, unsqueeze=False)['preds']
    # save results of the i2p model for the initial window
    per_frame_res['i2p_pcds'][i] = output[ref_id]['pts3d'].cpu()
    per_frame_res['i2p_confs'][i] = output[ref_id]['conf'][0].cpu()

    # construct the input for L2W model
    input_views[i]['pts3d_cam'] = output[ref_id]['pts3d']
    valid_mask = output[ref_id]['conf'] > conf_thres_i2p
    input_views[i]['pts3d_cam'] = normalize_views([input_views[i]['pts3d_cam']],
                                                            [valid_mask])[0]
    input_views[i]['pts3d_cam'][~valid_mask] = 0

    local_confs_mean_up2now = [conf.mean() for conf in per_frame_res['i2p_confs']]
    print(f"finish revocer pcd of frame {i} in their local coordinates, with a mean confidence of {torch.stack(local_confs_mean_up2now).mean():.2f} up to now.")
    return local_confs_mean_up2now

def register_online_view(ref_views, input_views, l2w_model,
                         args, max_id, ni, per_frame_res, registered_confs_mean, i, next_register_id):
    l2w_input_views = ref_views + [input_views[i]]
    
    output = l2w_inference(l2w_input_views, l2w_model,
                            ref_ids=list(range(len(ref_views))),
                            device=args.device,
                            normalize=args.norm_input)
    
    # process the output of L2W model
    src_ids_local = [id + len(ref_views) for id in range(max_id - ni + 1)]
    src_ids_global = [id for id in range(ni, max_id + 1)]

    output_id = src_ids_local[0] # the id of the output in the output list
    view_id = src_ids_global[0]    # the id of the view in all views
    conf_map = output[output_id]['conf'] # 1,224,224
    input_views[view_id]['pts3d_world'] = output[output_id]['pts3d_in_other_view'] # 1,224,224,3
    per_frame_res['l2w_confs'][view_id] = conf_map[0]
    registered_confs_mean[view_id] = conf_map[0].mean().cpu()
    per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
    next_register_id += 1
    return next_register_id
def update_buffer_set(next_register_id, update_buffer_intv, max_buffer_size, kf_stride, buffering_set_ids, 
                      strategy, registered_confs_mean, local_confs_mean_up2now, candi_frame_id, milestone):
    if next_register_id - milestone >= update_buffer_intv:
        while(next_register_id - milestone >= kf_stride):
            candi_frame_id += 1
            full_flag = max_buffer_size > 0 and len(buffering_set_ids) >= max_buffer_size
            insert_flag = (not full_flag) or ((strategy == 'fifo') or 
                                            (strategy == 'reservoir' and np.random.rand() < max_buffer_size/candi_frame_id))
            if not insert_flag: 
                milestone += kf_stride
                continue
            # Use offest to ensure the selected view is not too close to the last selected view
            # If the last selected view is 0, 
            # the next selected view should be at least kf_stride*3//4 frames away
            start_ids_offset = max(0, buffering_set_ids[-1]+kf_stride*3//4 - milestone)
                
            # get the mean confidence of the candidate views
            mean_cand_recon_confs = torch.stack([registered_confs_mean[i]
                                    for i in range(milestone+start_ids_offset, milestone+kf_stride)])
            mean_cand_local_confs = torch.stack([local_confs_mean_up2now[i]
                                    for i in range(milestone+start_ids_offset, milestone+kf_stride)])
            # normalize the confidence to [0,1], to avoid overconfidence
            mean_cand_recon_confs = (mean_cand_recon_confs - 1)/mean_cand_recon_confs # transform to sigmoid
            mean_cand_local_confs = (mean_cand_local_confs - 1)/mean_cand_local_confs
            # the final confidence is the product of the two kinds of confidences
            mean_cand_confs = mean_cand_recon_confs*mean_cand_local_confs
            
            most_conf_id = mean_cand_confs.argmax().item()
            most_conf_id += start_ids_offset
            id_to_buffer = milestone + most_conf_id
            buffering_set_ids.append(id_to_buffer)
            # print(f"add ref view {id_to_buffer}")                
            # since we have inserted a new frame, overflow must happen when full_flag is True
            if full_flag:
                if strategy == 'reservoir':
                    buffering_set_ids.pop(np.random.randint(max_buffer_size))
                elif strategy == 'fifo':
                    buffering_set_ids.pop(0)
            # print(next_register_id, buffering_set_ids)
            milestone += kf_stride
            
    return milestone, candi_frame_id

def scene_recon_pipeline_online(i2p_model:Image2PointsModel,
                         l2w_model:Local2WorldModel,
                         picture_capture:picture_reader,
                         args,
                         save_dir = "results"):
    win_r = args.win_r
    num_scene_frame = args.num_scene_frame
    initial_winsize = args.initial_winsize
    conf_thres_l2w = args.conf_thres_l2w
    conf_thres_i2p = args.conf_thres_i2p
    num_points_save = args.num_points_save
    kf_stride = args.keyframe_stride
    
    scene_id = "demo_output"
    data_views = []
    num_views = 0
    rgb_imgs = []
    input_views = []
    res_shapes = []
    res_feats = []
    res_poses = []
    
    local_confs_mean_up2now = []
    adj_distance = kf_stride
    fail_view = {}
    
    assert initial_winsize >= 2, "not enough views for initializing the scene reconstruction"
    per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
    registered_confs_mean = []
    i = -1
    frame_num = 0
    success, frame = picture_capture.read()
    if not success:
        return

    
    while frame is not None:
        if frame_num % args.perframe == 0:
            frame_num, num_views, i, frame = init_frame_process_method(picture_capture, data_views, rgb_imgs,
                                      frame_num, num_views, i, frame, args)
            encode_single_frame(res_shapes, res_feats, res_poses, input_views
                                ,per_frame_res, registered_confs_mean, data_views, i
                                ,i2p_model)
            
            # accumulate enough frames for scene initialization
            if i < (initial_winsize - 1) * kf_stride:
                success, frame = picture_capture.read()
                # continue to read next frame
                if not success:
                    return
                continue
            # it's time to initialize the scene
            elif i == (initial_winsize - 1) * kf_stride:
                buffering_set_ids, init_ref_id, init_num = initial_scene_for_accumulated_frames(input_views, initial_winsize, kf_stride,
                                                     i2p_model, per_frame_res, registered_confs_mean,
                                                     args, conf_thres_i2p)
                local_confs_mean_up2now = recover_points_in_initial_window(i, buffering_set_ids, kf_stride,
                                                 init_ref_id, per_frame_res, win_r,
                                                 adj_distance, input_views, i2p_model,
                                                 conf_thres_i2p)
                
                # Special treatment: register the frames within the range of initial window with L2W model
                if kf_stride > 1:
                    max_conf_mean = register_initial_window_frames(init_num, kf_stride, buffering_set_ids,
                                                   input_views, l2w_model, args, per_frame_res, registered_confs_mean)
                    # A problem is that the registered_confs_mean of the initial window is generated by I2P model,
                    # while the registered_confs_mean of the frames within the initial window is generated by L2W model,
                    # so there exists a gap. Here we try to align it.
                    max_initial_conf_mean = -1
                    for ii in range(init_num):
                        if registered_confs_mean[ii*kf_stride] > max_initial_conf_mean:
                            max_initial_conf_mean = registered_confs_mean[ii*kf_stride]
                    factor = max_conf_mean/max_initial_conf_mean
                    # print(f'align register confidence with a factor {factor}')
                    for ii in range(init_num):
                        per_frame_res['l2w_confs'][ii*kf_stride] *= factor
                        registered_confs_mean[ii*kf_stride] = per_frame_res['l2w_confs'][ii*kf_stride].mean().cpu()
                
                # register the rest frames with L2W model
                next_register_id = (init_num - 1) * kf_stride + 1
                milestone = init_num * kf_stride + 1
                update_buffer_intv = kf_stride*args.update_buffer_intv   # update the buffering set every update_buffer_intv frames
                max_buffer_size = args.buffer_size
                strategy = args.buffer_strategy
                candi_frame_id = len(buffering_set_ids) # used for the reservoir sampling strategy
                success, frame = picture_capture.read()
                if not success:
                    break
                continue
            
            # start recovering the online views
            # skip the views in the initial window
            if i in buffering_set_ids:
                # trick to mark the keyframe in the initial window
                if i // kf_stride == init_ref_id:
                    per_frame_res['i2p_pcds'][i] = per_frame_res['l2w_pcds'][i].cpu()
                else:
                    per_frame_res['i2p_pcds'][i] = torch.zeros_like(per_frame_res['l2w_pcds'][i], device="cpu")
                per_frame_res['i2p_confs'][i] = per_frame_res['l2w_confs'][i].cpu()
                continue
                
            sel_ids, ref_views, ni, max_id = select_ids_as_reference(buffering_set_ids, i, next_register_id, 
                                                num_views, input_views, i2p_model, num_scene_frame, win_r,
                                                adj_distance)
            local_views = [input_views[id] for id in sel_ids]
            ref_id = 0
            
            local_confs_mean_up2now = recover_points_in_online_views(local_views, i2p_model, i, ref_id, per_frame_res,
                                           input_views, conf_thres_i2p)
            next_register_id = register_online_view(ref_views, input_views, l2w_model, args,
                                 max_id, ni, per_frame_res, registered_confs_mean, i, next_register_id)
            milestone, candi_frame_id = update_buffer_set(next_register_id, update_buffer_intv, max_buffer_size, kf_stride,
                              buffering_set_ids, strategy, registered_confs_mean, local_confs_mean_up2now, candi_frame_id, milestone)
            
            conf = registered_confs_mean[i]
            if conf < 10:
                fail_view[i] = conf.item()
            print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confs_mean).mean().item():.2f}')
            print(f"{len(fail_view)} views with low confidence: ", {key:round(fail_view[key],2) for key in fail_view.keys()})

            if args.save_online:
                # TODO:
                # save the reconstruction results in real time using viser
                pass
            
            frame_num += 1
            success, frame = picture_capture.read()
            
            if not success:
                break
        else:
            success, frame = picture_capture.read()
            frame_num += 1
            if not success:
                break
    save_recon(input_views, num_views, save_dir, scene_id, 
                      args.save_all_views, rgb_imgs, registered_confs=per_frame_res['l2w_confs'], 
                      num_points_save=num_points_save, 
                      conf_thres_res=conf_thres_l2w)
    print(frame_num)
    if args.save_preds:
        preds_dir = join(save_dir, 'preds')
        os.makedirs(preds_dir, exist_ok=True)
        print(f">> saving per-frame predictions to {preds_dir}") 
        np.save(join(preds_dir, 'local_pcds.npy'), torch.cat(per_frame_res['i2p_pcds']).cpu().numpy())
        np.save(join(preds_dir, 'registered_pcds.npy'), torch.cat(per_frame_res['l2w_pcds']).cpu().numpy())
        np.save(join(preds_dir, 'local_confs.npy'), torch.stack([conf.cpu() for conf in per_frame_res['i2p_confs']]).numpy())
        np.save(join(preds_dir, 'registered_confs.npy'), torch.stack([conf.cpu() for conf in per_frame_res['l2w_confs']]).numpy())
        np.save(join(preds_dir, 'input_imgs.npy'), np.stack(rgb_imgs))
        
        metadata = dict(scene_id=scene_id,
                        init_winsize=init_num,
                        kf_stride=kf_stride,
                        init_ref_id=init_ref_id)
        with open(join(preds_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    elif args.save_for_eval:
        preds_dir = join(save_dir, 'preds')
        os.makedirs(preds_dir, exist_ok=True)
        print(f">> saving per-frame predictions to {preds_dir}")
        np.save(join(preds_dir, 'registered_pcds.npy'), torch.cat(per_frame_res['l2w_pcds']).cpu().numpy())
        np.save(join(preds_dir, 'registered_confs.npy'), torch.stack([conf.cpu() for conf in per_frame_res['l2w_confs']]).numpy())
