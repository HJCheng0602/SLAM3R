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
import open3d as o3d
plt.ion()

from line_profiler import profile

from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Image2PointsModel, Local2WorldModel, inf
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import * 
from datasets_preprocess.get_webvideo import *
from slam3r.utils.image import process_single_frame

parser = argparse.ArgumentParser(description="Inference on a wild captured scene")
parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
parser.add_argument('--i2p_model', type=str, default="Image2PointsModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11)")
parser.add_argument("--l2w_model", type=str, default="Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11, need_encoder=False)")
parser.add_argument('--i2p_weights', type=str, help='path to the weights of i2p model')
parser.add_argument("--l2w_weights", type=str, help="path to the weights of l2w model")
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--dataset", type=str, help="a string indicating the dataset")
input_group.add_argument("--img_dir", type=str, help="directory of the input images")
parser.add_argument("--save_dir", type=str, default="results", help="directory to save the results") 
parser.add_argument("--test_name", type=str, required=True, help="name of the test")
parser.add_argument('--save_all_views', action='store_true', help='whether to save all views respectively')


# args for the whole scene reconstruction
parser.add_argument("--keyframe_stride", type=int, default=3, 
                    help="the stride of sampling keyframes, -1 for auto adaptation")
parser.add_argument("--initial_winsize", type=int, default=5, 
                    help="the number of initial frames to be used for scene initialization")
parser.add_argument("--win_r", type=int, default=3, 
                    help="the radius of the input window for I2P model")
parser.add_argument("--conf_thres_i2p", type=float, default=1.5, 
                    help="confidence threshold for the i2p model")
parser.add_argument("--num_scene_frame", type=int, default=10, 
                    help="the number of scene frames to be selected from \
                        buffering set when registering new keyframes")
parser.add_argument("--max_num_register", type=int, default=10, 
                    help="maximal number of frames to be registered in one go")
parser.add_argument("--conf_thres_l2w", type=float, default=12, 
                    help="confidence threshold for the l2w model(when saving final results)")
parser.add_argument("--num_points_save", type=int, default=2000000, 
                    help="number of points to be saved in the final reconstruction")
parser.add_argument("--norm_input", action="store_true", 
                    help="whether to normalize the input pointmaps for l2w model")
parser.add_argument("--save_frequency", type=int,default=3,
                    help="per xxx frame to save")
parser.add_argument("--save_each_frame",action='store_true',default=True,
                    help="whether to save each frame to .ply")
parser.add_argument("--video_path",type = str)

parser.add_argument("--update_buffer_intv", type=int, default=1, 
                    help="the interval of updating the buffering set")
parser.add_argument('--buffer_size', type=int, default=100, 
                    help='maximal size of the buffering set, -1 if infinite')
parser.add_argument("--buffer_strategy", type=str, choices=['reservoir', 'fifo'], default='reservoir', 
                    help='strategy for maintaining the buffering set: reservoir-sampling or first-in-first-out')
parser.add_argument("--save_online", action='store_true', 
                    help="whether to save the construct result online.")

#params for auto adaptation of keyframe frequency
parser.add_argument("--keyframe_adapt_min", type=int, default=1, 
                    help="minimal stride of sampling keyframes when auto adaptation")
parser.add_argument("--keyframe_adapt_max", type=int, default=20, 
                    help="maximal stride of sampling keyframes when auto adaptation")
parser.add_argument("--keyframe_adapt_stride", type=int, default=1, 
                    help="stride for trying different keyframe stride")
parser.add_argument("--perframe",type=int,default=1)
parser.add_argument("--enable_viewer",type=bool,default=False, help="whether to visualize the reconstruct progress.")

parser.add_argument("--seed", type=int, default=42, help="seed for python random")
parser.add_argument('--gpu_id', type=int, default=-1, help='gpu id, -1 for auto select')
parser.add_argument('--save_preds', action='store_true', help='whether to save all per-frame preds')    
parser.add_argument('--save_for_eval', action='store_true', help='whether to save partial per-frame preds for evaluation')   

class picture_reader:
    def __init__(self, dataset):
        # detect the type of the input data
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
        
        
            
            
    
        

class IncrementalReconstructor:
    def __init__(self,initial_capacity=10000, point_dim=3, growth_factor=1.5):
        self.res_rgbs = []
        self.res_confs = []
        self.valid_masks = []
        self.is_initialized = False
        self.pcds = []
        self.rgbs = []
        self.conf_masks = []
        self.frame_num = 0
        self.point_dim = point_dim
        
        self.growth_factor = growth_factor
        
        self._buffer1 = np.zeros((initial_capacity, self.point_dim), dtype=np.float32)
        self._capacity1 = initial_capacity
        self._size1 = 0
        
        # res_pcds 是外部访问的属性，它始终是正确大小的视图
        self.res_pcds = self._buffer1[:0]
        
        self._buffer2 = np.zeros((initial_capacity, 3), dtype=np.uint8)
        self._capacity2 = initial_capacity
        self._size2 = 0
        
        self.res_rgbs = self._buffer2[:0]
        

        self.vis = None
        self.pcd = None
    def _grow_capacity1(self, min_required_size: int):
        """
        内部方法，用于扩展缓冲区容量。
        """
        # 计算新容量，至少是所需大小，但通常会按增长因子变得更大
        new_capacity = int(max(min_required_size, self._capacity1 * self.growth_factor))
        print(f"New capacity set to: {new_capacity}")

        # 创建一个新的、更大的缓冲区
        new_buffer = np.zeros((new_capacity, 3), dtype=np.float32)
        
        # 将旧缓冲区的数据整体拷贝到新缓冲区
        # 这是主要的开销所在，但我们不常执行它
        new_buffer[:self._size1] = self._buffer1[:self._size1]
        
        # 替换旧缓冲区并更新容量
        self._buffer1 = new_buffer
        self._capacity1 = new_capacity
    def _grow_capacity2(self, min_required_size: int):
        """
        内部方法，用于扩展缓冲区容量。
        """
        # 计算新容量，至少是所需大小，但通常会按增长因子变得更大
        new_capacity = int(max(min_required_size, self._capacity2 * self.growth_factor))
        print(f"New capacity set to: {new_capacity}")

        # 创建一个新的、更大的缓冲区
        new_buffer = np.zeros((new_capacity, 3), dtype=np.uint8)
        
        # 将旧缓冲区的数据整体拷贝到新缓冲区
        # 这是主要的开销所在，但我们不常执行它
        new_buffer[:self._size2] = self._buffer2[:self._size2]
        
        # 替换旧缓冲区并更新容量
        self._buffer2 = new_buffer
        self._capacity2 = new_capacity

    def add_view(self, view, img, conf_thres=3, valid_mask=None, registed_conf=None):
        """add a new view to the reconstruction
        Args:
            view: a dict containing the following keys:
                'pts3d_world': (1,H,W,3) the pointmap in world coordinates
                'img': (1,3,H,W) the input image
                'conf': (1,H,W) the confidence map of the pointmap
            conf_thres: threshold for filtering out low-confidence points
        """
        
        
        
        registered_pcd = to_numpy(view['pts3d_world'][0])
        num_new_points = registered_pcd.shape[0] * registered_pcd.shape[1]

        if self._size1 + num_new_points > self._capacity1:
            print(f"Capacity insufficient for 1. Growing from {self._capacity1} to accommodate {num_new_points} new points.")
            self._grow_capacity1(self._size1 + num_new_points)
        if self._size2 + num_new_points > self._capacity2:
            print(f"Capacity insufficient for 2. Growing from {self._capacity2} to accommodate {num_new_points} new points.")
            self._grow_capacity2(self._size2 + num_new_points)
            
        if registered_pcd.shape[0] == 3:
            registered_pcd = registered_pcd.transpose(1,2,0)
        registered_pcd = registered_pcd.reshape(-1,3)
        rgb = img.reshape(-1,3)

        
        self._buffer1[self._size1:self._size1 + num_new_points] = registered_pcd
        self._size1 += num_new_points
        self.res_pcds = self._buffer1[:self._size1]
        
        self._buffer2[self._size2:self._size2 + num_new_points] = rgb
        self._size2 += num_new_points
        self.res_rgbs = self._buffer2[:self._size2]
        
        if valid_mask is not None:
            self.valid_masks.append(valid_mask)
        if registed_conf is not None:
            self.res_confs.append(registed_conf)
        if registed_conf is not None:
            conf_mask = (registed_conf > conf_thres).reshape(-1).cpu() 
            self.conf_masks.append(conf_mask)
            
    def save_online_view(self, save_dir, scene_id, num_points_save=200000, conf_thres_res=3):
        
        pts_count = self._size1
        valid_ids = np.arange(pts_count)
        

        
        if self.valid_masks:
            valid_masks = np.stack(self.valid_masks, axis=0).reshape(-1)
            # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
        else:
            valid_masks = np.ones(pts_count, dtype=bool)
        
        # filter out points with low confidence
        
        if self.res_confs is not None and len(self.res_confs) > 0:
            conf_masks = np.array(torch.cat(self.conf_masks))
            valid_ids = valid_ids[conf_masks&valid_masks]
            print('ratio of points filered out: {:.2f}%'.format((1.-len(valid_ids)/pts_count)*100) + f"saving the {self.frame_num}th frame")
            self.frame_num += 1
        
        # sample from the resulting pcd consisting of all frames
        n_samples = min(num_points_save, len(valid_ids))
        print(f"resampling {n_samples} points from {len(valid_ids)} points")
        sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
        sampled_pts = np.array(self.res_pcds)[sampled_idx]
        sampled_rgbs = np.array(self.res_rgbs)[sampled_idx]
        save_name = f"{scene_id}_{self.frame_num}th_recon_online.ply"

        save_ply(points=sampled_pts[:,:3], save_path=join(save_dir, save_name), colors=sampled_rgbs)

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
    
    # if use_mask:
    #     low_conf_mask = (corr_scores<0.1) | (cand_confs<0.1)
    # else:
    #     low_conf_mask = torch.zeros_like(corr_scores, dtype=bool)
    # exlude_mask = torch.zeros_like(corr_scores, dtype=bool)
    # if exclude_ids is not None:
    #     exlude_mask[exclude_ids] = True
    # invalid_mask = low_conf_mask | exlude_mask
    # confs[invalid_mask] = 0
    
    sel_ids = torch.argsort(confs, descending=True)[:sel_num]

    return sel_ids

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

    # if the best ref_id lead to a bad confidence, decrease the window size and try again
    # if winsize > 3 and max_med_conf < conf_thres:
    #     return initialize_scene(views, model, winsize-1, 
    #                             conf_thres=conf_thres, return_ref_id=return_ref_id)

    # get the initial point clouds and confidences with the best ref_id
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
    
@profile
def scene_recon_pipeline(i2p_model:Image2PointsModel,
                         l2w_model:Local2WorldModel,
                        picture_capture,args,
                         save_dir="results"):
    win_r = args.win_r
    num_scene_frame = args.num_scene_frame
    initial_winsize = args.initial_winsize
    conf_thres_l2w = args.conf_thres_l2w
    conf_thres_i2p = args.conf_thres_i2p
    num_points_save = args.num_points_save
    kf_stride = args.keyframe_stride


    reconstructor = IncrementalReconstructor()

    scene_id = "video_output"
    data_views = []
    num_views = 0
    rgb_imgs = []
    input_views = []
    res_shapes = []
    res_feats = []
    res_poses = []
    initialize_scene_list = []
    local_confs_mean_up2now = []
    adj_distance = kf_stride
    fail_view = {}
    

    assert initial_winsize >= 2, "not enough views for initializing the scene reconstruction"
    per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
    registered_confs_mean = []
    i = -1
    success, frame = picture_capture.read()
    

    
    frame_num = 0

    if not success:
        return

    while frame is not None:
        if frame_num % args.perframe == 0:

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

            # pre-extract img tokens by encoder, which can be reused 
            # in the following inference by both i2p and l2w models
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

            # accumulate the initial window frames
            if i < (initial_winsize - 1)*kf_stride and i % kf_stride == 0:
                success, frame = picture_capture.read()
                if not success:
                    # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                    return
                continue
            elif i == (initial_winsize - 1)*kf_stride:
                initial_pcds, initial_confs, init_ref_id = initialize_scene(input_views[:initial_winsize*kf_stride:kf_stride],i2p_model,winsize=initial_winsize,return_ref_id=True)
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

            elif i < (initial_winsize - 1) * kf_stride:
                success, frame = picture_capture.read()
                if not success:
                    # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                    return
                continue


            # recover the pointmap of each view in their local coordinates with the I2P model

            # first recover the accumulate views
            if i == (initial_winsize - 1) * kf_stride:
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

                # Special treatment: register the frames within the range of initial window with L2W model
                if kf_stride > 1:
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
                    # 如果 success 为 False，说明视频已经处理完毕，跳出循环
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
            for item in sel_pool_ids:
                real_sel_pool_ids.append(buffering_set_ids[item])
                sel_ids += real_sel_pool_ids
            
            
            local_views = [input_views[id] for id in sel_ids]
            ref_id = 0

            # recover poionts in the initial window, and save the keyframe points and confs
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
            
            

            

            # register the source frames in the local coordinates to the world coordinates with L2W model
            l2w_input_views = ref_views + [input_views[i]]
            input_view_num = len(ref_views) + 1
            
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

            # update the buffering set
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
            # transfer the data to cpu if it is not in the buffering set, to save gpu memory
            # for i in range(next_register_id):
            #     to_device(input_views[i], device=args.device if i in buffering_set_ids else 'cpu')
            conf = registered_confs_mean[i]
            if conf < 10:
                fail_view[i] = conf.item()
            print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confs_mean).mean().item():.2f}')
            print(f"{len(fail_view)} views with low confidence: ", {key:round(fail_view[key],2) for key in fail_view.keys()})

            if args.save_online:
                # TODO:
                # save the reconstruction results in real time
                reconstructor.add_view(input_views[i], rgb_imgs[i], conf_thres=conf_thres_l2w, 
                                       registed_conf=per_frame_res['l2w_confs'][i])
                reconstructor.save_online_view(save_dir, scene_id, num_points_save=num_points_save)
            frame_num += 1
            success, frame = picture_capture.read()

            if not success:
                # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                break
        else:
            frame_num += 1
            success, frame = picture_capture.read()

            if not success:
                # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                break

            
    save_recon(input_views, num_views, save_dir, scene_id, 
                      args.save_all_views, rgb_imgs, registered_confs=per_frame_res['l2w_confs'], 
                      num_points_save=num_points_save, 
                      conf_thres_res=conf_thres_l2w)
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

if __name__ == "__main__":

    args = parser.parse_args()
    if args.gpu_id == -1:
        args.gpu_id = get_free_gpu()
    
    print("using gpu: ", args.gpu_id)
    torch.cuda.set_device(f"cuda:{args.gpu_id}")
    # print(args)
    np.random.seed(args.seed)

    if args.i2p_weights is not None:
        i2p_model = load_model(args.i2p_model, args.i2p_weights, args.device)
    else:
        i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        i2p_model.to(args.device)
    if args.l2w_weights is not None:
        l2w_model = load_model(args.l2w_model, args.l2w_weights, args.device)
    else:
        l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
        l2w_model.to(args.device)
    i2p_model.eval()
    l2w_model.eval()
    
    
    picture_capture = picture_reader(args.dataset)

    save_dir = os.path.join(args.save_dir, args.test_name)
    os.makedirs(save_dir, exist_ok=True)

    scene_recon_pipeline(i2p_model, l2w_model,picture_capture, args, save_dir=save_dir)
            
    
    