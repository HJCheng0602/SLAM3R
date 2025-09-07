import argparse
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import subprocess
import threading
from queue import Queue,Empty
import viser

from recon_unified import get_img_tokens, initialize_scene, i2p_inference_batch, l2w_inference, normalize_views, scene_frame_retrieve
from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Local2WorldModel, Image2PointsModel
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import *
from datasets_preprocess.get_webvideo import *
from slam3r.utils.image import process_single_frame

point_cloud_queue = Queue()

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


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default="./tmp", help="value for tempfile.tempdir")

    return parser


def extract_frames(video_path: str, fps: float) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%03d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_path
    ]
    subprocess.run(command, check=True)
    return temp_dir

class picture_reader:
    def __init__(self, dataset):
        # detect the type of the input data
        self.dataset = dataset
        self.type = ""
        self.readnum = 0
        
        if isinstance(dataset, list):
            self.type = "imgs"
        elif dataset.startswith("http") or dataset.startswith("https"):
            self.type = "https"
        elif dataset.endswith(".mp4") or dataset.endswith(".avi") or dataset.endswith(".mov"):
            self.type = "video"
        if self.type == "imgs":
            print('loading dataset: ', self.dataset)
            self.data = Seq_Data(img_dir=self.dataset, \
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

def recon_scene(i2p_model:Image2PointsModel, 
                l2w_model:Local2WorldModel, 
                device, save_dir, fps, 
                files_type,
                video_url,
                img_dir_or_list, 
                keyframe_stride, win_r, initial_winsize, conf_thres_i2p,
                num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size,
                conf_thres_l2w, num_points_save):
    # print(f"device: {device},\n save_dir: {save_dir},\n fps: {fps},\n keyframe_stride: {keyframe_stride},\n win_r: {win_r},\n initial_winsize: {initial_winsize},\n conf_thres_i2p: {conf_thres_i2p},\n num_scene_frame: {num_scene_frame},\n update_buffer_intv: {update_buffer_intv},\n buffer_strategy: {buffer_strategy},\n buffer_size: {buffer_size},\n conf_thres_l2w: {conf_thres_l2w},\n num_points_save: {num_points_save}")
    np.random.seed(42)

    
    num_views = 0
    source = ""
    
    if files_type == "webcamera":
        source = video_url.value
    else:
        source = img_dir_or_list
    
    
    dataset = picture_reader(source)
    success, frame = dataset.read()  
    if not success:
        return
    input_views = []
    rgb_imgs = []
    res_shapes = []
    res_feats = []
    res_poses = []
    assert initial_winsize >= 2, "not enough views for initializing the scene reconstruction"
    per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[], rgb_imgs=[])
    registered_confs_mean = []
    fail_view = {}
    data_views = []
    frame_num = 0
    i = -1

    while frame is not None:
        if frame_num % fps == 0:    

            frame_num += 1
            num_views += 1
            i += 1
            # Pre-save the RGB images along with their corresponding masks
            # in preparation for visualization at last.
            if dataset.type != "imgs":
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
            to_device(data_views[i], device=device)

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
            if i < (initial_winsize - 1)*keyframe_stride and i % keyframe_stride == 0:
                success, frame = dataset.read()
                if not success:
                    # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                    return
                continue
            elif i == (initial_winsize - 1)*keyframe_stride:
                initial_pcds, initial_confs, init_ref_id = initialize_scene(input_views[:initial_winsize*keyframe_stride:keyframe_stride],i2p_model,winsize=initial_winsize,return_ref_id=True)
                # set up the world coordinates with the initial window
                init_num = len(initial_pcds)
                for j in range(init_num):
                    per_frame_res['l2w_confs'][j * keyframe_stride] = initial_confs[j][0].to(device)
                    registered_confs_mean[j * keyframe_stride] = per_frame_res['l2w_confs'][j * keyframe_stride].mean().cpu()
                # initialize the buffering set with the initial window
                assert buffer_size <= 0 or buffer_size >= init_num 
                buffering_set_ids = [j*keyframe_stride for j in range(init_num)]
                # set ip the woeld coordinates with frames in the initial window
                for j in range(init_num):
                    input_views[j*keyframe_stride]['pts3d_world'] = initial_pcds[j]
                initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs]
                normed_pts = normalize_views([view['pts3d_world'] for view in input_views[:init_num*keyframe_stride:keyframe_stride]],
                                                            initial_valid_masks)
                for j in range(init_num):
                    input_views[j*keyframe_stride]['pts3d_world'] = normed_pts[j]
                    # filter out points with low confidence
                    input_views[j*keyframe_stride]['pts3d_world'][~initial_valid_masks[j]] = 0
                    per_frame_res['l2w_pcds'][j*keyframe_stride] = normed_pts[j]

            elif i < (initial_winsize - 1) * keyframe_stride:
                success, frame = dataset.read()
                if not success:
                    # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                    return
                continue


            # recover the pointmap of each view in their local coordinates with the I2P model

            # first recover the accumulate views
            if i == (initial_winsize - 1) * keyframe_stride:
                for view_id in range(i + 1):
                    # skip the views in the initial window
                    if view_id in buffering_set_ids:
                        # trick to mark the keyframe in the initial window
                        if view_id // keyframe_stride == init_ref_id:
                            per_frame_res['i2p_pcds'][view_id] = per_frame_res['l2w_pcds'][view_id].cpu()
                        else:
                            per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(per_frame_res['l2w_pcds'][view_id], device="cpu")
                        per_frame_res['i2p_confs'][view_id] = per_frame_res['l2w_confs'][view_id].cpu()
                        print(f"finish revocer pcd of frame {view_id} in their local coordinates(in buffer set), with a mean confidence of {per_frame_res['i2p_confs'][view_id].mean():.2f} up to now.")
                        continue
                    # construct the local window with the initial views
                    sel_ids = [view_id]
                    for j in range(1, win_r + 1):
                        if view_id - j * keyframe_stride >= 0:
                            sel_ids.append(view_id - j * keyframe_stride)
                        if view_id + j * keyframe_stride < i:
                            sel_ids.append(view_id + j * keyframe_stride)
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
                if keyframe_stride > 1:
                    max_conf_mean = -1
                    for view_id in tqdm(range((init_num - 1) * keyframe_stride), desc="pre-registering"):
                        if view_id % keyframe_stride == 0:
                            continue
                        # construct the input for L2W model

                        l2w_input_views = [input_views[view_id]] + [input_views[id] for id in buffering_set_ids]
                        # (for defination of ref_ids, seee the doc of l2w_model)
                        output = l2w_inference(l2w_input_views, l2w_model,
                                                ref_ids=list(range(1,len(l2w_input_views))),
                                                device=device,
                                                )
                        # process the output of L2W model
                        input_views[view_id]['pts3d_world'] = output[0]['pts3d_in_other_view'] # 1,224,224,3
                        conf_map = output[0]['conf'] # 1,224,224
                        per_frame_res['l2w_confs'][view_id] = conf_map[0] # 224,224
                        registered_confs_mean[view_id] = conf_map.mean().cpu()
                        per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
                        
                        if registered_confs_mean[view_id] > max_conf_mean:
                            max_conf_mean = registered_confs_mean[view_id]
                    print(f'finish aligning {(init_num)*keyframe_stride} head frames, with a max mean confidence of {max_conf_mean:.2f}')
                    # A problem is that the registered_confs_mean of the initial window is generated by I2P model,
                    # while the registered_confs_mean of the frames within the initial window is generated by L2W model,
                    # so there exists a gap. Here we try to align it.
                    max_initial_conf_mean = -1
                    for ii in range(init_num):
                        if registered_confs_mean[ii*keyframe_stride] > max_initial_conf_mean:
                            max_initial_conf_mean = registered_confs_mean[ii*keyframe_stride]
                    factor = max_conf_mean/max_initial_conf_mean
                    # print(f'align register confidence with a factor {factor}')
                    for ii in range(init_num):
                        per_frame_res['l2w_confs'][ii*keyframe_stride] *= factor
                        registered_confs_mean[ii*keyframe_stride] = per_frame_res['l2w_confs'][ii*keyframe_stride].mean().cpu()
                        
                # register the rest frames with L2W model
                next_register_id = (init_num - 1) * keyframe_stride + 1
                milestone = init_num * keyframe_stride + 1
                update_buffer_intv = keyframe_stride*update_buffer_intv   # update the buffering set every update_buffer_intv frames
                max_buffer_size = buffer_size
                strategy = buffer_strategy
                candi_frame_id = len(buffering_set_ids) # used for the reservoir sampling strategy
                success, frame = dataset.read()
                point_cloud_queue.put((per_frame_res["l2w_pcds"][i][0], rgb_imgs[i], per_frame_res['l2w_confs'][i]))

                
                if not success:
                    # 如果 success 为 False，说明视频已经处理完毕，跳出循环
                    break
                continue

            # start recovering the online views
            # skip the views in the initial window
            if i in buffering_set_ids:
                # trick to mark the keyframe in the initial window
                if i // keyframe_stride == init_ref_id:
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
                if i - j * keyframe_stride >= 0:
                    sel_ids.append(i - j * keyframe_stride)
            
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
                                    device=device,
                                    )
            
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
                while(next_register_id - milestone >= keyframe_stride):
                    candi_frame_id += 1
                    full_flag = max_buffer_size > 0 and len(buffering_set_ids) >= max_buffer_size
                    insert_flag = (not full_flag) or ((strategy == 'fifo') or 
                                                    (strategy == 'reservoir' and np.random.rand() < max_buffer_size/candi_frame_id))
                    if not insert_flag: 
                        milestone += keyframe_stride
                        continue
                    # Use offest to ensure the selected view is not too close to the last selected view
                    # If the last selected view is 0, 
                    # the next selected view should be at least kf_stride*3//4 frames away
                    start_ids_offset = max(0, buffering_set_ids[-1]+keyframe_stride*3//4 - milestone)
                        
                    # get the mean confidence of the candidate views
                    mean_cand_recon_confs = torch.stack([registered_confs_mean[i]
                                            for i in range(milestone+start_ids_offset, milestone+keyframe_stride)])
                    mean_cand_local_confs = torch.stack([local_confs_mean_up2now[i]
                                            for i in range(milestone+start_ids_offset, milestone+keyframe_stride)])
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
                    milestone += keyframe_stride
            # transfer the data to cpu if it is not in the buffering set, to save gpu memory
            # for i in range(next_register_id):
            #     to_device(input_views[i], device=args.device if i in buffering_set_ids else 'cpu')
            conf = registered_confs_mean[i]
            if conf < 10:
                fail_view[i] = conf.item()
            print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confs_mean).mean().item():.2f}')
            print(f"{len(fail_view)} views with low confidence: ", {key:round(fail_view[key],2) for key in fail_view.keys()})
            
            
            frame_num += 1
            success, frame = dataset.read()
            
            point_cloud_queue.put((per_frame_res["l2w_pcds"][i][0], rgb_imgs[i], per_frame_res['l2w_confs'][i]))

            if not success:
                break
        else:
            frame_num += 1
            success, frame = dataset.read()

            if not success:
                break
    per_frame_res['rgb_imgs'] = rgb_imgs
    save_path = get_model_from_scene(per_frame_res=per_frame_res, 
                                     save_dir=save_dir, 
                                     num_points_save=num_points_save, 
                                     conf_thres_res=conf_thres_l2w)
    point_cloud_queue.put(None)

    return save_path, per_frame_res

def print_model_viser():
    server = viser.ViserServer()


    points_buffer = np.zeros((0, 3), dtype=np.float32)
    colors_buffer = np.zeros((0, 3), dtype=np.uint8)

    point_cloud_handle = server.scene.add_point_cloud(
        name="/reconstruction_cloud",
        points=points_buffer,
        colors=colors_buffer,
        point_size=0.001
    )


    conf_thres_res = 12
    num_points_per_frame = 1000000 
    
    while True:
        try:
            new_data = point_cloud_queue.get(block=True, timeout=0.1)
            
            if new_data is None:
                print("consumer: received termination signal.")
                break

            new_frame_points_data, new_frame_colors_data, new_frame_confs_data = new_data
            
            # --- 数据处理部分不变 ---
            if isinstance(new_frame_points_data, torch.Tensor):
                new_frame_points = new_frame_points_data.cpu().numpy()
            else:
                new_frame_points = new_frame_points_data

            if isinstance(new_frame_colors_data, torch.Tensor):
                new_frame_colors = new_frame_colors_data.cpu().numpy().astype(np.uint8)
            else:
                new_frame_colors = new_frame_colors_data.astype(np.uint8)
            
            if isinstance(new_frame_confs_data, torch.Tensor):
                new_frame_confs = new_frame_confs_data.cpu().numpy()
            else:
                new_frame_confs = new_frame_confs_data

            flattened_points = new_frame_points.reshape(-1, 3)
            flattened_colors = new_frame_colors.reshape(-1, 3)
            flattened_confs = new_frame_confs.reshape(-1)
            
            conf_mask = flattened_confs > conf_thres_res
            filtered_points = flattened_points[conf_mask]
            filtered_colors = flattened_colors[conf_mask]
            
            n_points_in_frame = len(filtered_points)
            
            
            n_samples = min(num_points_per_frame, n_points_in_frame)
            
            if n_samples > 0:
                sampled_idx = np.random.choice(n_points_in_frame, n_samples, replace=False)
                sampled_pts = filtered_points[sampled_idx]
                sampled_colors = filtered_colors[sampled_idx]

                points_buffer = np.concatenate((points_buffer, sampled_pts), axis=0)
                colors_buffer = np.concatenate((colors_buffer, sampled_colors), axis=0)


                point_cloud_handle.points = points_buffer
                point_cloud_handle.colors = colors_buffer
            
                print(f"consumer: point cloud updated with {n_samples} new points, total {len(points_buffer)} points now.")
            else:
                print("consumer: no points passed the confidence threshold in this frame.")

        except Empty:
            pass
        
        except Exception as e:
            print(f"consumer: encountered an error: {e}")
            break
            
    print("consumer: exiting visualization thread.")
    
def get_model_from_scene(per_frame_res, save_dir, 
                         num_points_save=200000, 
                         conf_thres_res=3, 
                         valid_masks=None
                        ):  
        
    # collect the registered point clouds and rgb colors
    pcds = []
    rgbs = []
    pred_frame_num = len(per_frame_res['l2w_pcds'])
    registered_confs = per_frame_res['l2w_confs']   
    registered_pcds = per_frame_res['l2w_pcds']
    rgb_imgs = per_frame_res['rgb_imgs']
    for i in range(pred_frame_num):
        registered_pcd = to_numpy(registered_pcds[i])
        if registered_pcd.shape[0] == 3:
            registered_pcd = registered_pcd.transpose(1,2,0)
        registered_pcd = registered_pcd.reshape(-1,3)
        rgb = rgb_imgs[i].reshape(-1,3)
        pcds.append(registered_pcd)
        rgbs.append(rgb)
        
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
    sampled_pts[:, :2] *= -1 # flip the x,y axis for better visualization
    
    save_name = f"recon.glb"
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=sampled_pts, colors=sampled_rgbs/255.))
    save_path = join(save_dir, save_name)
    scene.export(save_path)

    return save_path

def display_inputs(images):
    img_label = "Click or use the left/right arrow keys to browse images", 

    if images is None or len(images) == 0: 
        return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                gradio.update(value=None, visible=False, scale=2, height=300,)]  

    if isinstance(images, str): 
        file_path = images
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        if any(file_path.endswith(ext) for ext in video_extensions):
            return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                gradio.update(value=file_path, autoplay=True, visible=True, scale=2, height=300,)]
        else:
            return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                    gradio.update(value=None, visible=False, scale=2, height=300,)] 
            
    return [gradio.update(label=img_label, value=images, visible=True, selected_index=0, scale=2, preview=True, height=300,),
            gradio.update(value=None, visible=False, scale=2, height=300,)]

def change_inputfile_type(input_type):
    # 为所有可能的输入组件和video_extract_fps准备更新对象
    update_file = gradio.update(visible=False)
    update_webcam = gradio.update(visible=False)
    update_external_webcam_html = gradio.update(visible=False)
    update_video_fps = gradio.update(visible=False, value=1, scale=0) # 默认值和隐藏
    update_url = gradio.update(visible=False)

    if input_type == "directory":
        update_file = gradio.update(file_count="directory", file_types=["image"],
                                 label="Select a directory containing images", visible=True, value=None) # value=None清除之前的文件
    elif input_type == "images":
        update_file = gradio.update(file_count="multiple", file_types=["image"],
                                 label="Upload multiple images", visible=True, value=None)
    elif input_type == "video":
        update_file = gradio.update(file_count="single", file_types=["video"],
                                 label="Upload a mp4 video", visible=True, value=None)
        update_video_fps = gradio.update(interactive=True, scale=1, visible=True, value=5) # 显式设置值
    elif input_type == "webcamera":
        # 如果你希望直接使用Gradio的内置摄像头
        #update_webcam = gradio.update(visible=True)
        # 如果你坚持嵌入外部网页，则显示HTML组件
        update_external_webcam_html = gradio.update(visible=True)
        update_url = gradio.update(visible=True)
        update_video_fps = gradio.update(interactive=True, scale = 1, visible=True, value = 5)

    # 这里的返回顺序必须与main_demo中outputs的顺序一致
    return update_file, update_webcam, update_external_webcam_html, update_video_fps, update_url,update_url,update_url,update_url
    
def change_kf_stride_type(kf_stride, inputfiles, win_r):
    max_kf_stride = 10
    if kf_stride == "auto":
        kf_stride_fix = gradio.Slider(value=-1,minimum=-1, maximum=-1, step=1, 
                                      visible=False, interactive=True, 
                                      label="stride between keyframes",
                                      info="For I2P reconstruction!")
    elif kf_stride == "manual setting":
        kf_stride_fix = gradio.Slider(value=1,minimum=1, maximum=max_kf_stride, step=1, 
                                      visible=True, interactive=True, 
                                      label="stride between keyframes",
                                      info="For I2P reconstruction!")
    return kf_stride_fix

def change_buffer_strategy(buffer_strategy):
    if buffer_strategy == "reservoir" or buffer_strategy == "fifo":
        buffer_size = gradio.Number(value=100, precision=0, minimum=1,
                                    interactive=True, 
                                    visible=True,
                                    label="size of the buffering set",
                                    info="For L2W reconstruction!")
    elif buffer_strategy == "unbounded":
        buffer_size = gradio.Number(value=10000, precision=0, minimum=1,
                                    interactive=True, 
                                    visible=False,
                                    label="size of the buffering set",
                                    info="For L2W reconstruction!")
    return buffer_size

def main_demo(i2p_model, l2w_model, device, tmpdirname, server_name, server_port):
    recon_scene_func = functools.partial(recon_scene, i2p_model, l2w_model, device)
    
    
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="SLAM3R Demo",) as demo:
        # scene state is save so that you can change num_points_save... without rerunning the inference
        per_frame_res = gradio.State(None)
        tmpdir_name = gradio.State(tmpdirname)
        auth_url = gradio.State("")
        
        gradio.HTML('<h2 style="text-align: center;">SLAM3R Demo</h2>')
        with gradio.Column():
            with gradio.Row():
                with gradio.Column():
                    input_type = gradio.Dropdown([ "directory", "images", "video", "webcamera"],
                                                scale=1,
                                                value='directory', label="select type of input files")
                    video_extract_fps = gradio.Number(value=1,
                                                    scale=0,
                                                    interactive=True,
                                                    visible=False,
                                                    label="fps for extracting frames from video")
                    input_url_web_cam = gradio.Textbox(label="your ip camera's url",
                                                   visible=False,
                                                   interactive=True
                                                   )
                    with gradio.Row():
                        web_cam_account = gradio.Textbox(label="your account",
                                                         visible= False, interactive=True)
                        web_cam_password = gradio.Textbox(label="your password",
                                                          visible= False, interactive=True)
                    confirm_button = gradio.Button("apply", visible= False, interactive=True)
                    
                    
                
                inputfiles = gradio.File(file_count="directory", file_types=["image"],
                                         scale=2,
                                         height=200,
                                         label="Select a directory containing images")
                inputfiles_webcam = gradio.Image(sources="webcam", type="filepath",
                                                  scale=2,
                                                  height=200,
                                                  label="Webcam Input",
                                                  visible=False) # 默认隐藏
                
                
                
                inputfiles_external_webcam_html = gradio.HTML(""
                )
              
                image_gallery = gradio.Gallery(label="Click or use the left/right arrow keys to browse images",
                                            visible=False,
                                            selected_index=0,
                                            preview=True,   
                                            height=300,
                                            scale=2)
                video_gallery = gradio.Video(label="Uploaded Video",
                                            visible=False,
                                            height=300,
                                            scale=2)
                

            with gradio.Row():
                kf_stride = gradio.Dropdown(["manual setting"], label="how to choose stride between keyframes",
                                           value="manual setting", interactive=True,  
                                           info="For I2P reconstruction!")
                kf_stride_fix = gradio.Slider(value=3, minimum=0, maximum=100, step=1, 
                                              visible=True, interactive=True, 
                                              label="stride between keyframes",
                                              info="For I2P reconstruction!")
                win_r = gradio.Number(value=5, precision=0, minimum=1, maximum=200,
                                      interactive=True, 
                                      label="the radius of the input window",
                                      info="For I2P reconstruction!")
                initial_winsize = gradio.Number(value=5, precision=0, minimum=2, maximum=200,
                                      interactive=True, 
                                      label="the number of frames for initialization",
                                      info="For I2P reconstruction!")
                conf_thres_i2p = gradio.Slider(value=1.5, minimum=1., maximum=10,
                                      interactive=True, 
                                      label="confidence threshold for the i2p model",
                                      info="For I2P reconstruction!")
            
            with gradio.Row():
                num_scene_frame = gradio.Slider(value=10, minimum=1., maximum=100, step=1,
                                      interactive=True, 
                                      label="the number of scene frames for reference",
                                      info="For L2W reconstruction!")
                buffer_strategy = gradio.Dropdown(["reservoir", "fifo","unbounded"], 
                                           value='reservoir', interactive=True,  
                                           label="strategy for buffer management",
                                           info="For L2W reconstruction!")
                buffer_size = gradio.Number(value=100, precision=0, minimum=1,
                                      interactive=True, 
                                      visible=True,
                                      label="size of the buffering set",
                                      info="For L2W reconstruction!")
                update_buffer_intv = gradio.Number(value=1, precision=0, minimum=1,
                                      interactive=True, 
                                      label="the interval of updating the buffering set",
                                      info="For L2W reconstruction!")
            
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                conf_thres_l2w = gradio.Slider(value=12, minimum=1., maximum=100,
                                      interactive=True, 
                                      label="confidence threshold for the result",
                                      )
                # adjust the camera size in the output pointcloud
                num_points_save = gradio.Number(value=1000000, precision=0, minimum=1,
                                      interactive=True, 
                                      label="number of points sampled from the result",
                                      )
            with gradio.Row():
                outmodel = gradio.Model3D(height=500,
                                        clear_color=(0.,0.,0.,0.3)) 
                
                outtext = gradio.HTML('<iframe src="http://localhost:8080" width="100%" height="600px" style="border:none;"></iframe>', visible=True)
                
            # events
            inputfiles.change(display_inputs,
                                inputs=[inputfiles],
                                outputs=[image_gallery, video_gallery])
            input_type.change(change_inputfile_type,
                                inputs=[input_type],
                                outputs=[inputfiles, inputfiles_webcam, inputfiles_external_webcam_html, video_extract_fps,input_url_web_cam, confirm_button, web_cam_account, web_cam_password])
            kf_stride.change(change_kf_stride_type,
                                inputs=[kf_stride, inputfiles, win_r],
                                outputs=[kf_stride_fix])
            buffer_strategy.change(change_buffer_strategy,
                                inputs=[buffer_strategy],
                                outputs=[buffer_size])
            run_btn.click(fn=recon_scene_func,
                          inputs=[tmpdir_name, video_extract_fps,
                                  input_type, auth_url,
                                  inputfiles, kf_stride_fix, win_r, initial_winsize, conf_thres_i2p,
                                  num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size,
                                  conf_thres_l2w, num_points_save],
                          outputs=[outmodel, per_frame_res])
            conf_thres_l2w.release(fn=get_model_from_scene,
                                 inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w],
                                 outputs=outmodel)
            num_points_save.change(fn=get_model_from_scene,
                            inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w],
                            outputs=outmodel)
            confirm_button.click(fn=change_web_camera_url,
                                 inputs=[inputfiles_external_webcam_html, input_url_web_cam, web_cam_account, web_cam_password,auth_url],
                                 outputs=[inputfiles_external_webcam_html, auth_url])

    demo.launch(share=False, server_name=server_name, server_port=server_port,debug=True)

def change_web_camera_url(inputs_external_webcam, web_url, web_cam_account, web_cam_password, auth_url):
    # 将 URL 分割成协议和其余部分
    protocol, rest_of_url = web_url.split("://")

    # 构造新的 URL
    auth_url = gradio.State(f"{protocol}://{web_cam_account}:{web_cam_password}@{rest_of_url}")
    

    inputs_external_webcam = gradio.HTML(
                    f"""
                    <p>Web Camera presentation：</p>
                    <iframe src="{web_url}" width="100%" height="600px" style="border:none;"></iframe>
                    """,
                    visible=True,
                )
    return inputs_external_webcam, auth_url
    
    

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'
    
    i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
    l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
    i2p_model.to(args.device)
    l2w_model.to(args.device)
    i2p_model.eval()
    l2w_model.eval()

    # slam3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='slam3r_gradio_demo') as tmpdirname:
        main_demo(i2p_model, l2w_model, args.device, tmpdirname, server_name, args.server_port)


if __name__ == "__main__":
    productor = threading.Thread(target=main)
    displayer = threading.Thread(target=print_model_viser)
    
    productor.start()
    displayer.start()