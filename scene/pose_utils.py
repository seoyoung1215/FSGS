import numpy as np
import os
import sys
import imageio
# import skimage.transform

from .colmap_read_model import read_cameras_binary, read_images_binary, read_points3d_binary

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)

    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    # print( 'Cameras', camdata, len(list_of_keys))

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)
    
    w2c_mats = []
    # all_hwf = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    aa = [k for k in imdata]

    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    # for kk in list_of_keys:
    #     cam = camdata[kk]
    #     h, w, f = cam.height, cam.width, cam.params[0]
    #     print(kk, h, w, f)
    #     hwf = np.array([h,w,f]).reshape([3,1])
    #     all_hwf.append(hwf)

    
    w2c_mats = np.stack(w2c_mats, 0)
    # all_hwf = np.stack(all_hwf, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # all_hwf = all_hwf.transpose([1,2,0])
    # poses = np.concatenate([poses, all_hwf], 1)
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    all_ind = []
    for k in pts3d:
        for ind in pts3d[k].image_ids:
            all_ind.append(ind)
    all_ind = sorted(np.unique(np.array(all_ind)))
    # print(all_ind)
    for k in pts3d:        
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind_raw in pts3d[k].image_ids:
            ind = all_ind.index(ind_raw)
            # print(len(cams), ind - 1)
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, 0.1), np.percentile(zs, 99.9)
        print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    print("saving dir of poses_bounds.npy", os.path.join(basedir, 'poses_bounds.npy'))
            
