import argparse
import glob
import natsort
import os
import pathlib

import nibabel as nib
import numpy as np
import scipy.io as sio
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import contrastStretch, normalize
from utils import myCrop3D
from utils import performDenoising


def generate_constraint_maps_batch(parse_cfg):
    '''
    Generates constraint maps for each subject (arranged in a separate folder)
    and saves constraint maps as mat files
    '''
    datadir = parse_cfg.data_dir  # location of pre-processed nii files for pretraining
    num_param_clusters = parse_cfg.numCluster  # number of clusters for Kmeans, 20 or 30 work well
    opShape = (parse_cfg.opShape, parse_cfg.opShape)  # output image shape for pretraining
    contrast = parse_cfg.contrast
    save_base_dir = parse_cfg.save_dir

    sub_list = natsort.natsorted(os.listdir(datadir))

    # Generate constraint maps for each subject in the training list
    for subName in tqdm(sub_list):
        print('\nSubject ', subName)
        save_dir = os.path.join(save_base_dir, contrast)
        save_str = f'Constraint_map_{subName}_' + str(num_param_clusters) + '.mat'
        save_path = os.path.join(save_dir, save_str)
        img_exists = len(glob.glob(os.path.join(datadir, subName, f"{contrast}/imagesTr/*")))
        lbl_exists = len(glob.glob(os.path.join(datadir, subName, f"{contrast}/labelsTr/*volume*")))
        if not os.path.exists(save_path) and img_exists and lbl_exists:
            img, mask = load_img(datadir, subName, contrast, opShape)
            print('Generating parametric cluster for K=', num_param_clusters)
            kp = mask
            # kp = generate_parametric_clusters(img, mask, num_cluster=num_param_clusters, random_state=0)
            # kp = np.stack([normalize(kp[...,i]) for i in range(kp.shape[-1])],-1)
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            temp = {}
            temp['param'] = kp
            temp['img'] = img
            temp['mask'] = mask
            sio.savemat(save_path, temp)





def generate_parametric_clusters(parameter_volume, mask, num_cluster=10, random_state=0, num_PC_2retain=4):
    '''
    Unsupervised KMeans clustering on 3D/4D MR volumes to summarize tissue parameter information
    in a constraint map.
    Performs PCA for denoising and dimensionality reduction
    (if contrast dimension is large in multi-contrast space e.g., T2-weighted TE images for T2 constraint maps)
    Input: Parameter volume (4D) HxWxDxT or (3D) HxWxT where T is the contrast dimension
    Output: Parameter constraint map
    '''
    assert len(parameter_volume.shape) == 4

    xDim, yDim, zDim, tDim = parameter_volume.shape

    ''' Optional: Use this section to generate a brain mask to avoid including background'''
    # mask = np.zeros(parameter_volume[...,0].shape)    # optional to generate a mask to avoid including background in constraint maps
    # mask[parameter_volume[...,0] > 0] = 1

    ''' Perform PCA decomposition'''
    temp_f = np.reshape(parameter_volume, (-1, tDim))
    temp_pc = PCA(n_components=num_PC_2retain).fit_transform((temp_f))
    temp_pc = np.reshape(temp_pc, (xDim, yDim, zDim, num_PC_2retain))
    temp_pc = normalize(temp_pc)

    weight_map = mask#np.stack([bounding_box_mask(mask[...,i], margin=5) for i in range(mask.shape[-1])],-1)

    ''' Denoise PC images using TV'''
    for idx in range(num_PC_2retain):
        # temp_pc[..., idx] = performDenoising(temp_pc[..., idx], wts=40)
        temp_pc[...,idx] = weight_map * performDenoising(temp_pc[...,idx], wts=5)

    img_blk_vec = np.reshape(temp_pc, (-1, num_PC_2retain))
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=random_state).fit(img_blk_vec)
    class_labels_vec = kmeans.labels_
    param_clusters = np.reshape(class_labels_vec, (xDim, yDim, zDim))
    return param_clusters

def bounding_box_mask(mask, margin=5):
    """
    Generate a binary mask of the bounding box around a segmentation mask.

    Parameters:
        mask (np.ndarray): A 2D binary segmentation mask (values: 0 or 1).
        margin (int): Margin to add around the bounding box.

    Returns:
        np.ndarray: A binary mask of the same shape as the input with the bounding box.
    """
    # Find the indices of non-zero (True) elements in the mask
    rows, cols = np.nonzero(mask)

    # If the mask is empty, return an array of zeros
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # Calculate the bounding box
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()

    # Add the margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(mask.shape[1] - 1, x_max + margin)
    y_max = min(mask.shape[0] - 1, y_max + margin)

    # Create an array to represent the bounding box
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

    return bbox_mask

''' Use an appropriate dataloader to load multi-contrast training data
    Example here is for the brain tumor segmentation dataset'''


def load_img(datadir, subName, contrast, opShape):
    print(f'Loading {contrast} image for ', subName)
    paths = glob.glob(os.path.join(datadir, subName, f"{contrast}/imagesTr/*"))
    temp = [np.flip(np.rot90(nib.load(p).get_fdata(), -1),1) for p in paths]
    temp = np.stack(temp, axis=-1)
    temp = np.squeeze(temp,-1) if temp.shape[-1] == 1 else temp
    if temp.shape[0] > opShape[0] or temp.shape[1] > opShape[1]:
        temp = myCrop3D(temp, opShape)
    temp = normalize(temp)

    # read mask
    mask_path = glob.glob(os.path.join(datadir, subName, f"{contrast}/labelsTr/*volume*"))
    #mask_cortex_path = glob.glob(os.path.join(datadir, subName, f"{contrast}/labelsTr/*cortex*"))
    path_mask_left = [s for s in mask_path if "left" in s.lower()][0]
    path_mask_right = [s for s in mask_path if "right" in s.lower()][0]
    #path_cortex_left = [s for s in mask_cortex_path if "left" in s.lower()][0]
    #path_cortex_right = [s for s in mask_cortex_path if "right" in s.lower()][0]

    vol_left = np.flip(np.rot90(nib.load(path_mask_left).get_fdata(), -1),1)
    vol_right = np.flip(np.rot90(nib.load(path_mask_right).get_fdata(), -1), 1)
    cortex_left = np.zeros_like(vol_left)#np.flip(np.rot90(nib.load(path_cortex_left).get_fdata(), -1), 1)
    cortex_right = np.zeros_like(vol_right)#np.flip(np.rot90(nib.load(path_cortex_right).get_fdata(), -1), 1)
    medulla_right = vol_right - cortex_right
    medulla_left = vol_left - cortex_left
    background = np.ones_like(vol_left) - vol_left - vol_right
    mask = np.argmax(np.stack((background, cortex_left, cortex_right, medulla_left, medulla_right), axis=-1),-1)
    mask[mask == 0] = 5
    if mask.shape[0] > opShape[0] or mask.shape[1] > opShape[1]:
        mask = myCrop3D(mask, opShape)

    # histogram based channel-wise contrast stretching
    temp = contrastStretch(temp, mask, 0.01, 99.9)
    print(temp.shape, mask.shape)
    return temp, mask

def main():
    description_txt = 'Constraint map generation for CCL'
    parser = argparse.ArgumentParser(description=description_txt)
    parser.add_argument("--save_dir", type=str,
                        default="/home/raghoul1/Renal_fMRI/data/Constraint_maps/param")
    parser.add_argument("--data_dir", type=str,
                        default="/home/raghoul1/Renal_fMRI/data/resampled")
    parser.add_argument("--opShape", type=int, default=256,
                        help="Matrix size X (Int)(Default 160)")
    parser.add_argument("--contrast", type=str, default="BOLD")
    parser.add_argument("--numCluster", type=int, default=20,
                        help="NUmber of clusters for Kmeans (Int)(Default 20)")
    parse_cfg = parser.parse_args()
    if parse_cfg.data_dir == None:
        raise ValueError('An input data directory must be provided')
    if parse_cfg.save_dir == None:
        raise ValueError('An output data directory must be provided to save constraint maps')
    generate_constraint_maps_batch(parse_cfg)


if __name__ == "__main__":
    print('Running generate_constraint_maps.py')
    main()

