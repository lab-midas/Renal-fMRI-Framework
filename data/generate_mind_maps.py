import os, natsort
import scipy.io as sio
import pathlib
import sys, numpy as np
import nibabel as nib
import argparse
from utils import myCrop3D
from utils import contrastStretch, normalize
import glob
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt as edt

def generate_mind_maps_batch(parse_cfg):
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
    for subName in tqdm(sub_list[50:60]):
        print('\nSubject ', subName)
        save_dir = os.path.join(save_base_dir, contrast)
        save_str = f'Constraint_map_{subName}_' + str(num_param_clusters) + '.mat'
        save_path = os.path.join(save_dir, save_str)
        img_exists = len(glob.glob(os.path.join(datadir, subName, f"{contrast}/imagesTr/*")))
        #lbl_exists = len(glob.glob(os.path.join(datadir, subName, f"{contrast}/labelsTr/*cortex*")))
        if not os.path.exists(save_path) and img_exists:# and lbl_exists:
            img, mask = load_img(datadir, subName, contrast, opShape)

            print('Generating mind maps')
            img = img[...,None] if img.ndim == 3 else img
            kp = np.zeros_like(img)
            for i in range(kp.shape[-2]):
                for j in tqdm(range(kp.shape[-1])):
                    kp[...,i,j] = mo_mind(img[...,i,j], mask[...,i])


            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            temp = {}
            temp['param'] = kp
            temp['img'] = img
            temp['mask'] = mask
            sio.savemat(save_path, temp)


def gaussian_weighted_patch(image, x, patch_size, sigma):
    """
    Extract a Gaussian-weighted patch around the pixel `x`.
    """
    h, w = image.shape
    ps = patch_size // 2
    x_min, x_max = max(0, x[0] - ps), min(h, x[0] + ps + 1)
    y_min, y_max = max(0, x[1] - ps), min(w, x[1] + ps + 1)
    patch = image[x_min:x_max, y_min:y_max]

    # Generate Gaussian weights
    gauss = np.exp(-((np.arange(-ps, ps + 1)) ** 2) / (2 * sigma ** 2))
    gauss_kernel = np.outer(gauss, gauss)
    gauss_kernel /= gauss_kernel.sum()

    # Apply Gaussian weights to the patch
    weighted_patch = patch * gauss_kernel[:patch.shape[0], :patch.shape[1]]
    return weighted_patch

def generate_weight_map(mask):
        decay_rate = 0.25
        seg_map = np.copy(mask)
        seg_map[seg_map > 0] = 1
        seg_map_inv = ~seg_map.astype(bool)
        distance_map = edt(seg_map_inv)
        weight_map = np.exp(-decay_rate * distance_map)
        return weight_map.astype(np.float32)


def modified_variance(image, neighborhood_size):
    """
    Compute the modified variance `V_m` for each pixel in the image.
    """
    h, w = image.shape
    pad_size = neighborhood_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    # print(padded_image.shape)
    variance_map = np.zeros((h, w))
    sum_map = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            x_min, x_max = i, i + neighborhood_size
            y_min, y_max = j, j + neighborhood_size
            # print(x_min, x_max, y_min, y_max)
            neighborhood = padded_image[x_min:x_max, y_min:y_max]
            # print(neighborhood)
            patch_sums = np.sum(neighborhood, axis=(0, 1))
            # print(patch_sums)
            sum_map[i, j] = patch_sums

    for i in range(h):
        for j in range(w):
            x_min, x_max = i, i + neighborhood_size
            y_min, y_max = j, j + neighborhood_size
            neighborhood = padded_image[x_min:x_max, y_min:y_max]
            variance_map[i, j] = np.var(neighborhood) + 1e-6

    return variance_map  # + 1e-6  # Small value to avoid division by zero


def mo_mind(image, mask=None, patch_size=3, search_region=[(-1, 0), (1, 0), (0, -1), (0, 1)], sigma=1.0):
    """
    Compute the mo-MIND descriptor for the entire image.

    Parameters:
    - image: Input 2D grayscale image.
    - patch_size: Size of Gaussian-weighted patches.
    - search_region: List of relative offsets defining the search region.
    - sigma: Standard deviation of the Gaussian kernel.

    Returns:
    - mo_mind_map: Descriptor map of shape (H, W, len(search_region)).
    """
    h, w = image.shape
    mo_mind_map = np.zeros((h, w, len(search_region)))
    if mask is not None:
        map = generate_weight_map(mask)
        image = image*map
    # Compute modified variance
    variance_map = modified_variance(image, patch_size)

    # Iterate over search region
    for idx, r in enumerate(search_region):
        shifted_image = np.roll(image, shift=r, axis=(0, 1))

        # Compute Gaussian-weighted patch differences
        for i in range(h):
            for j in range(w):
                patch_x = gaussian_weighted_patch(image, (i, j), patch_size, sigma)
                patch_xr = gaussian_weighted_patch(shifted_image, (i, j), patch_size, sigma)
                patch_distance = np.sum((patch_x - patch_xr) ** 2)
                mo_mind_map[i, j, idx] = np.exp(-patch_distance / variance_map[i, j])

    return np.mean(mo_mind_map,-1)


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
    paths = glob.glob(os.path.join(datadir, subName, f"{contrast}/imagesTr/*"))[:1]
    temp = [np.flip(np.rot90(nib.load(p).get_fdata(), -1),1) for p in paths]
    temp = np.stack(temp, axis=-1)
    temp = np.squeeze(temp,-1) if temp.shape[-1] == 1 else temp
    if temp.shape[0] > opShape[0] or temp.shape[1] > opShape[1]:
        temp = myCrop3D(temp, opShape)
    temp = normalize(temp)

    # read mask
    mask_path = glob.glob(os.path.join(datadir, subName, f"{contrast}/labelsTr/*volume*"))
    mask_cortex_path = glob.glob(os.path.join(datadir, subName, f"{contrast}/labelsTr/*cortex*"))
    path_mask_left = [s for s in mask_path if "left" in s.lower()][0]
    path_mask_right = [s for s in mask_path if "right" in s.lower()][0]
    #path_cortex_left = [s for s in mask_cortex_path if "left" in s.lower()][0]
    #path_cortex_right = [s for s in mask_cortex_path if "right" in s.lower()][0]

    vol_left = np.flip(np.rot90(nib.load(path_mask_left).get_fdata(), -1),1)
    vol_right = np.flip(np.rot90(nib.load(path_mask_right).get_fdata(), -1), 1)
    #cortex_left = np.flip(np.rot90(nib.load(path_cortex_left).get_fdata(), -1), 1)
    #cortex_right = np.flip(np.rot90(nib.load(path_cortex_right).get_fdata(), -1), 1)
    #medulla_right = vol_right - cortex_right
    #medulla_left = vol_left - cortex_left
    background = np.ones_like(vol_left) - vol_left - vol_right
    mask = np.argmax(np.stack((background, vol_left, vol_right), axis=-1),-1)
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
                        default="/home/raghoul1/Renal_fMRI/data/mind_maps/guassian")
    parser.add_argument("--data_dir", type=str,
                        default="/home/raghoul1/Renal_fMRI/data/resampled")
    parser.add_argument("--opShape", type=int, default=256,
                        help="Matrix size X (Int)(Default 160)")
    parser.add_argument("--contrast", type=str, default="Diffusion")
    parser.add_argument("--numCluster", type=int, default=20,
                        help="NUmber of clusters for Kmeans (Int)(Default 20)")
    parse_cfg = parser.parse_args()
    if parse_cfg.data_dir == None:
        raise ValueError('An input data directory must be provided')
    if parse_cfg.save_dir == None:
        raise ValueError('An output data directory must be provided to save constraint maps')
    generate_mind_maps_batch(parse_cfg)


if __name__ == "__main__":
    print('Running generate_constraint_maps.py')
    main()

