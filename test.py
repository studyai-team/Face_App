# %%

# import sys

# sys.path.append('../../')

import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from infer import InferenceWrapper

import math


# %% md

### Load the model

# %%

args_dict = {
    'project_dir': '.',
    'init_experiment_dir': 'runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}

# %%

module = InferenceWrapper(args_dict)

# %% md

### Calculate poses, segmentation and do the inference

# %%

input_data_dict = {
    'source_imgs': np.asarray(Image.open('examples/ひげ.png')),  # H x W x 3
    'target_imgs': np.asarray(Image.open('examples/target/typeA.jpg'))[None]}  # B x H x W x # 3

output_data_dict = module(input_data_dict, crop_data=True)


# %%

def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255

    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))


# https://github.com/nkmk/python-tools/blob/bff489f645f5bf854c3b7a3d406dc8d491973d0c/tool/lib/imagelib.py#L57-L68
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# %%

source_img = to_image(output_data_dict['source_imgs'][0, 0])
source_img.save("source.png")

# %%

hf_texture = to_image(output_data_dict['pred_enh_tex_hf_rgbs'][0, 0])
hf_texture.save("hf.png")

# %%

target_pose = to_image(output_data_dict['target_stickmen'][0, 0])
target_pose.save("target.png")

# %%

pred_img = to_image(output_data_dict['pred_enh_target_imgs'][0, 0], output_data_dict['pred_target_segs'][0, 0])
pred_img.save("pred.png")

height, width = output_data_dict["source_pos"][3] - output_data_dict["source_pos"][1], output_data_dict["source_pos"][2] - output_data_dict["source_pos"][0]
output_size = output_data_dict["source_pos"][2] - output_data_dict["source_pos"][0]

real_range_x = [0, output_data_dict["source_pos"][2] - output_data_dict["source_pos"][0]]
effective_range_x = [0 - output_data_dict["source_pos"][0], input_data_dict["source_imgs"].shape[1] - output_data_dict["source_pos"][0]]

real_range_y = [0, output_data_dict["source_pos"][3] - output_data_dict["source_pos"][1]]
effective_range_y = [0 - output_data_dict["source_pos"][1], input_data_dict["source_imgs"].shape[0] - output_data_dict["source_pos"][1]]

crop_range_x = [math.ceil(max(0, effective_range_x[0]) * (256 / output_size)), math.floor(min(real_range_x[1], effective_range_x[1]) * (256 / output_size))]
crop_range_y = [math.ceil(max(0, effective_range_y[0]) * (256 / output_size)), math.floor(min(real_range_y[1], effective_range_y[1]) * (256 / output_size))]

cropped_img = pred_img.crop((crop_range_x[0], crop_range_y[0], crop_range_x[1], crop_range_y[1]))
cropped_img = expand2square(cropped_img, (255, 255, 255)).resize((256, 256))
cropped_img.save("cropped.png")
