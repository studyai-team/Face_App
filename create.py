import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

from infer import InferenceWrapper

import math


# PytorchのTensor型のデータをPILのImage型に変換する。bilayer-modelのtest.pyから引用
def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255

    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))


# 画像を正方形のサイズに拡張する
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


# 設定、触る可能性があるのはdir関係のみだと思われる
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

module = InferenceWrapper(args_dict)

for i in range(2):
    # 生成したい表情の画像をリストで格納(typeA: 真顔 typeB: 笑顔)
    type = ['examples/target/typeA.jpg', 'examples/target/typeB.png'][i]
    type_name = ["typeA", "typeB"][i]
    # 表情を変更したい画像をexample/images/から読み込む
    for im in glob.glob("examples/images/*"):
        # 表情を変更したい画像をリストで格納。真顔、笑顔の画像を両方作成する
        input_data_dict = {
            'source_imgs': np.asarray(Image.open(im)),  # H x W x 3
            'target_imgs': np.asarray(Image.open(type))[None]}  # B x H x W x # 3

        file_name = os.path.basename(im)
        print(type_name, file_name)

        output_data_dict = module(input_data_dict, crop_data=True)

        # 生成結果の出力、以下結果の画像の説明
        # source -> 顔の中心を画像の真ん中に持ってくる前処理を行った後の入力データ
        # hf -> 高周波領域の画像(ネットワーク内部で使用している)
        # target -> 真顔または笑顔の表情のランドマーク
        # pred -> 生成画像
        # cropped -> 生成画像を基に画像の意味のある部分を切り取った画像(最終的な出力)

        source_img = to_image(output_data_dict['source_imgs'][0, 0])
        source_img.save("examples/results/{}_source_{}".format(type_name, file_name))

        hf_texture = to_image(output_data_dict['pred_enh_tex_hf_rgbs'][0, 0])
        hf_texture.save("examples/results/{}_hf_{}".format(type_name, file_name))

        target_pose = to_image(output_data_dict['target_stickmen'][0, 0])
        target_pose.save("examples/results/{}_target_{}".format(type_name, file_name))

        pred_img = to_image(output_data_dict['pred_enh_target_imgs'][0, 0], output_data_dict['pred_target_segs'][0, 0])
        pred_img.save("examples/results/{}_pred_{}".format(type_name, file_name))

        # Preprocess

        height, width = output_data_dict["source_pos"][3] - output_data_dict["source_pos"][1], \
                        output_data_dict["source_pos"][2] - output_data_dict["source_pos"][0]
        output_size = output_data_dict["source_pos"][2] - output_data_dict["source_pos"][0]

        real_range_x = [0, output_data_dict["source_pos"][2] - output_data_dict["source_pos"][0]]
        effective_range_x = [0 - output_data_dict["source_pos"][0],
                             input_data_dict["source_imgs"].shape[1] - output_data_dict["source_pos"][0]]

        real_range_y = [0, output_data_dict["source_pos"][3] - output_data_dict["source_pos"][1]]
        effective_range_y = [0 - output_data_dict["source_pos"][1],
                             input_data_dict["source_imgs"].shape[0] - output_data_dict["source_pos"][1]]

        crop_range_x = [math.ceil(max(0, effective_range_x[0]) * (256 / output_size)),
                        math.floor(min(real_range_x[1], effective_range_x[1]) * (256 / output_size))]
        crop_range_y = [math.ceil(max(0, effective_range_y[0]) * (256 / output_size)),
                        math.floor(min(real_range_y[1], effective_range_y[1]) * (256 / output_size))]

        cropped_img = pred_img.crop((crop_range_x[0], crop_range_y[0], crop_range_x[1], crop_range_y[1]))
        cropped_img = expand2square(cropped_img, (255, 255, 255)).resize((256, 256))
        cropped_img.save("examples/results/{}_cropped_{}".format(type_name, file_name))
