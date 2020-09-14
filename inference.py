import argparse
import os
from PIL import Image
import numpy as np
import face_alignment

import torch
from torchvision.utils import save_image

from models import *
from networks import preprocess_image, preprocess_landmark

parser = argparse.ArgumentParser()
parser.add_argument("--source_image_path", default='')
parser.add_argument("--output", default='output/', help="where to save output")
parser.add_argument("--generator_path", default="saved_models/generator_10.pth", help="generator model pass")
parser.add_argument("--embedder_path", default="saved_models/embedder_10.pth", help="generator model pass")
parser.add_argument("--type", type=str, default="A", help="A: magao B: egao")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()

os.makedirs(opt.output, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Define model and load model checkpoint
generator = Generator().to(device)
embedder = Embedder().to(device)

if torch.cuda.is_available():
    generator.load_state_dict(torch.load(opt.generator_path))
    embedder.load_state_dict(torch.load(opt.embedder_path))
    face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
else:
    generator.load_state_dict(torch.load(opt.generator_path, map_location=torch.device("cpu")))
    embedder.load_state_dict(torch.load(opt.embedder_path, map_location=torch.device("cpu")))
    face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

# Preprocess landmarks
magao_landmark = preprocess_landmark("target_landmarks/typeA.png", face_aligner, device).type(Tensor)
egao_landmark = preprocess_landmark("target_landmarks/typeB.png", face_aligner, device).type(Tensor)

# Calculate image
output_1_m, output_2_m, output_3_m, output_4_m, output_5_m = embedder(magao_landmark)
output_1_e, output_2_e, output_3_e, output_4_e, output_5_e = embedder(egao_landmark)
if opt.type == "A":
    output_1, output_2, output_3, output_4, output_5 = output_1_m, output_2_m, output_3_m, output_4_m, output_5_m
elif opt.type == "B":
    output_1, output_2, output_3, output_4, output_5 = output_1_e, output_2_e, output_3_e, output_4_e, output_5_e
else:
    print("Unpredictable type")
    exit()

source_image = preprocess_image(opt.source_image_path, device).type(Tensor)
generated_image = generator(source_image, output_1, output_2, output_3, output_4, output_5)
save_image(generated_image, opt.output + "output.png")
print('Output generated image')
