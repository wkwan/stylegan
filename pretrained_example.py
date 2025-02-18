# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

import argparse
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import json

import seaborn as sns; sns.set()

import cv2

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    image = Transform(image)
    image = image.unsqueeze(0)
    return image


def predict(image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    score = preds.detach().numpy().item()
    print('Popularity score: ' + str(round(score, 10)))
    return str(round(score, 10))

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl

    # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    f = open('./karras2019stylegan-ffhq-1024x1024.pkl', 'rb')
    _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    image_to_latent = {}

    def make_image(i):

        # Pick latent vector.
        rnd = np.random.RandomState()
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)


        pil_image = PIL.Image.fromarray(images[0], 'RGB')

        # Instagram popularity prediction
        model = torchvision.models.resnet50()
        # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
        model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        model.load_state_dict(torch.load('model/model-resnet50.pth', map_location=torch.device('cpu')))
        model.eval()

        score = predict(pil_image, model)
        png_filename = os.path.join(config.result_dir, 'example_{}_{}.png'.format(i, score))
        pil_image.save(png_filename)
        image_to_latent[png_filename] = latents.tolist()


    for i in range(3):
        make_image(i)

    with open('image_to_latent_0.json', 'w') as f:
        json.dump(image_to_latent, f)

    for i in range(3, 6):
        make_image(i)

    with open('image_to_latent_1.json', 'w') as f:
        json.dump(image_to_latent, f)

    for i in range(6, 9):
        make_image(i)

    with open('image_to_latent_2.json', 'w') as f:
        json.dump(image_to_latent, f)

    for i in range(9, 12):
        make_image(i)

    with open('image_to_latent_3.json', 'w') as f:
        json.dump(image_to_latent, f)


if __name__ == "__main__":
    main()
