# -*- coding: utf-8 -*-
import argparse
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import json

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
    print('Popularity score: ' + str(round(score, 2)))
    return round(score, 2)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, default='images/0.jpg')
    # config = parser.parse_args()
    # image = Image.open(config.image_path)

    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load('model/model-resnet50.pth', map_location=torch.device('cpu')))
    model.eval()

    result = []

    for i in range(1, 23):
        vidcap = cv2.VideoCapture('./youtube/' + str(i) + '.mp4')
        success, vidcap_image = vidcap.read()
        count = 0

        # while success and count < max_frames:
        while success:
            name = str(i) + '_' + str(count)
            cv2.imwrite("./frames_saved/" + name + ".png", vidcap_image)  # save frame as png file
            cv2_image = cv2.cvtColor(vidcap_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            result.append((predict(pil_image, model), name))
            count += 10
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
            success, vidcap_image = vidcap.read()

    result.sort(reverse=True)
    print(result)

    with open('result.json', 'w') as filehandle:
        filehandle.write(json.dumps(result))
