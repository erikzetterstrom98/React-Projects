from flask import Flask, request
from flask_cors import CORS
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage, PILToTensor
import PIL
from image_classifier import *

app = Flask(__name__)
model: MNIST_Classifier = None
cors = CORS(
    app = app,
    origins = 'http://localhost:*'
)

def convert_data_to_image(data: List) -> torch.Tensor:
    height, width = data['image_height'], data['image_width']
    
    image = torch.zeros((height*width))

    for index, pixel in enumerate(data['data']):
        image[index] = pixel

    image = ToPILImage()(image.reshape((height, width)))
    image = image.resize((28, 28))
    image = PILToTensor()(image)
    return image.float()/255
    
def load_model():
    model = torch.load('model-efficient_capsules.pth')
    model.eval()
    return model

@app.route('/')
def test_function():
    return 'kurva'

@app.route('/predict', methods = ['POST'])
def hoo():
    image_data = request.get_json()
    image = convert_data_to_image(image_data)
    if torch.equal(image, torch.zeros(image.shape)):
        return {"guess": -1}
    
    # print(image.squeeze(dim=0))
    # print(image.squeeze(dim=0).shape)
    # plt.imshow(image.squeeze(dim=0))
    # plt.show()
    capsule_lengths, probabilities = model(image.unsqueeze(dim = 0))
    #print(capsule_lengths)
    #print(probabilities)
    #print(capsule_lengths.shape)
    result_json = {
        "capsule lengths": {},
        "probabilities": {},
        "guess": torch.argmax(capsule_lengths, dim = 1).item()
    }

    for number in range(10):
        result_json["capsule lengths"][str(number)] = capsule_lengths[0][number].item()
        result_json["probabilities"][str(number)] = probabilities[0][number].item()

    #print(result_json)
    return result_json

def main():
    global model
    model = load_model()
    app.run(debug = True)

if __name__ == '__main__':
    main()