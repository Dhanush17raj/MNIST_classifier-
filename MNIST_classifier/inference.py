import requests
import json
import numpy as np
import tensorflow as tf
import time
import cv2
from mnist_classifier import model

image = cv2.imread('/home/dhanush/Documents/ML/bag.jpeg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (28, 28))

image = np.expand_dims(image, axis = 0)

#creating an inference request

start_time = time.time()

url = "http://localhost:8501/v1/models/mnist_classifier:predict"

data = json.dumps({"signature_name":"serving_default", "instances": image.tolist()})#encoding the image data to json format


headers = {"content-type": "application/json"} # header to identify the json format

response = requests.post(url, data = data, headers = headers)
prediction = json.loads(response.text)["predictions"]

predictions = model.predict('/home/dhanush/Documents/ML/bag.jpeg')
print(predictions)


# predict(model, image, label)