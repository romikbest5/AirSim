import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import matplotlib.pyplot as plt

"""
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
responses = client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.DepthVis,False,False)])
response = responses[0]
img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(response.height, response.width, -1)


print(img_rgb)

plt.imshow(img_rgb)
plt.show()

client.enableApiControl(False)
"""