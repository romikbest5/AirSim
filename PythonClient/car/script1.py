import setup_path
import airsim
import sys
import math
import time
import argparse
import pprint
import numpy

def parse_lidarData(data):
    points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
    points = numpy.reshape(points, (int(points.shape[0] / 3), 3))
    return points




client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
"""
car_controls.throttle = 0.5
car_controls.steering = 1
client.setCarControls(car_controls)
#client.reset()
"""
#"""
lidarData = client.getLidarData(lidar_name='Lidar2')
print(lidarData)
points = parse_lidarData(lidarData)
print("\tReading: time_stamp: %d number_of_points: %d" % (lidarData.time_stamp, len(points)))
print(points)
print(points.shape)
#car_controls.throttle = 0.5
#client.setCarControls(car_controls)

#time.sleep(5)
print(client.simGetCollisionInfo().has_collided)
print(client.getCarState().speed)
client.reset()
client.enableApiControl(False)
#"""