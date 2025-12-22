import sys
sys.path.append('../')
from utils.utils import *

import numpy as np
from scipy import ndimage

from skimage.filters import sobel_h
from skimage.filters import sobel_v
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d as mp3d

from tensorflow.python.client import device_lib

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16, ResNet50

from tensorflow.nn import depthwise_conv2d
from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm, sin, cos, abs, reduce_variance
from tensorflow import stack, concat, expand_dims

import tensorflow_probability as tfp
import scienceplots
from mayavi  import mlab 

plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '300'})

model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_layers = []
for l in model.layers:
	if 'conv2d' in str(type(l)).lower():
		if l.kernel_size == (7, 7) or l.kernel_size == (3,3):
			conv_layers.append(l)

filters, _ = conv_layers[0].get_weights()
filters = filters #/ np.sqrt(reduce_variance(filters, axis=None))
theta = getSobelTF(filters)
print(filters.shape)
s, a = getSymAntiSymTF(filters)
a_mag = reduce_euclidean_norm(a, axis=[0,1])
s_mag = reduce_euclidean_norm(s, axis=[0,1])

mag = reduce_euclidean_norm(filters, axis=[0,1])
fig =  mlab.figure(size=(600, 643), bgcolor=(0.8980392156862745, 0.8980392156862745, 0.8980392156862745), fgcolor=(0, 0, 0))

mlab.clf()


_, dom_theta = getDominantAngle(filters)




print("Dominant angles (degrees): ", dom_theta*180/np.pi)
dom_theta = (dom_theta + np.pi) / (2*np.pi)
print("Dominant angles (degrees): ", dom_theta*180/np.pi)


n_filters = filters.shape[-1]

order = np.argsort(dom_theta)  # sort by angle
#base_colors = plt.cm.tab20c(np.linspace(0, 1, n_filters, endpoint=False))[:, :3]

c20  = plt.cm.tab20(np.linspace(0, 1, 22))[:, :3]
c20b = plt.cm.tab20b(np.linspace(0, 1, 22))[:, :3]
c20c = plt.cm.tab20c(np.linspace(0, 1, 22))[:, :3]

base_colors = np.vstack([c20, c20b, c20c])[:n_filters]

colors = np.zeros_like(base_colors)
colors[order] = base_colors   # smallest angle gets hue ~0, etc.


for F in range(filters.shape[-1]):
	x =(a_mag[:,F]*np.cos((theta[:,F]))).numpy()*9   #times 30 for random , *9 for vgg
	y =( a_mag[:,F]*np.sin((theta[:,F]))).numpy()*9
	z =(s_mag[:,F]*np.sign(np.mean(s, axis=(0,1)))[:,F]).numpy()*9





	mlab.points3d(x[0], y[0], z[0], np.ones(z[0].shape),  color=tuple(plt.cm.hsv(dom_theta)[F][:3]), scale_factor=0.5)
	mlab.points3d(x[1], y[1], z[1], np.ones(z[0].shape),  color=tuple(plt.cm.hsv(dom_theta)[F][:3]), scale_factor=0.5)
	mlab.points3d(x[2], y[2], z[2], np.ones(z[0].shape),  color=tuple(plt.cm.hsv(dom_theta)[F][:3]), scale_factor=0.5)


mlab.plot3d(np.linspace(-10, 10, 100, endpoint=True), np.zeros(100), np.zeros(100), np.ones(100), color=(0,0,0), tube_radius=0.05)
mlab.plot3d( np.zeros(100), np.linspace(-10, 10, 100, endpoint=True),np.zeros(100), np.ones(100), color=(0,0,0), tube_radius=0.05)
mlab.plot3d(np.zeros(100), np.zeros(100),np.linspace(-10, 10, 100, endpoint=True),  np.ones(100), color=(0,0,0), tube_radius=0.05)

xx, yy = np.mgrid[-10.:10.01:0.01, -10.:10.01:0.1]
mlab.surf(xx, yy, np.zeros_like(xx), opacity=0.25, color=(0,0,1)) 
mlab.show()