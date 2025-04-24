'''
Using blendertoolbox and bpy to render the bare mesh from a specified view.
https://github.com/HTDerekLiu/BlenderToolbox/blob/master/template.py
'''
import blendertoolbox as bt
import bpy
import os
import math
import numpy as np


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--meshPath", type=str, default='assets/examples/jacket.obj')
parser.add_argument("--distance", type=int, default=10)
parser.add_argument("--fov", type=int, default=30)

# main settings 
parser.add_argument("--scale_value", type=int, default=2)
parser.add_argument("--rot_x", type=float, default=0)       # np.pi/2, -np.pi/2
parser.add_argument("--rot_y", type=float, default=0)
parser.add_argument("--rot_z", type=float, default=0)

args = parser.parse_args()

meshPath = meshPath = args.meshPath
assert os.path.isfile(meshPath)

dirname, filename = os.path.split(meshPath)
outputPath = os.path.join(dirname, f'{filename[:-4]}.png')

## initialize blender
distance = args.distance   
fov = args.fov
azimuth = 0
elevation = 0
imgRes_x = 720 # recommend > 1080 
imgRes_y = 720 # recommend > 1080 
numSamples = 512 # recommend > 200
exposure = 1.5 
use_GPU = True
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure, use_GPU)

## read mesh
location = (0, 0, 0) # (GUI: click mesh > Transform > Location)
rotation = (0, 0, 0) # (GUI: click mesh > Transform > Rotation)
scale = (1.0, 1.0, 1.0) # (GUI: click mesh > Transform > Scale)
mesh = bt.readMesh(meshPath, location, rotation, scale)

## set shading (uncomment one of them)
bpy.ops.object.shade_smooth() # Option1: Gouraud shading
# bpy.ops.object.shade_flat() # Option2: Flat shading
# bt.edgeNormals(mesh, angle = 10) # Option3: Edge normal shading
bpy.data.objects[bpy.data.objects.keys()[0]].rotation_mode = 'YXZ'   


scale_value = args.scale_value
bpy.data.objects[bpy.data.objects.keys()[0]].scale = (args.scale_value, args.scale_value, args.scale_value)
bpy.data.objects[bpy.data.objects.keys()[0]].rotation_euler = (args.rot_y, args.rot_x,  args.rot_z)


## subdivision
bt.subdivision(mesh, level = 1)

###########################################
## Set your material here (see other demo scripts)

# bt.colorObj(RGBA, Hue, Saturation, Value, Bright, Contrast)
# RGBA = (144.0/255, 210.0/255, 236.0/255, 1)
RGBA = (114.0/255, 180.0/255, 236.0/255, 1)
meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
bt.setMat_plastic(mesh, meshColor)

## End material
###########################################

## set invisible plane (shadow catcher)
# bt.invisibleGround(location=(0,0,-1), shadowBrightness=0.9)

## set camera 
azimuth = math.radians(azimuth)
elevation = math.radians(elevation)
fov = math.radians(fov)

camLocation = (distance * math.cos(elevation) * math.cos(azimuth) ,  distance * math.cos(elevation) * math.sin(azimuth), distance * math.sin(elevation))
lookAtLocation = (0,0,0)
focalLength = 0.5 * 36 / math.tan(0.5 * fov)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

## set light
## Option1: Three Point Light System 
# bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')
## Option2: simple sun light
lightAngle = (137, -197, 51) 
strength = 2
shadowSoftness = 0.3
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold (optional but recommended)
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

## save rendering
bt.renderImage(outputPath, cam)
