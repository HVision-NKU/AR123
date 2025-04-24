"""
Blender script to render images of 3D models.

Example usage:
    blender --background --python blender_script.py -- \
    --object_path my_object.glb \
    --output_dir ./output \
    --object_uid_level 1
"""
import os
import sys
import math
from mathutils import Vector, Matrix
from pathlib import Path
import argparse
import random
import numpy as np
import argparse
import json
import time
import multiprocessing
import bpy

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str)
parser.add_argument("--output_dir", type=str, help="Directory where the output images and data will be saved.")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--device", type=str, default='CUDA')
parser.add_argument("--object_uid", type=str, default=None, help="UID for the object (can be set manually or defaults to parent folder names).")
parser.add_argument("--object_uid_level", type=int, default=0, help="Level of the directory to use as object_uid. 1 for direct parent, 2 for grandparent, etc.")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

if args.object_uid is None:
    object_path = args.object_path
    target_dir = object_path
    for _ in range(args.object_uid_level):
        target_dir = os.path.abspath(os.path.join(target_dir, ".."))
    
    args.object_uid = os.path.basename(target_dir)


print('===================', args.engine, '===================')
print(f"Using folder path: {args.object_path}")
print(f"Output directory: {args.output_dir}")
print(f"Object UID: {args.object_uid}")

engine = args.engine
device = args.device

context = bpy.context
scene = context.scene
render = scene.render
cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.sensor_width = 32
fov = 30
cam.data.lens = 0.5 * cam.data.sensor_width / math.tan(0.5 * math.radians(fov))
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
render.engine = engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100
scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True
bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device
bpy.context.scene.cycles.tile_size = 8192
scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_z = True
tree = bpy.context.scene.node_tree
nodes = tree.nodes
links = tree.links
# Add passes for normals
scene.view_layers["ViewLayer"].use_pass_normal = True
# Clear default nodes
for n in nodes:
    nodes.remove(n)

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #


def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = -camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def set_camera_matrix_world(matrix_world):
    camera = bpy.data.objects["Camera"]
    camera.matrix_world = Matrix(matrix_world.tolist())
    return camera

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

# load the glb model
import bpy
import os

def load_object(object_path: str) -> None:
    if not os.path.isfile(object_path):
        raise ValueError(f"Provided path is not a valid file: {object_path}")

    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        print(f"Loaded GLB model: {object_path}")

    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
        print(f"Loaded FBX model: {object_path}")

    elif object_path.endswith(".obj"):
 
        folder_path = os.path.dirname(object_path)
        mtl_path = object_path.replace(".obj", ".mtl")
        
        if not os.path.exists(mtl_path):
            print(f"Warning: .mtl file not found for {object_path}. The object will be loaded without materials.")

        bpy.ops.wm.obj_import(filepath=object_path)
        print(f"Loaded OBJ model: {object_path}")
    
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    #bpy.ops.transform.rotate(value=math.radians(90), orient_axis='X')
    #bpy.ops.transform.rotate(value=math.radians(-45), orient_axis='Y')
    #bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='Z')

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    # Calculate the centroid of the scene's bounding box
    bbox_min, bbox_max = scene_bbox()
    scale_cube = 2 / max(bbox_max - bbox_min)
    centroid = (bbox_min + bbox_max) / 2
    
    # Calculate the maximum distance from the centroid to any vertex in the scene
    max_distance = 0
    for obj in scene_meshes():
        for vertex in obj.data.vertices:
            # Transform vertex coordinate to world space
            world_coord = obj.matrix_world @ vertex.co
            distance = (world_coord - centroid).length
            if distance > max_distance:
                max_distance = distance
    
    # Calculate the scale factor to fit the mesh inside a unit sphere
    scale = 1 / max_distance
    scale *= 0.9
    
    # Scale and translate the root objects
    for obj in scene_root_objects():
        obj.scale *= scale
        obj.location -= centroid * scale
    
    # Apply the transformations
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")

    return scale / scale_cube


def center_looking_at_camera_pose(camera_position: np.ndarray, look_at: np.ndarray = None, up_world: np.ndarray = None):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # by default, looking at the origin and world up is pos-z
    camera_position = camera_position.astype(np.float32)
    if look_at is None:
        look_at = np.array([0, 0, 0], dtype=np.float32)
    if up_world is None:
        up_world = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.tile(np.expand_dims(look_at, 0), (camera_position.shape[0], 1))
    up_world = np.tile(np.expand_dims(up_world, 0), (camera_position.shape[0], 1))

    z_axis = camera_position - look_at
    z_axis = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)
    x_axis = np.cross(up_world, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
    extrinsics = np.stack([x_axis, y_axis, z_axis, camera_position], axis=-1)
    padding = np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (camera_position.shape[0], 1, 1))
    extrinsics = np.concatenate([extrinsics, padding], axis=1)
    return extrinsics


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()

    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    T_world2bcam = -1*R_world2bcam @ location

    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT


def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K


def save_images(object_file: str, output_dir: str, object_uid: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    reset_scene()
    load_object(object_file)

    scale = normalize_scene()
    scale = np.array(scale).astype(float)

    # create an empty object to track
    # empty = bpy.data.objects.new("Empty", None)
    # scene.collection.objects.link(empty)
    # cam_constraint.target = empty

    # lighting
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 0.5
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0

    render_layers = tree.nodes.new('CompositorNodeRLayers')

    #depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    #depth_file_output.label = 'Depth Output'
    #depth_file_output.base_path = os.path.join(args.output_dir, object_uid)
    #depth_file_output.format.file_format = 'PNG'
    #depth_file_output.format.color_mode = "BW"
    
    #MIN_DEPTH, MAX_DEPTH = 0.0, 6.0
    #map = nodes.new(type="CompositorNodeMapValue")
    #map.use_min = True
    #map.min = [MIN_DEPTH]
    #map.use_max = True
    #map.max = [MAX_DEPTH]
    #map.size = [1.0 / (MAX_DEPTH - MIN_DEPTH)]
    #links.new(render_layers.outputs['Depth'], map.inputs[0])
    #links.new(map.outputs[0], depth_file_output.inputs[0])

    # create normal output node
    #scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    #scale_normal.blend_type = 'MULTIPLY'
    #scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    #links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

    #bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    #bias_normal.blend_type = 'ADD'
    #bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    #links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    #normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    #normal_file_output.label = 'Normal Output'
    #normal_file_output.base_path = os.path.join(args.output_dir, object_uid)
    #normal_file_output.format.file_format = 'PNG'
    #links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    #(Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)

    # Wonder3D camera poses

    # cam_distance = random.uniform(1.8, 3.0)
    # azimuth_input = random.uniform(0, 360)
    # elevation_input = random.uniform(-20, 50)
    # azimuths = np.array([0, 30, 90, 150, 210, 270, 330]) + azimuth_input
    # elevations = np.array([elevation_input, 30, -20, 30, -20, 30, -20])

    # # 遍历场景中的所有对象，找到网格对象（即包含几何信息的对象）
    # mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    # # 获取网格对象的包围盒大小
    # object_dimensions = mesh_objects[0].dimensions
    # # 计算相机到物体的距离
    # distance_to_object = object_dimensions.length / (2 * math.tan(bpy.data.objects['Camera'].data.angle / 2))
    # print('distance: ', distance_to_object)

    cam_distance_input = random.uniform(1, 1.3) / np.tan(np.radians(30 / 2))    
    # cam_distance_input = random.uniform(2.0, 5.0)
    cam_distance_target = 1.0 / np.tan(np.radians(30 / 2))
    cam_distance = np.array([cam_distance_input] + [cam_distance_target]*6)
    #azimuth_input = random.uniform(0, 360)
    #elevation_input = random.uniform(-20, 45)
    azimuth_input=0
    elevation_input=20

    azimuths = np.array([0, 30, 90, 150, 210, 270, 330]) + azimuth_input
    elevations = np.array([elevation_input, 20, -10, 20, -10, 20, -10])
    #azimuths = np.array([0]) + azimuth_input
    #elevations = np.array([elevation_input])
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    x = cam_distance * np.cos(elevations) * np.cos(azimuths)
    y = cam_distance * np.cos(elevations) * np.sin(azimuths)
    z = cam_distance * np.sin(elevations)

    cam_locations = np.stack([x, y, z], axis=-1)

    cam_poses = []
    for i in range(7):
        # set camera
        cam_pt = cam_locations[i]
        camera = set_camera_location(cam_pt)
        RT = get_3x4_RT_matrix_from_blender(camera)
        cam_poses.append(RT)

        render_path = os.path.join(output_dir, object_uid, f"{i:03d}.png")
        if os.path.exists(render_path): continue
        scene.render.filepath = os.path.abspath(render_path)
        #depth_file_output.file_slots[0].path = f"{i:03d}_depth#"
        #normal_file_output.file_slots[0].path = f"{i:03d}_normal#"

        bpy.ops.render.render(write_still=True)

        #os.rename(
        #    os.path.join(args.output_dir, object_uid, f'{i:03d}_depth1.png'),
        #    os.path.join(args.output_dir, object_uid, f'{i:03d}_depth.png'))
        #os.rename(
        #    os.path.join(args.output_dir, object_uid, f'{i:03d}_normal1.png'),
        #    os.path.join(args.output_dir, object_uid, f'{i:03d}_normal.png'))

    cam_poses = np.stack(cam_poses, axis=0)

    np.savez_compressed(
        os.path.join(output_dir, object_uid, 'cameras.npz'),
        cam_poses=cam_poses, 
        scale=scale,
    )

if __name__ == "__main__":
    object_path = args.object_path
    output_dir = args.output_dir
    object_uid = args.object_uid

    os.makedirs(output_dir, exist_ok=True)
    save_images(object_path , output_dir, object_uid)