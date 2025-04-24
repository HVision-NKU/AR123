import os
import argparse
import torch

import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.cuda.amp import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import EulerAncestralDiscreteScheduler
from ar123.models.nvs.pipeline import Zero123PlusPipeline as DiffusionPipeline

from ar123.utils.train_util import instantiate_from_config
from ar123.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from ar123.utils.mesh_util import save_obj, save_obj_with_mtl
from ar123.utils.infer_util import load_image, pil_to_tensor, save_video


def load_mv_images(mv_img_root, name=None, outputs=[]):
    if name is not None:
        assert os.path.isfile(mv_img_root, f'{name}.png')
        filenames = [f'{name}.png']
    else:
        filenames = [filename for filename in os.listdir(mv_img_root) if filename.endswith('.png')]

    for filename in tqdm(filenames):
        img_path = os.path.join(mv_img_root, filename)
        mv_img = load_image(img_path, no_rembg=True, add_batch_dim=False)           # (3, 960, 640) 
        print('mv_img: ', mv_img.shape)
        images = rearrange(mv_img, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
        outputs.append({'name': os.path.splitext(filename)[0], 'images': images})
    return outputs


def ar123_sampling(cond_img, pipeline, num_inference_steps=75):
    # print('cond_img: ', cond_img.shape, cond_img.device)
    h, w = cond_img.shape[-2:]
    output = []
    for _ in range(1, 4):
        step_image_pil = pipeline(
            cond_img, 
            num_inference_steps=num_inference_steps, 
            output_type='pil', 
            height=320, width=640,
        ).images[0]
        
        step_image = pil_to_tensor(step_image_pil, add_batch_dim=1).to(cond_img.device)           # (1, 3, 320, 640)
        output.append(step_image)
        # print('step_image:', step_image.shape, step_image.device)

        parts = torch.split(step_image, split_size_or_sections=step_image.shape[-1]//2, dim=-1)
        parts = [cond_img] + [F.interpolate(part, size=(h, w), mode='bilinear', align_corners=False) for part in parts]
        cond_img = torch.cat(parts, dim=0)

    output = torch.cat(output, dim=-2)                                        # (1, 3, 960, 640)
    return output


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='configs/ar123_infer.yaml', help='Path to config file.')
parser.add_argument('--input_path', type=str, required=True, help='Path to input image or directory.')
parser.add_argument('--mode', type=str, default='itomv', help='Generation Modes: itomv, mvto3d, ito3d.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.base)
config_name = os.path.basename(args.base).replace('.yaml', '')
lrm_model_config = config.lrm_model_config
nvs_model_config = config.nvs_model_config
infer_config = config.infer_config
IS_FLEXICUBES = True

device = torch.device('cuda')

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')
SPECIFIC_NAME = None


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
os.makedirs(image_path, exist_ok=True)

outputs = []
if args.mode in ['itomv', 'ito3d']:

    # load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        # custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    assert os.path.isfile(infer_config.nvs_path)
    checkpoint = torch.load(infer_config.nvs_path, map_location='cpu')
    unet_state_dict =  checkpoint['unet_state_dict']
    lstm_state_dict =  checkpoint['lstm_state_dict']

    pipeline.unet.load_state_dict(unet_state_dict, strict=True)
    pipeline.global_fusion = instantiate_from_config(nvs_model_config)
    pipeline.global_fusion.load_state_dict(lstm_state_dict, strict=True)
    pipeline.global_fusion.to(device,dtype=torch.float16)
    pipeline = pipeline.to(device)

    for idx, image_file in enumerate(input_files):
        name = os.path.basename(image_file).split('.')[0]
        if len(input_files) == 1:
            SPECIFIC_NAME = name
        print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

        # remove background optionally
        cond_img = load_image(image_file, no_rembg=args.no_rembg, add_batch_dim=True).to(device)    # (1, 3, h, w)
        
        # 推理单张图像
        with torch.no_grad(), autocast(enabled=True, dtype=torch.float16):
            output = ar123_sampling(cond_img, pipeline, num_inference_steps=args.diffusion_steps)         # (1, 3, 960, 640)
        
        # 保存结果
        output_path = os.path.join(image_path, f'{name}.png')
        torchvision.utils.save_image(output, output_path)
        print(f'Results saved to {output_path}')

        outputs.append({
            'name': name, 
            'images': rearrange(output.squeeze(0), 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
        })

    # delete pipeline to save memory
    del pipeline


if args.mode == 'mvto3d':
    outputs = load_mv_images(mv_img_root=image_path, name=SPECIFIC_NAME, outputs=outputs)

###############################################################################
# Stage 2: Reconstruction.
###############################################################################
if args.mode in ['mvto3d', 'ito3d']:
    # make output directories
    mesh_path = os.path.join(args.output_path, config_name, 'meshes')
    os.makedirs(mesh_path, exist_ok=True)
    
    if args.save_video:
        video_path = os.path.join(args.output_path, config_name, 'videos')
        os.makedirs(video_path, exist_ok=True)

    assert len(outputs) > 0, "multi-view images are not existing"

    # load reconstruction model
    print('Loading reconstruction model ...')
    model = instantiate_from_config(lrm_model_config)
    assert os.path.exists(infer_config.lrm_path)
    model_ckpt_path = infer_config.lrm_path

    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1

    for idx, sample in enumerate(outputs):
        name = sample['name']
        print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

        images = sample['images'].unsqueeze(0).to(device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)     # (1, 6, 3, 320, 320)

        if args.view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(device)
            images = images[:, indices]
            input_cameras = input_cameras[:, indices]

        with torch.no_grad():
            # get triplane
            planes = model.forward_planes(images, input_cameras)

            # get mesh
            mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args.export_texmap,
                **infer_config,
            )
            if args.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, mesh_path_idx)
            print(f"Mesh saved to {mesh_path_idx}")

            # get video
            if args.save_video:
                video_path_idx = os.path.join(video_path, f'{name}.mp4')
                render_size = infer_config.render_resolution
                render_cameras = get_render_cameras(
                    batch_size=1, 
                    M=120, 
                    radius=args.distance, 
                    elevation=20.0,
                    is_flexicubes=IS_FLEXICUBES,
                ).to(device)
                
                frames = render_frames(
                    model, 
                    planes, 
                    render_cameras=render_cameras, 
                    render_size=render_size, 
                    chunk_size=chunk_size, 
                    is_flexicubes=IS_FLEXICUBES,
                )

                save_video(
                    frames,
                    video_path_idx,
                    fps=30,
                )
                print(f"Video saved to {video_path_idx}")
