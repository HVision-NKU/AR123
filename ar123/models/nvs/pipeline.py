from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available

from torchvision.transforms import v2


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                # encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler, use_checkpoint=False,sampling_rate =1) -> None:
        super().__init__()
        self.unet = unet
        if use_checkpoint:
            self.unet.enable_gradient_checkpointing()
        self.train_sched = train_sched
        self.val_sched = val_sched
        self.sampling_rate = sampling_rate
        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        **kwargs
    ):
        '''
        cond_lat: (B, 2k-1, dim, h, w)
        prompt_embeds_set: (B, 2k-1, 77, 1024)，空文本与2k-1个条件图片的clip特征的编码，二者直接相加，用于记录参考特征forward unet时的self-att
        encoder_hidden_states: (B, 77, 1024)，空文本与2k-1个条件图片的clip特征的编码，已经设置好, 用于正常forward unet过程中cross-attention
        '''
        cond_lat = cross_attention_kwargs['cond_lat']
        prompt_embeds_set = cross_attention_kwargs.get('prompt_embeds_set', None)
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)

        # 1. 为cond_latent添加同等程度的噪声
        noise = torch.randn_like(cond_lat[:, 0, ...])        # noise = torch.randn_like(cond_lat)
        if self.training:
            # noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            # noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)

            noisy_cond_lat_list = [self.train_sched.scale_model_input( \
                self.train_sched.add_noise(cond_lat[:, idx, ...], noise, timestep), timestep) \
                    for idx in range(cond_lat.shape[1])
            ]
        else:
            # noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            # noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))

            noisy_cond_lat_list = [self.val_sched.scale_model_input( \
                self.val_sched.add_noise(cond_lat[:, idx, ...], noise, timestep.reshape(-1)), timestep.reshape(-1)) \
                    for idx in range(cond_lat.shape[1])
            ]

        # ref_dict = {}
        # self.forward_cond(
        #     noisy_cond_lat, timestep,
        #     encoder_hidden_states, class_labels,
        #     ref_dict, is_cfg_guidance, **kwargs
        # )

        # 2. 以noisy_cond_lat作为sample，forward一遍unet，记录每个层级中self-attention的输入值，作为后续修改Key和Value的参考特征
        ref_dict_list = []
        for idx in range(cond_lat.shape[1]):
            cur_ref_dict = {}
            self.forward_cond(
                noisy_cond_lat_list[idx], timestep,
                encoder_hidden_states if prompt_embeds_set is None else prompt_embeds_set[:, idx, ...], class_labels,
                cur_ref_dict, is_cfg_guidance, **kwargs
            )
            ref_dict_list.append(cur_ref_dict)
        def sample_tensor(tensor, sampling_rate):
            """
            对给定的 [bs, hw, c] 张量进行采样。
            :param tensor: 形状为 [bs, hw, c] 的输入张量
            :param sampling_rate: 采样率 (0, 1]，表示采样后保留的比例
            :return: 采样后的张量
            """
            bs, hw, c = tensor.shape
            
            # 计算采样后的 hw 大小
            sampled_hw = int(hw * sampling_rate)
            
            # 随机选择 sampled_hw 个索引进行采样
            sampled_indices = torch.randperm(hw)[:sampled_hw]
            
            # 根据采样的索引对张量进行采样
            sampled_tensor = tensor[:, sampled_indices, :]
        
            return sampled_tensor
        
        # 3. 更新参考特征
        ref_dict = ref_dict_list[0]
        for key_ in ref_dict.keys():
            value_ = ref_dict[key_]
            other_values = [ref_dict_list[idx][key_] for idx in range(1, cond_lat.shape[1])]
            values = [value_] + other_values    # [bs, hw, c]
            ref_dict[key_] = sample_tensor(torch.cat(values, dim=1),sampling_rate=self.sampling_rate)



        # 4. 正常forward一遍unet
        weight_dtype = self.unet.dtype
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


class DepthControlUNet(torch.nn.Module):
    def __init__(self, unet: RefOnlyNoisedUNet, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, *args, cross_attention_kwargs: dict, **kwargs):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop('control_depth')
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class SuperNet(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        state_dict = OrderedDict((k, state_dict[k]) for k in sorted(state_dict.keys()))
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k
            return key.split('.')[0]

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class Zero123PlusPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor, 
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.global_fusion = None
        self.sampling_rate = 1

    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler,sampling_rate = self.sampling_rate).eval()
        if self.unet.sampling_rate != self.sampling_rate:
            self.unet.sampling_rate = self.sampling_rate
    def add_controlnet(self, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0):
        self.prepare()
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        return SuperNet(OrderedDict([('controlnet', self.unet.controlnet)]))

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image

    @torch.no_grad()
    def __call__(
        self,
        image_=None,
        prompt = "",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        depth_image: Image.Image = None,
        output_type: Optional[str] = "pil",
        width=640,
        height=960,
        num_inference_steps=28,
        return_dict=True,
        **kwargs
    ):
        self.prepare()
        assert self.global_fusion is not None
        if image_ is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        # assert not isinstance(image_, torch.Tensor)
        
        '''
        input - image_: tensor, (2k-1, 3, h, w)
        output - image: tensor, (1, 3, h, 2w)
        '''

        # 1. 依次处理每张条件图片，并记录对应的vae特征和clip特征
        cond_lat_list = []
        global_embeds_list = []
        for idx in range(image_.shape[0]):
            image_pil = v2.functional.to_pil_image(image_[idx])
            image = to_rgb_image(image_pil)
    
            image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
            image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values
            if depth_image is not None and hasattr(self.unet, "controlnet"):
                depth_image = to_rgb_image(depth_image)
                depth_image = self.depth_transforms_multi(depth_image).to(
                    device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
                )
            image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
            image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
            cond_lat = self.encode_condition_image(image)
            
            if guidance_scale > 1:
                negative_lat = self.encode_condition_image(torch.zeros_like(image))
                cond_lat = torch.cat([negative_lat, cond_lat])
            cond_lat_list.append(cond_lat.unsqueeze(1))
            
            encoded = self.vision_encoder(image_2, output_hidden_states=False)
            global_embeds = encoded.image_embeds
            global_embeds = global_embeds.unsqueeze(-2)
            global_embeds_list.append(global_embeds.unsqueeze(1))

        # 2. 将vae特征组合成新的tensor，传入unet，用于修改self-attention的key和value
        cond_lat = torch.cat(cond_lat_list, dim=1)                  # (2/1, 2k-1, c, h, w), 

        # 3. 将clip视觉特征和clip空文本特征融合，传入unet，用于执行cross-attention
        global_embeds = torch.cat(global_embeds_list, dim=1)        # (1, 2k-1, 1, 1024)
        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )
        # ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        # encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        
        # encoder_hidden_states = torch.matmul(ramp.repeat_interleave(image_.shape[0], dim=-1), global_embeds.squeeze(-2))    # (77, 2k-1) @ (1, 2k-1, 1024) => (1, 77, 1024)
        encoder_hidden_states = self.global_fusion(global_embeds, global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1))
        # 4. 将经过特定处理融合的特征作为encoder_hidden_states；按照zero123++方式相加融合的特征，保留了2k-1信息，用于后续记录self-att的输入
        cak = dict(cond_lat=cond_lat)
        
        if hasattr(self.unet, "controlnet"):
            cak['control_depth'] = depth_image
        latents: torch.Tensor = super().__call__(
            None,
            *args,
            cross_attention_kwargs=cak,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            output_type='latent',
            width=width,
            height=height,
            **kwargs
        ).images
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

  
