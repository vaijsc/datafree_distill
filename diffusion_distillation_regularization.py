# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torchvision
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from download import find_model
import copy
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def batch_mse(input, target):
    return torch.mean((input - target)**2, dim=(1,2,3))

def huber_loss(x, y, c = 0.01):
    c = torch.tensor(c, device=x.device)
    return torch.sqrt(batch_mse(x, y) + c**2) - c

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    # if dist.get_rank() == 0:  # real logger
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    # else:  # dummy logger (does nothing)
    #     logger = logging.getLogger(__name__)
    #     logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    batch_size = args.global_batch_size
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        if args.resume_ckpt is None:
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_dir = f"{args.results_dir}/{args.exp}"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            experiment_dir = args.exp
            checkpoint_dir = f"{experiment_dir}/checkpoints" 
            os.makedirs(checkpoint_dir, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Resume writing at directory at {experiment_dir}")
    else:
        logger = create_logger(None)
    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    
    # Note that parameter initialization is done within the DiT constructor
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(str(args.nfe))  # default: 1000 steps, linear noise schedule
    logger.info(f"We will use the following timesteps: {diffusion.timestep_map}")
    logger.info(f"We will use the following noise scale (sqrt(alpha_cumprod)): {diffusion.sqrt_alphas_cumprod}")
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    with torch.no_grad():
        z = torch.randn((4, 4, latent_size, latent_size), device=device)
        y = torch.tensor([207, 360, 387, 88], device=device)
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * len(y), device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = teacher.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = teacher.forward
        sample_list, x0_list = diffusion.ddim_sample_loop(sample_fn, 
                                                z.shape, 
                                                z, 
                                                clip_denoised=False, 
                                                model_kwargs=model_kwargs, 
                                                progress=True, 
                                                device=device)
        fake_image = sample_list
        if using_cfg:
            fake_image, _ = fake_image.chunk(2, dim=0)
            x0_list = [x0.chunk(2, dim=0)[0] for x0 in x0_list]
        fake_image = vae.decode(fake_image / args.scale_factor).sample
        torchvision.utils.save_image(fake_image, os.path.join(experiment_dir, 'image_sample.png'), normalize=True)
        x0_list = [vae.decode(x0 / args.scale_factor).sample for x0 in x0_list]
        x0_list = torch.cat(x0_list, dim=0)
        torchvision.utils.save_image(x0_list, os.path.join(experiment_dir, 'x0_sample.png'), normalize=True, nrow=4)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    if args.resume_ckpt is not None:
        resume_path = os.path.join(experiment_dir, f"checkpoints/{args.resume_ckpt:07d}.pt")
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)["model"]
        model.load_state_dict(checkpoint)
    else:
        start_steps = 0
        
    assistant = copy.deepcopy(model).to(device)
    assistant.eval()
    running_loss = 0
    log_steps = 0
    start_time = time()
    T = torch.tensor([diffusion.num_timesteps-1] * batch_size, device=device)
    logger.info(f"Using {args.nfe} NFEs teacher for training")

    for idx in range(1, args.nfe-1):
        if rank == 0:
            logger.info(f"========== Update time {idx+1} ===========")
        n_iter = args.based_iter
        for train_steps in tqdm(range(1, n_iter + 1)):
            idxs = torch.tensor([idx] * batch_size, device=device) 
            noise = torch.randn((batch_size, 4, latent_size, latent_size), device=device)
            y = torch.randint(0, args.num_classes + 1, (noise.shape[0],), device=device) # I fix code to avoid dropout random during training model
            model_kwargs = dict(y=y)
            opt.zero_grad()
            
            # Be careful with _predict_eps_from_xstart, could create error gap
            with torch.no_grad():
                # get z_{T-delta_t} from zT with jump delta_t using assistant
                out = diffusion.p_mean_variance(model=assistant, x=noise, t=T, model_kwargs=model_kwargs)
                eps = diffusion._predict_eps_from_xstart(noise, T, out["pred_xstart"]) # carefully with this, might have bug
                zt_assistant = diffusion.q_sample(out["pred_xstart"], T-idxs, noise=eps)
                
                # get z_{T-delta_t-skip} from z_{T-delta_t} using teacher
                out = diffusion.p_mean_variance(model=teacher, x=zt_assistant, t=T-idxs, model_kwargs=model_kwargs)
                eps = diffusion._predict_eps_from_xstart(zt_assistant, T-idxs, out["pred_xstart"])
                z_next_t_teacher = diffusion.q_sample(out["pred_xstart"], T-idxs-1, noise=eps)
                
                out = diffusion.p_mean_variance(model=teacher, x=noise, t=T, model_kwargs=model_kwargs)
                eps = diffusion._predict_eps_from_xstart(noise, T, out["pred_xstart"])
                z_T_minus_1_teacher = diffusion.q_sample(out["pred_xstart"], T-1, noise=eps)

                # check the logic here, it differs from initial idea              
                
            # student engage
            t = (diffusion.num_timesteps - 1) * torch.ones(noise.shape[0], dtype=torch.long, device=device)
            out = diffusion.p_mean_variance(model=model, x=noise, t=T, model_kwargs=model_kwargs)
            eps = diffusion._predict_eps_from_xstart(noise, T, out["pred_xstart"])
            z0_student = out["pred_xstart"]
            z0_next_t_student = diffusion.q_sample(z0_student, T-idxs-1, noise=eps)
            # get zt. to train student to next jump
            z_T_minus_1_student = diffusion.q_sample(z0_student, t, noise=eps)
            # z0t_student = out["pred_xstart"]
            
            # 4-8*(t-0.5)**2 : weight that max at the middle
            # weight = lambda t: 4-8*(t/diffusion.num_timesteps - 0.5)**2
            weight = lambda t: 1
            rescale = lambda t: torch.tensor(1./diffusion.sqrt_alphas_cumprod[t.cpu()], device=device)
            # rescale = lambda t: 100
            loss = torch.mean(huber_loss(z0_next_t_student, z_next_t_teacher)) + \
                                        torch.mean(huber_loss(z_T_minus_1_student, z_T_minus_1_teacher))
            
            loss.backward()
            opt.step()
            
            # Log loss values:
            running_loss += loss.item()
            # log the loss function
            if train_steps % args.log_every == 0 and log_steps > 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            # Save model checkpoint:
            if train_steps % (n_iter//2) == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{idx:02d}_{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            # Sample for visualization
            if train_steps % 500 == 0:
                if rank == 0:
                    with torch.no_grad():
                        model.eval()
                        z = torch.randn((4, 4, latent_size, latent_size), device=device)
                        y = torch.tensor([207, 360, 387, 88], device=device)
                        if using_cfg:
                            z = torch.cat([z, z], 0)
                            y_null = torch.tensor([1000] * len(y), device=device)
                            y = torch.cat([y, y_null], 0)
                            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                            sample_fn = model.module.forward_with_cfg
                        else:
                            model_kwargs = dict(y=y)
                            sample_fn = model.module.forward
                        sample_list, x0_list = diffusion.ddim_sample_loop(sample_fn, 
                                                                z.shape, 
                                                                z, 
                                                                clip_denoised=False, 
                                                                model_kwargs=model_kwargs, 
                                                                progress=True, 
                                                                device=device)
                        fake_image = sample_list
                        if using_cfg:
                            fake_image, _ = fake_image.chunk(2, dim=0)
                            x0_list = [x0.chunk(2, dim=0)[0] for x0 in x0_list]
                        fake_image = vae.decode(fake_image / args.scale_factor).sample
                        x0_list = [vae.decode(x0 / args.scale_factor).sample for x0 in x0_list]
                        x0_list = torch.cat(x0_list, dim=0)
                        torchvision.utils.save_image(x0_list, os.path.join(experiment_dir, f'x0_s{idx:02d}_iter{train_steps:07d}.jpg'), normalize=True, nrow=4)
                        torchvision.utils.save_image(fake_image, os.path.join(experiment_dir, f'image_s{idx:02d}_iter{train_steps:07d}.jpg'), normalize=True)
                        model.train()
                    logger.info("Finish sampling")
                dist.barrier()
            log_steps += 1
            
        # Increase skip step
        assistant = copy.deepcopy(model)  
        assistant.eval()
        logger.info("Finishing copy assistant model")
            
            

    
    # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    
    parser.add_argument("--based_iter", type=int, default=1000)
    parser.add_argument("--nfe", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--exp", type=str, default="first", required=True)
    parser.add_argument("--resume_ckpt", type=int, default=None)
    
    parser.add_argument("--scale_factor", type=float, default=0.18215)
    parser.add_argument("--global-batch-size", type=int, default=12)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)