import os
import json
import argparse
from functools import partial
from tqdm import tqdm
import torch
from diffusers.utils.torch_utils import randn_tensor

from groundit.pipeline_groundit import ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from groundit.utils import *


def main(args):
    # Set seed and device
    device = torch.device(f"cuda:{args.gpu_id}")
    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Data loading
    with open(args.input_config_path, "r") as f:
        total_data_input = json.load(f)
    num_total_data = len(total_data_input)
    print(f"Will process {num_total_data} data.")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    if args.model_version == "512":
        hw_bin = ASPECT_RATIO_512_BIN
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
    elif args.model_version == "1024":
        # WARNING: 1024 version is not tested due to GPU memory limitation !!!
        hw_bin = ASPECT_RATIO_1024_BIN
        model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
    else:
        raise ValueError(f"Invalid model version: {args.model_version}. Choose either 512 or 1024.")
    
    pipe, tokenizer = load_groundit_model(model_id, device)

    # Generate samples for each data input 
    for idx in tqdm(range(num_total_data), desc="Generating Samples: "):
        # Create a directory and construct the path for saving the samples
        sample_path = os.path.join(args.save_dir, str(idx))
        os.makedirs(sample_path, exist_ok=True)
        img_path = os.path.join(sample_path, "img.png")
        img_with_bbox_path = os.path.join(sample_path, "img_with_bbox.png")

        # Fetch data from the input config
        data = total_data_input[str(idx)]
        prompt = data["prompt"]
        phrases = data["phrases"]
        bboxes = data["bboxes"]

        bbox_list = sanity_check(bboxes, phrases)
        
        # Find the location of phrase indices in the prompt after tokenization, do it for all pharse in phrases list.
        phrases_idx = get_phrases_idx_in_prompt(prompt, phrases, tokenizer)

        # Convert bbox coordinates to pixel & latent & patch space. Get the indices of the patches in the patch space that are covered by the bounding box.
        # Here boundnig box region corresponds to "Local Patch" in the paper. See Figure 2.
        if 'height' in data and 'width' in data:
            original_height, original_width = data['height'], data['width']
            target_height, target_width = pipe.classify_height_width_bin(original_height, original_width, hw_bin)
        elif 'aspect_ratio' in data:
            target_height, target_width = pipe.classify_aspect_ratio_bin(data['aspect_ratio'], hw_bin)
            original_height, original_width = target_height, target_width
        else:
            raise ValueError("Invalid data format. Need to provide either height/width or aspect_ratio.")
            
        latent_height, latent_width = target_height // pipe.vae_scale_factor, target_width // pipe.vae_scale_factor
        all_bbox_coord_in_pixel_space = get_bbox_coord_in_pixel_space(bbox_list, target_height, target_width)
        all_bbox_coord_in_latent_space = get_bbox_coord_in_latent_space(bbox_list, latent_height, latent_width)

        # Get the "Object Image" height and width in pixel space, where "Object Image" is introduced in the paper. See Figure 2.
        object_image_hw_in_pixel_space = get_bbox_region_hw(
            all_bbox_coord_in_pixel_space, 
            hw_bin_classify_func=partial(pipe.classify_height_width_bin, ratios=hw_bin)
        )

        # Main image latent
        latent_shape = (1, 4, latent_height, latent_width)
        latent = randn_tensor(
            latent_shape, generator=generator, 
            device=device, dtype=torch.float16
        ) * pipe.scheduler.init_noise_sigma

        # Generate object images latents
        object_image_latents_list = []

        for object_image_hw_list in object_image_hw_in_pixel_space:
            latents_for_each_phrase = []

            for object_image_hw in object_image_hw_list:
                # Calculate latent shape based on the VAE scale factor
                latent_shape = (
                    1, 4,
                    object_image_hw[0] // pipe.vae_scale_factor,
                    object_image_hw[1] // pipe.vae_scale_factor,
                )
                
                # Generate random tensor for latents
                object_image_latents = (
                    randn_tensor(
                        latent_shape, 
                        generator=generator, 
                        device=device, 
                        dtype=torch.float16
                    ) * pipe.scheduler.init_noise_sigma
                )

                latents_for_each_phrase.append(object_image_latents)

            object_image_latents_list.append(latents_for_each_phrase)
        
        # Generate sample
        images = pipe(
            prompt=prompt, 
            width=original_width, 
            height=original_height, 
            latents=latent, 
            num_inference_steps=args.num_inference_steps,
            # General Arguments
            groundit_gamma=args.groundit_gamma,
            bbox_list=bbox_list, 
            phrases=phrases, 
            phrases_idx=phrases_idx, 
            # Local Update Config
            object_image_latents_list=object_image_latents_list,
            object_image_hw_in_pixel_space=object_image_hw_in_pixel_space,
            all_bbox_coord_in_latent_space=all_bbox_coord_in_latent_space
        )

        # Save the generated samples
        image = images[0][0]
        image.save(img_path)
        draw_box(image, bbox_list, ";".join(phrases), original_height, original_width)
        image.save(img_with_bbox_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_version", choices=["512", "1024"], default="512")
    parser.add_argument("--input_config_path", type=str, default="./config.json")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--groundit_gamma", type=float, default=0.5, help="Apply GrounDiT for initial gamma range of timesteps.")

    args = parser.parse_args()
    main(args)


############################################################################################
#  ________  ________  ________  ___  ___  ________   ________  ___  _________  ___        #
# |\   ____\|\   __  \|\   __  \|\  \|\  \|\   ___  \|\   ___ \|\  \|\___   ___\\  \       #
# \ \  \___|\ \  \|\  \ \  \|\  \ \  \\\  \ \  \\ \  \ \  \_|\ \ \  \|___ \  \_\ \  \      #
#  \ \  \  __\ \   _  _\ \  \\\  \ \  \\\  \ \  \\ \  \ \  \ \\ \ \  \   \ \  \ \ \  \     #
#   \ \  \|\  \ \  \\  \\ \  \\\  \ \  \\\  \ \  \\ \  \ \  \_\\ \ \  \   \ \  \ \ \__\    #
#    \ \_______\ \__\\ _\\ \_______\ \_______\ \__\\ \__\ \_______\ \__\   \ \__\ \|__|    #
#     \|_______|\|__|\|__|\|_______|\|_______|\|__| \|__|\|_______|\|__|    \|__|     ___  #
#                                                                                    |\__\ #
#                                                                                    \|__| #
############################################################################################