import torch

@torch.no_grad()
def MainBranch(prompt_args, misc_args, latents, pipe):
    """
    Main branch for processing latents with a noise prediction model and optional guidance.

    Args:
        prompt_args (dict): Contains prompt-related arguments such as embeddings and attention masks.
        misc_args (dict): Miscellaneous arguments such as guidance scale, timesteps, and configuration.
        latents (torch.Tensor): Latent tensor to process.
        pipe: The pipeline object containing the scheduler and transformer.

    Returns:
        torch.Tensor: Updated latents after processing.
    """
    # Extract arguments from dictionaries
    prompt_embeds = prompt_args["prompt_embeds"]
    prompt_attention_mask = prompt_args["prompt_attention_mask"]
    added_cond_kwargs = prompt_args["added_cond_kwargs"]

    do_cfg = misc_args["do_cfg"]
    guidance_scale = misc_args["guidance_scale"]
    extra_step_kwargs = misc_args["extra_step_kwargs"]
    t = misc_args["t"]
    latent_channels = misc_args["latent_channels"]

    # Prepare latent model input
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Process the current timestep
    current_timestep = pipe._timestep_process(
        t, 
        latent_model_input.device, 
        latent_model_input.shape[0]
    )

    # Predict noise model output
    noise_pred, _, _ = pipe.transformer(
        latent_model_input,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=current_timestep,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
        return_attn_maps=False,  # GrounDiT flag
        is_object_branch=False,  # GrounDiT flag
    )

    # Perform classifier-free guidance (CFG)
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Adjust noise prediction for learned sigma
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        noise_pred = noise_pred.chunk(2, dim=1)[0]

    # Compute previous latents (x_t -> x_t-1)
    latents = pipe.scheduler.step(
        noise_pred, t, latents, 
        **extra_step_kwargs, 
        return_dict=False
    )[0]

    return latents

@torch.no_grad()
def ObjectBranch(object_image_args, local_patch_args, phrase_args, misc_args, latents, pipe):
    # Get all the arguments #####################################################################
    object_image_latents_list = object_image_args["object_image_latents_list"]
    obj_img_scheduler_list = object_image_args["obj_img_scheduler_list"]
    obj_img_added_cond_kwrags_list = object_image_args["obj_img_added_cond_kwrags_list"]

    all_bbox_coord_in_latent_space = local_patch_args["all_bbox_coord_in_latent_space"]
    local_patch_scheduler_list = local_patch_args["local_patch_scheduler_list"]

    phrase_embeds_list = phrase_args["phrase_embeds_list"]
    phrase_attention_mask_list = phrase_args["phrase_attention_mask_list"]

    do_cfg = misc_args["do_cfg"]
    guidance_scale = misc_args["guidance_scale"]
    extra_step_kwargs = misc_args["extra_step_kwargs"]
    t = misc_args["t"]
    latent_channels = misc_args["latent_channels"]
    ################################################################################################

    local_patch_latent_list = dict()
    zips = enumerate(zip(object_image_latents_list, obj_img_scheduler_list, obj_img_added_cond_kwrags_list, all_bbox_coord_in_latent_space, 
                            local_patch_scheduler_list, phrase_embeds_list, phrase_attention_mask_list)) 
    
    for phrase_idx, zips_item in zips:
        phrase_embeds = zips_item[5]
        phrase_attention_mask = zips_item[6]
        zipss = enumerate(zip(zips_item[0], zips_item[1], zips_item[2], zips_item[3], zips_item[4]))
        for inner_idx, zipss_item in zipss:
            object_image_latent = zipss_item[0]
            obj_img_scheduler = zipss_item[1]
            added_cond_kwargs_obj = zipss_item[2]
            local_patch_coord = zipss_item[3]
            local_patch_scheduler = zipss_item[4]

            latent_obj_input = torch.cat([object_image_latent] * 2) if do_cfg else object_image_latent
            latent_obj_input = obj_img_scheduler.scale_model_input(latent_obj_input, t)

            local_patch_ul_x, local_patch_ul_y = local_patch_coord[0], local_patch_coord[1]
            local_patch_lr_x, local_patch_lr_y = local_patch_coord[2] + 1, local_patch_coord[3] + 1

            local_patch_latent = latents[:, :, local_patch_ul_y:local_patch_lr_y, local_patch_ul_x:local_patch_lr_x].clone()
            local_patch_latent_input = torch.cat([local_patch_latent] * 2) if do_cfg else local_patch_latent
            local_patch_latent_input = local_patch_scheduler.scale_model_input(local_patch_latent_input, t)

            current_timestep = pipe._timestep_process(t, latent_obj_input.device, latent_obj_input.shape[0])

            noise_pred_obj_img, _, noise_pred_local_patch = pipe.transformer(
                latent_obj_input,
                encoder_hidden_states=phrase_embeds,
                encoder_attention_mask=phrase_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs_obj,
                return_dict=False,
                # >>> GrounDiT
                return_attn_maps = False,
                is_object_branch = True,
                local_patch_latent_input = local_patch_latent_input,
                # <<< GrounDiT
            )

            # perform guidance
            if do_cfg:
                # perform guidance on Object Image
                noise_pred_obj_img_uncond, noise_pred_obj_img_text = noise_pred_obj_img.chunk(2)
                noise_pred_obj_img = noise_pred_obj_img_uncond + guidance_scale * (noise_pred_obj_img_text - noise_pred_obj_img_uncond)
                # perform guidance on Local Patch
                noise_pred_local_patch_uncond, noise_pred_local_patch_text = noise_pred_local_patch.chunk(2)
                noise_pred_local_patch = noise_pred_local_patch_uncond + guidance_scale * (noise_pred_local_patch_text - noise_pred_local_patch_uncond)
            # learned sigma
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                noise_pred_obj_img = noise_pred_obj_img.chunk(2, dim=1)[0]
                noise_pred_local_patch = noise_pred_local_patch.chunk(2, dim=1)[0]
            else:
                noise_pred_obj_img = noise_pred_obj_img
                noise_pred_local_patch = noise_pred_local_patch
            # compute previous less noisy image for Object Image and Local Patch
            object_image_latents_list[phrase_idx][inner_idx] = obj_img_scheduler.step(noise_pred_obj_img, t, object_image_latent, **extra_step_kwargs, return_dict=False)[0]
            local_patch_latent_list[f'{phrase_idx}_{inner_idx}'] = local_patch_scheduler.step(noise_pred_local_patch, t, local_patch_latent, **extra_step_kwargs, return_dict = False)[0].clone()
    return local_patch_latent_list

@torch.no_grad()
def NoisyPatchTransplantation(latents, local_patch_latent_list, all_bbox_coord_in_latent_space):
    """
    Transplant noisy local patch latents into the global noisy image.

    Args:
        latents (torch.Tensor): The global noisy image to update.
        local_patch_latent_list (dict): A dictionary mapping patch indices to local patch latents.
        all_bbox_coord_in_latent_space (list): Bounding box coordinates in latent space, organized by phrase index.

    Returns:
        None: The input `latents` tensor is updated in place.
    """
    for phrase_idx, bbox_coords_in_latent_space in enumerate(all_bbox_coord_in_latent_space):
        for inner_idx, local_patch_coord in enumerate(bbox_coords_in_latent_space):
            # Extract bounding box coordinates
            ul_x, ul_y = local_patch_coord[0], local_patch_coord[1]
            lr_x, lr_y = local_patch_coord[2] + 1, local_patch_coord[3] + 1

            # Get latents for the current bounding box
            bbox_latents_from_mfd = local_patch_latent_list[f'{phrase_idx}_{inner_idx}']
            bbox_latents_from_main_branch = latents[:, :, ul_y:lr_y, ul_x:lr_x].clone()

            # Combine latents from main branch and local patch
            mfd_weight = 1.0
            bbox_latents_updated = (
                bbox_latents_from_mfd * mfd_weight 
                + bbox_latents_from_main_branch * (1 - mfd_weight)
            )

            # Update global latents with the combined result
            latents[:, :, ul_y:lr_y, ul_x:lr_x] = bbox_latents_updated