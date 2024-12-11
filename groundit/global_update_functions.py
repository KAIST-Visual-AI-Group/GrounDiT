import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Originally from https://github.com/StevenShaw1999/RnB
class GaussianSmoothing(nn.Module):
    """
    Apply Gaussian smoothing on a 1D, 2D, or 3D tensor. Filtering is performed separately for each channel
    in the input using depthwise convolution.

    Args:
        channels (int): Number of channels in the input tensors. The output will have the same number of channels.
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or sequence): Standard deviation of the Gaussian kernel.
        dim (int, optional): Number of dimensions of the data. Default is 2 (spatial).
    """

    def __init__(self, channels: int = 1, kernel_size: int = 3, sigma: float = 0.5, dim: int = 2):
        super().__init__()

        # Ensure kernel_size and sigma are lists of length `dim`
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # Create the Gaussian kernel
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (1 / (std * math.sqrt(2 * math.pi))) * torch.exp(
                -((mgrid - mean) ** 2) / (2 * std ** 2)
            )

        # Normalize kernel to ensure the sum equals 1
        kernel = kernel / torch.sum(kernel)

        # Reshape kernel for depthwise convolution
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        # Assign the appropriate convolution function based on the dimension
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1, 2, and 3 dimensions are supported. Received {dim}.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the Gaussian smoothing to the input tensor.

        Args:
            input (torch.Tensor): The input tensor to apply smoothing on.

        Returns:
            torch.Tensor: The smoothed tensor.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
    
# Originally from https://github.com/StevenShaw1999/RnB
def edge_loss(attn_map, mask, iou, pipe):
    """
    Compute edge loss.
    """
    mask_clone = mask.clone()[1:-1, 1:-1]

    attn_map_clone = attn_map.unsqueeze(0).unsqueeze(0)        
    attn_map_clone = attn_map_clone / (attn_map_clone.max().detach() + 1e-4)
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode='reflect')
    
    # Smoothing
    attn_map_clone = pipe.smth_3(attn_map_clone)

    sobel_output_x = pipe.sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_output_y = pipe.sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_sum = torch.sqrt(sobel_output_y ** 2  + sobel_output_x ** 2)

    loss_ = 1 - (sobel_sum * mask_clone).sum() / (sobel_sum.sum() + 1e-4) * (1 - iou)
    
    return loss_

# Originally from https://github.com/StevenShaw1999/RnB
def aggregated_grounding_loss(attn_maps, bbox_list, object_positions, latent_h, latent_w, pipe):
    '''
    Computes aggregated groundiung loss used in Global Update.
    '''
    loss = 0
    object_number = len(bbox_list)
    device = pipe._execution_device

    if object_number == 0:
        return torch.tensor(0).float().to(device) if torch.cuda.is_available() else torch.tensor(0).float()

    # Get the attention maps
    attn_all = attn_maps[1]
    attn_edge = attn_maps[1] 
    attn_edge = attn_edge.reshape(latent_h, latent_w, -1)
    assert attn_all.shape[0] == latent_h * latent_w, f"attn_all.shape[0]: {attn_all.shape[0]}, height * width: {latent_h * latent_w}"
    
    # Reshape attn_all to [height, width, num_tokens]
    attn_all = attn_all.reshape(latent_h, latent_w, -1)
    obj_loss = 0

    for obj_idx in range(object_number):
        for num, obj_pos in enumerate(object_positions[obj_idx]):
            if num == 0:
                att_map_obj_raw = attn_all[:, :, obj_pos]
                att_map_edge = attn_edge[:, :, obj_pos]
            else:
                att_map_obj_raw = att_map_obj_raw + attn_all[:, :, obj_pos]
                att_map_edge = att_map_edge + attn_edge[:, :, obj_pos]

        attn_norm = (att_map_obj_raw - att_map_obj_raw.min()) / (att_map_obj_raw.max() - att_map_obj_raw.min() + 1e-4)

        # Init mask
        mask = torch.zeros(size = (latent_h, latent_w)).to(device) 
        mask_clone = mask.clone()

        for obj_box in bbox_list[obj_idx]:
            x_min = int(obj_box[0] * latent_w)
            y_min = int(obj_box[1] * latent_h)
            x_max = int(obj_box[2] * latent_w)
            y_max = int(obj_box[3] * latent_h)

            # Apply mask for the object
            mask[y_min: y_max, x_min: x_max] = 1
        
        # Background region
        mask_none_cls = (1 - mask)

        # Set threshold
        if mask_none_cls.sum() != 0:
            threshold = (attn_norm * mask).sum() / mask.sum() / 5 * 2
            threshold = threshold + ((attn_norm * mask_none_cls).sum() / mask_none_cls.sum() / 5 * 3)
        else:
            threshold = 0

        thres_image = attn_norm.gt(threshold) * 1.0
        noise_image = F.sigmoid(20 * (attn_norm - threshold))

        rows, cols = torch.where(thres_image > 0.3)
        if len(rows) == 0 or len(cols) == 0:
            continue
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()

        mask_aug = mask_clone
        mask_aug[y1: y2, x1: x2] = 1    
        mask_aug_in = mask_aug * mask 
        iou = (mask_aug * mask).sum() / torch.max(mask_aug, mask).sum()

        if iou < 0.85:
            this_cls_diff_aug_1 = (mask_aug - attn_norm).detach() + attn_norm
            this_cls_diff_aug_in_1 = (mask_aug_in - attn_norm).detach() + attn_norm
            
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum() * (1 / this_cls_diff_aug_1.sum().detach())
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum().detach() * (1 / this_cls_diff_aug_1.sum())
            if object_number > 1 and obj_idx > -1:
                if (att_map_obj_raw * mask).max() < (att_map_obj_raw * (1 - mask)).max():
                    obj_loss += edge_loss(att_map_edge, mask, iou, pipe) * 1 

            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum() * (1 / noise_image.sum().detach())) * 0.5
            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum().detach() * (1 / noise_image.sum())) * 0.5
    
    loss = loss + (obj_loss / object_number)
    
    return loss

# ================================================================================ #

def GlobalUpdateFunc(pipe, latents, timestep, timestep_idx, denoiser_args, loss_args):
    """
    Update latents during the denoising process with grounding loss.

    Args:
        pipe: The pipeline object with scheduling and model functions.
        latents: The latent tensor to update.
        timestep: Current timestep in the denoising schedule.
        timestep_idx: Index of the current timestep.
        denoiser_args: Dictionary containing denoiser parameters.
        loss_args: Dictionary containing loss-related parameters.

    Returns:
        Tuple: Updated latents and the computed loss.
    """
    # Clone and prepare latents
    latents = latents.clone().detach().float().requires_grad_(True)

    # Scale latent model input
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    # Process timestep for transformer
    timestep = pipe._timestep_process(timestep, latent_model_input.device, latent_model_input.shape[0])

    # Forward pass through the transformer to get attention maps
    _, attn_maps, _ = pipe.transformer(
        latent_model_input.half(),
        encoder_hidden_states=denoiser_args["prompt_embeds"],
        encoder_attention_mask=denoiser_args["prompt_attention_mask"],
        timestep=timestep,
        added_cond_kwargs=denoiser_args["added_cond_kwargs"],
        return_dict=False,
        return_attn_maps=True,  # GrounDiT flag
        is_object_branch=False,  # GrounDiT flag
    )

    # Calculate the grounding loss
    loss = aggregated_grounding_loss(
        attn_maps,
        bbox_list=loss_args["bbox_list"],
        object_positions=loss_args["phrases_idx"],
        latent_h=loss_args["height"] // (pipe.vae_scale_factor * 2),
        latent_w=loss_args["width"] // (pipe.vae_scale_factor * 2),
        pipe=pipe,
    )
    loss = loss * loss_args["loss_scale"]

    # Update latents based on gradient if loss is non-zero
    if loss != 0:
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
        grad_cond = torch.nan_to_num(grad_cond)
        latents = latents - grad_cond * loss_args["gradient_weight"]
    elif timestep_idx < 5:
        loss = 10

    # Cleanup
    del attn_maps
    torch.cuda.empty_cache()

    return latents.half(), loss