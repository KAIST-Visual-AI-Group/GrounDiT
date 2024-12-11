import torch
import numpy as np
import random
from PIL import Image, ImageDraw
from transformers import T5Tokenizer

from groundit.pipeline_groundit import PixArtAlphaPipeline
from groundit.transformer_2d import Transformer2DModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def draw_box(pil_img, bboxes, phrases, height, width):
    """
    Draws bounding boxes with associated phrases on a PIL image.

    Args:
        pil_img (PIL.Image.Image): The image to draw on.
        bboxes (list of list of list): Bounding boxes, where each box is represented by [ul_x, ul_y, lr_x, lr_y].
            For the bbox convention, see the `sanity_check` function in this file.
        phrases (str): Semicolon-separated phrases corresponding to the bounding boxes.
        height (int): Height of the image.
        width (int): Width of the image.
    """
    draw = ImageDraw.Draw(pil_img)
    
    # Split and clean phrases
    phrases_list = [phrase.strip() for phrase in phrases.split(";")]
    
    # Iterate over bounding boxes and phrases
    for obj_bboxes, phrase in zip(bboxes, phrases_list):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox
            # Scale bounding box coordinates to image dimensions
            scaled_bbox = [x_0 * width, y_0 * height, x_1 * width, y_1 * height]
            
            # Draw the bounding box
            draw.rectangle(scaled_bbox, outline="red", width=5)
            
            # Draw the associated phrase
            draw.text((scaled_bbox[0] + 5, scaled_bbox[1] + 5), phrase, fill=(255, 0, 0))


def image_grid(images, n_rows, n_cols):
    """
    Concatenates a list of PIL images into a grid with the specified number of rows and columns.

    Args:
        images (list of PIL.Image.Image): The list of images to arrange in a grid.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.

    Returns:
        PIL.Image.Image: The concatenated grid image.
    """
    if len(images) != n_rows * n_cols:
        raise ValueError("Number of images does not match the grid dimensions (n_rows * n_cols).")

    # Get the dimensions of the individual images (assuming all images are the same size)
    width, height = images[0].size

    # Create a blank canvas for the grid
    grid_width = n_cols * width
    grid_height = n_rows * height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste images onto the grid canvas
    for idx, image in enumerate(images):
        row, col = divmod(idx, n_cols)
        x_offset = col * width
        y_offset = row * height
        grid_image.paste(image, (x_offset, y_offset))

    return grid_image


def load_groundit_model(model_id, device):
    """
    Loads the GroundIt model pipeline with specified configurations.

    Args:
        model_id (str): Identifier for the pretrained model.
        device (torch.device): Device to load the model onto.

    Returns:
        tuple: The configured pipeline and tokenizer.
    """
    # Load pipeline and transformer
    pipe = PixArtAlphaPipeline.from_pretrained(model_id, transformer=None, torch_dtype=torch.float16)
    transformer = Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    
    # Attach transformer to the pipeline
    pipe.transformer = transformer
    pipe.to(device)

    # Set evaluation mode for key components
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.transformer.eval()

    # Disable gradient computation for components
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    for param in pipe.transformer.parameters():
        param.requires_grad = False

    # Move weights to device and convert to half precision
    pipe.smth_3.weight = pipe.smth_3.weight.to(device).half()
    pipe.sobel_conv_x = pipe.sobel_conv_x.to(device).half()
    pipe.sobel_conv_y = pipe.sobel_conv_y.to(device).half()

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    
    return pipe, tokenizer


#################################################
#                                               #
#       Prompt & Phrases Pre-processing         #
#                                               #
#################################################

def text_preprocessing(text):
    """
    Preprocesses input text by converting to lowercase and stripping whitespace.

    Args:
        text (str, tuple, or list): The text or collection of text items to preprocess.

    Returns:
        list: A list of preprocessed text items.
    """
    if not isinstance(text, (tuple, list)):
        text = [text]

    def process(single_text):
        return single_text.lower().strip()

    return [process(t) for t in text]


def sanity_check(bbox_list, phrases):
    """
    Perform a sanity check on bounding boxes to ensure validity.

    Bounding box convention: [ul_x, ul_y, lr_x, lr_y].
    - ul_x: x-coordinate of the upper-left corner of the bounding box.
    - ul_y: y-coordinate of the upper-left corner of the bounding box.
    - lr_x: x-coordinate of the lower-right corner of the bounding box.
    - lr_y: y-coordinate of the lower-right corner of the bounding box.

    If any bounding box has ul_x >= lr_x or ul_y >= lr_y, an error is raised.

    Args:
        bbox_list (list of list of list): List of bounding boxes for each object.
        phrases (list): List of phrases corresponding to bounding boxes.

    Returns:
        list: A new list of validated and adjusted bounding boxes.

    Raises:
        AssertionError: If bbox_list and phrases lengths do not match.
        ValueError: If any bounding box is invalid.
    """
    assert len(bbox_list) == len(phrases), "bbox_list and phrases should have the same length"

    bbox_list_new = []

    for obj_bboxes in bbox_list:
        obj_bbox_new = []
        for obj_bbox in obj_bboxes:
            # Check if the bounding box coordinates are valid
            if obj_bbox[0] >= obj_bbox[2] or obj_bbox[1] >= obj_bbox[3]:
                raise ValueError(
                    "Bounding box is not valid! x0 should be less than x1 and y0 should be less than y1. (x0, y0, x1, y1): {}".format(obj_bbox)
                )

            ul_x, ul_y = obj_bbox[0], obj_bbox[1]
            lr_x, lr_y = obj_bbox[2], obj_bbox[3]

            # Adjust bounding box coordinates if necessary
            if lr_x >= 1.0:
                lr_x = 1.0 - 1e-9
            if lr_y >= 1.0:
                lr_y = 1.0 - 1e-9

            obj_bbox_new.append([ul_x, ul_y, lr_x, lr_y])

        bbox_list_new.append(obj_bbox_new)

    return bbox_list_new


def get_phrases_idx_in_prompt(prompt, phrases, tokenizer):
    """
    Find the location of phrase indices in the prompt after tokenization.

    Each phrase in the phrases list is located in the tokenized prompt. A single phrase can
    appear multiple times in the prompt, and each phrase can contain multiple words.

    Args:
        prompt (str): The input prompt text.
        phrases (list of str): List of phrases to locate in the prompt.
        tokenizer: The tokenizer to use for tokenizing the prompt and phrases.

    Returns:
        list of list of int: A list of lists where each inner list contains the indices of the
        corresponding phrase in the tokenized prompt.

    Example:
        prompt = "bird is flying above the brown tree while bird is below the sun."
        phrases = ["bird", "brown tree", "sun"]
        # Output: [[0, 8], [5, 6], [12]]
    """
    special_tokens = ["</s>", "<pad>"]
    prompt_preprocessed = text_preprocessing(prompt)

    # Tokenize the prompt
    prompt_input_ids = tokenizer(
        prompt_preprocessed,
        padding="max_length",
        max_length=120,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    ).input_ids[0]
    
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_input_ids)
    all_phrases_idx = []

    for phrase in phrases:
        phrase_idx = []
        phrase_preprocessed = text_preprocessing(phrase)
        
        # Tokenize the phrase
        phrase_input_ids = tokenizer(
            phrase_preprocessed,
            padding="max_length",
            max_length=120,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids[0]
        
        phrase_tokens = tokenizer.convert_ids_to_tokens(phrase_input_ids)
        phrase_tokens_filtered = [token for token in phrase_tokens if token not in special_tokens]

        # Locate the phrase in the prompt tokens
        for idx in range(len(prompt_tokens) - len(phrase_tokens_filtered) + 1):
            if all(prompt_tokens[idx + i] == phrase_tokens_filtered[i] for i in range(len(phrase_tokens_filtered))):
                phrase_idx.extend(range(idx, idx + len(phrase_tokens_filtered)))

        all_phrases_idx.append(phrase_idx)

    return all_phrases_idx


#################################################
#                                               #
#         Bounding Box Pre-processing           #
#                                               #
#################################################

def get_bbox_coord_in_pixel_space(bbox_list, height, width):
    """
    Get bounding boxes' upper left and lower right coordinates in pixel space.

    Space conversion:
        Pixel Space ---> Latent Space ---> Patch Space

    Args:
        bbox_list (list): List of bounding boxes for objects.
        height (int): Height of the pixel space.
        width (int): Width of the pixel space.

    Returns:
        list: Bounding box coordinates in pixel space for each bounding box.
    """
    all_bbox_coord_in_pixel_space = []

    for obj_bboxes in bbox_list:
        bbox_coord_in_pixel_space = []

        for bbox in obj_bboxes:
            # Convert bounding box coordinates to pixel space
            ul_x_pixel = int(bbox[0] * width)
            ul_y_pixel = int(bbox[1] * height)
            lr_x_pixel = int(bbox[2] * width)
            lr_y_pixel = int(bbox[3] * height)

            # Adjust for boundary conditions
            if lr_x_pixel == width:
                lr_x_pixel -= 1
            if lr_y_pixel == height:
                lr_y_pixel -= 1

            bbox_coord_in_pixel_space.append([ul_x_pixel, ul_y_pixel, lr_x_pixel, lr_y_pixel])

        all_bbox_coord_in_pixel_space.append(bbox_coord_in_pixel_space)

    return all_bbox_coord_in_pixel_space


def get_bbox_coord_in_latent_space(bbox_list, latent_height, latent_width):
    """
    Get bounding boxes' upper left and lower right coordinates in latent space.

    Space conversion:
        Pixel Space ---> Latent Space ---> Patch Space

    Args:
        bbox_list (list): List of bounding boxes for objects.
        latent_height (int): Height of the latent space.
        latent_width (int): Width of the latent space.

    Returns:
        list: Bounding box coordinates in the latent space for each bounding box.
    """
    all_bbox_coord_in_latent_space = []

    for obj_bboxes in bbox_list:
        bbox_coord_in_latent_space = []

        for bbox in obj_bboxes:
            # Convert bounding box coordinates to latent space
            ul_x_latent = int(bbox[0] * latent_width)
            ul_y_latent = int(bbox[1] * latent_height)
            lr_x_latent = int(bbox[2] * latent_width)
            lr_y_latent = int(bbox[3] * latent_height)

            # Adjust for boundary conditions
            if lr_x_latent == latent_width:
                lr_x_latent -= 1
            if lr_y_latent == latent_height:
                lr_y_latent -= 1

            # Ensure bounding box dimensions are odd
            if (lr_x_latent - ul_x_latent) % 2 == 0:
                lr_x_latent += 1
                if lr_x_latent == latent_width:
                    lr_x_latent -= 2

            if (lr_y_latent - ul_y_latent) % 2 == 0:
                lr_y_latent += 1
                if lr_y_latent == latent_height:
                    lr_y_latent -= 2

            bbox_coord_in_latent_space.append([ul_x_latent, ul_y_latent, lr_x_latent, lr_y_latent])

        all_bbox_coord_in_latent_space.append(bbox_coord_in_latent_space)

    return all_bbox_coord_in_latent_space


# map_bbox_to_patch_space is currently not used in the codebase !!! Left here in case.
def map_bbox_to_patch_space(bbox_list, latent_height, latent_width, patch_size=2):
    """
    Map bounding boxes to patch space and get the indices of patches covered by the bounding boxes.

    Space conversion:
        Pixel Space ---> Latent Space ---> Patch Space
    
    Returns:
        all_bbox_idx_in_patch_space: Indices of patches in patch space covered by each bounding box. 
            Each bounding box's indices are represented as a list of 1D indices (flattened).
        all_bbox_coord_in_patch_space: Bounding box coordinates in patch space for each bounding box.
    """
    all_bbox_idx_in_patch_space = []
    all_bbox_coord_in_patch_space = []

    num_patch_in_height = latent_height // patch_size  # Number of patches in height direction after patchification
    num_patch_in_width = latent_width // patch_size    # Number of patches in width direction after patchification

    for obj_bboxes in bbox_list:
        bbox_coord_in_patch_space = []
        bbox_idx_in_patch_space = []

        for bbox in obj_bboxes:
            # Calculate upper-left and lower-right coordinates of bounding box in patch space
            ul_x_patch = int(bbox[0] * num_patch_in_width)
            ul_y_patch = int(bbox[1] * num_patch_in_height)
            lr_x_patch = int(bbox[2] * num_patch_in_width)
            lr_y_patch = int(bbox[3] * num_patch_in_height)

            # Adjust for edge cases where bounding box extends to the edge of the patch grid
            if lr_x_patch == num_patch_in_width:
                lr_x_patch -= 1
            if lr_y_patch == num_patch_in_height:
                lr_y_patch -= 1

            # Store coordinates in patch space
            bbox_coord_in_patch_space.append([ul_x_patch, ul_y_patch, lr_x_patch, lr_y_patch])

            # Calculate indices of patches covered by the bounding box
            patch_indices = []
            row_indices = torch.arange(
                num_patch_in_width * ul_y_patch + ul_x_patch, 
                num_patch_in_width * ul_y_patch + lr_x_patch + 1
            )
            for h in range(lr_y_patch - ul_y_patch + 1):
                patch_indices.append(row_indices + h * num_patch_in_width)

            # Flatten and store patch indices
            bbox_idx_in_patch_space.append(torch.cat(patch_indices))

        # Append results for current object
        all_bbox_coord_in_patch_space.append(bbox_coord_in_patch_space)
        all_bbox_idx_in_patch_space.append(bbox_idx_in_patch_space)

    return all_bbox_idx_in_patch_space, all_bbox_coord_in_patch_space


def get_bbox_region_hw(bbox_list, hw_bin_classify_func=None):
    """
    Calculate the height and width of the bounding box region for each bounding box.

    Args:
        bbox_list (list): A list of bounding boxes grouped by objects.
        hw_bin_classify_func (callable, optional): A function to classify or modify the 
            height and width values. Defaults to None.

    Returns:
        list: A list containing the height and width for each bounding box, grouped by objects.
    """
    all_bbox_region_hw = []

    for obj_bboxes in bbox_list:
        bbox_region_hw = []

        for bbox in obj_bboxes:
            bbox_region_height = bbox[3] - bbox[1] + 1
            bbox_region_width = bbox[2] - bbox[0] + 1

            # Apply height/width classification function if provided
            if hw_bin_classify_func is not None:
                bbox_region_height, bbox_region_width = hw_bin_classify_func(
                    bbox_region_height, bbox_region_width
                )
            
            bbox_region_hw.append((bbox_region_height, bbox_region_width))

        all_bbox_region_hw.append(bbox_region_hw)

    return all_bbox_region_hw
