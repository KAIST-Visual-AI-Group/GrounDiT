# GrounDiT: Grounding Diffusion Transformers via Noisy Patch Transplantation, NeurIPS 2024

![teaser](assets/teaser.png)
[Phillip Y. Lee](https://phillipinseoul.github.io/)\*, [Taehoon Yoon](https://github.com/taehoon-yoon)\*, [Minhyuk Sung](https://mhsung.github.io/) (* equal contribution)

| [**Website**](https://groundit-diffusion.github.io/) | [**Paper**](https://groundit-diffusion.github.io/static/groundit_paper.pdf) | [**arXiv**](https://arxiv.org/abs/2410.20474) |

<br />

## 🚀 Introduction
This repository contains the official implementation of **GrounDiT: Grounding Diffusion Transformers via Noisy Patch Transplantation**. <br><br>
**GrounDiT** is a training-free method for spatial grounding in text-to-image generation, using Diffusion Transformers (DiT) to generate precise, controllable images based on user-specified bounding boxes.
More results can be viewed on our [project page](https://groundit-diffusion.github.io/).

[//]: # (### Abstract)
> We introduce a novel training-free spatial grounding technique for text-to-image
generation using Diffusion Transformers (DiT). Spatial grounding with bounding
boxes has gained attention for its simplicity and versatility, allowing for enhanced
user control in image generation. However, prior training-free approaches often
rely on updating the noisy image during the reverse diffusion process via backprop-
agation from custom loss functions, which frequently struggle to provide precise
control over individual bounding boxes. In this work, we leverage the flexibility of
the Transformer architecture, demonstrating that DiT can generate noisy patches
corresponding to each bounding box, fully encoding the target object and allowing
for fine-grained control over each region. Our approach builds on an intriguing
property of DiT, which we refer to as semantic sharing. Due to semantic sharing,
when a smaller patch is jointly denoised alongside a generatable-size image, the
two become "semantic clones". Each patch is denoised in its own branch of the gen-
eration process and then transplanted into the corresponding region of the original
noisy image at each timestep, resulting in robust spatial grounding for each bound-
ing box. In our experiments on the HRS and DrawBench benchmarks, we achieve
state-of-the-art performance compared to previous training-free spatial grounding
approaches.

## 🛠️ Environment

1. Download [PyTorch with CUDA version 11.8](https://pytorch.org/get-started/locally/). (Any PyTorch version with ≥ `2.0.0` would be fine.)
   
2. Install other dependencies via ```pip install -r requirements.txt```

## 🔥 Inference

ipynb demo is available at ```groundit_demo.ipynb```.

Or you can generate image via following command

```
python main.py 
```

### Additional Arguments

| Argument                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `--save_dir`             | Directory where the results will be saved.                                  |
| `--model_version`        | Model version to use. Options: `512` or `1024`.                             |
| `--input_config_path`    | Path to the input configuration file.                                       |
| `--gpu_id`               | GPU ID to use for inference Default: `0`                                    |
| `--seed`                 | Random seed.                                                                |
| `--num_inference_steps`  | Number of inference steps to perform. Default: `50`                         |
| `--groundit_gamma`       | Apply GrounDiT for the initial γ% steps. Default: `0.5`                     |

<!-- ## Input Data Format

You can find the example of input data format in the ```config.json``` file. 

- `prompt` : Input text for the image generation.

- `phrases` : list of string. Where each string is the phrase describing the desired object to be placed in the box. 
It can be multi-word like *brown bear*. **Each phrase must be presented inside the `prompt`.**

- `bboxes` : list containing the lists of location information of bounding boxes for each phrase. Each list for the phrase can contain multiple list corresponding to multiple bounding boxs per phrase.  The convention for bounding box is `[ul_x, ul_y, lr_x, lr_y]`. Each number should be in `[0, 1]`, where it represents the fraction of corresponding length.
  - `ul_x` : x-coordinate of the upper-left corner of the bounding box.
  - `ul_y` : y-coordinate of the upper-left corner of the bounding box.
  - `lr_x` : x-coordinate of the lower-right corner of the bounding box.
  - `lr_y` : y-coordinate of the lower-right corner of the bounding box.

- `height, width` **or** `aspect_ratio` : Specify either `height` and `width` or just `aspect_ratio`. You can use any aspect ratio you want but too abnormal value would result in implausible image. Recommended range is `[0.25, 4.0]`. If you are specify `height` and `width`, any value wolud be fine but as mentioned in the paper if the value for resolution is far from the **generatable resolution** of PixArt-α, the resulting image would be implausible. For the details of **generatable resolution**, please see the appendix D in our paper. You can consult reasonable resolution values in the `ASPECT_RATIO_512_BIN` or `ASPECT_RATIO_1024_BIN` dictionary, depending on your specified model_version, inside the `/groundit/pipeline_groundit.py` file.  -->


## 📝 Input Data Format  

You can find the example of input data format in the ```config.json``` file.

<details>
<summary>Detailed explanation of the input data format</summary>
<br>

```json
{
    "0": {
        "prompt": "a wide view picture of an antique living room with a chair, table, fireplace, and a bed",
        "phrases": ["chair", "table", "fireplace", "bed"],
        "bboxes": [[[0.0, 0.4, 0.15, 1.0]], [[0.25, 0.6, 0.45, 1.0]], [[0.475, 0.1, 0.65, 0.9]], [[0.7, 0.5, 1.0, 1.0]]],
        "height": 288,
        "width": 896
    }
}
```

### Fields

1. **`prompt`**  
   - Type: `str`  
   - Description: The input text describing the image to be generated.  
   - Example: `"a wide view picture of an antique living room with a chair, table, fireplace, and a bed"`  

2. **`phrases`**  
   - Type: `list[str]`  
   - Description: A list of object descriptions (**phrase**) that you want to position in the image.  
   - **IMPORTANT: Each phrase must be presented inside the `prompt`.**
   - Notes:  
     - Each phrase can contain multiple words (e.g., *brown bear*).  
   - Example: `["chair", "table", "fireplace", "bed"]`  

3. **`bboxes`**  
   - Type: `list[list[list[float]]]`  
   - Description: A list containing bounding box coordinates for each phrase.  
   - **IMPORTANT: The order of bounding boxes list must match the order of `phrases`.**
   - Notes:
     - Each phrase can have multiple bounding boxes.  
     - Bounding boxes follow the format `[ul_x, ul_y, lr_x, lr_y]`, where:  
       - `ul_x`: x-coordinate of the upper-left corner (0 to 1).  
       - `ul_y`: y-coordinate of the upper-left corner (0 to 1).  
       - `lr_x`: x-coordinate of the lower-right corner (0 to 1).  
       - `lr_y`: y-coordinate of the lower-right corner (0 to 1).  
   - Example:  
     ```json
     "bboxes": [
         [[0.0, 0.4, 0.15, 1.0]],    // Bounding box for "chair"
         [[0.25, 0.6, 0.45, 1.0]],   // Bounding box for "table"
         [[0.475, 0.1, 0.65, 0.9]],  // Bounding box for "fireplace"
         [[0.7, 0.5, 1.0, 1.0]]      // Bounding box for "bed"
     ]
     ```  

4. **`height` and `width`**   
   - Type: `int`  
   - Description: The dimensions of the generated image in pixels.  
   - Notes:  
     - Use either `height` and `width` **or** `aspect_ratio`. At least one should be present.
     - Specify both `height` and `width` for exact resolution.  
     - Values that deviate significantly from the [**generatable resolutions**](#guidelines-for-resolution) may result in implausible images.  
   - Example:  
     ```json
     "height": 288,
     "width": 896
     ```

5. **`aspect_ratio`**   
   - Type: `float`  
   - Description: The aspect ratio of the image (width / height).  
   - Notes:  
     - Use either `height` and `width` **or** `aspect_ratio`. At least one should be present.
     - Recommended range: `[0.25, 4.0]`  
     - Extreme values may result in unrealistic images.  

---

### Guidelines for Resolution  

- You can consult reasonable resolution values in the `ASPECT_RATIO_512_BIN` or `ASPECT_RATIO_1024_BIN` dictionaries, depending on your specified `model_version`, inside the `/groundit/pipeline_groundit.py` file.

- For the details of **generatable resolution**, please check **Appendix D** in our [paper](https://groundit-diffusion.github.io/static/groundit_paper.pdf).

</details>

## 🙏 Acknowledgements
This code is heavily based on diffusers library, and the official code for [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) and [R&B](https://github.com/StevenShaw1999/RnB). 

We sincerely thank the authors for open-sourcing their code. 

## 🎓 Citation
```
@inproceedings{lee2024groundit,
  title={GrounDiT: Grounding Diffusion Transformers via Noisy Patch Transplantation},
  author={Lee, Phillip Y. and Yoon, Taehoon and Sung, Minhyuk},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
