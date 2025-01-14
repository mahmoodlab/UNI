# UNI 

## Towards a General-Purpose Foundation Model for Computational Pathology
*Nature Medicine* <img src=".github/uni.jpg" width="300px" align="right" />

[Journal Link](https://www.nature.com/articles/s41591-024-02857-3) | [Open Access Read Link](https://rdcu.be/dBMgh) | [Download Models](#model-weights) | [Download Pre-extracted Embeddings](#pre-extracted-embeddings) | [Cite](#reference) 

### Updates
- **01/14/2025: UNI 2 model weights, benchmark results and pre-extracted embeddings are released.**
- 03/19/2024: UNI is published! Model weights and initial benchmark results are released.

UNI 2 was trained on over 200 million pathology H&E and IHC images sampled from 350+ thousand diverse whole slide images.

Unfamiliar with UNI? Please refer to the original README ([here](./README_old.md)) for more details or refer to the accompanying Nature Medicine study ([here](https://www.nature.com/articles/s41591-024-02857-3)).


## Model weights
| Model Name    | Release Date | Model Architecture | Download Link            |
|---------------------|--------------|---------------------|-------------------------------------------------------------|
| UNI2-h      |   01/2025        | ViT-h/14-reg8               | [HF Link](https://huggingface.co/MahmoodLab/UNI2-h) |
| UNI          |   03/2024        | ViT-l/16                 | [HF Link](https://huggingface.co/MahmoodLab/uni)  |


## Pre-extracted Embeddings
To facilitate downstream tasks, we provide pre-extracted embeddings for the UNI 2 model (UNI2-h) for TCGA, CPTAC and PANDA, which can be downloaded [here](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features).

## Benchmarking UNI 2

### ROI Benchmarks
<table>
  <thead>
    <tr>
      <th>Model name</th>
      <th>Pretraining</th>
      <th>Model size</th>
      <th>HEST (Regression, Public)</th>
      <th>CRC-100K-Raw (9 classes, Public)</th>
      <th>TCGA Uniform Tumor (32 classes, Public)</th>
      <th>C17-WILDS (2 classes, Public)</th>
      <th>Kather MSI （2 classes, Public)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UNI</td>
      <td>Vision</td>
      <td>ViT-l/16</td>
      <td>0.386</td>
      <td>0.925</td>
      <td>0.595</td>
      <td>0.972</td>
      <td>0.679</td>
    </tr>
    <tr>
      <td colspan="8"></td>
    </tr>
    <tr>
      <td><strong>UNI2-h</strong></td>
      <td>Vision</td>
      <td>ViT-h/14</td>
      <td><strong>0.414</strong></td>
      <td><strong>0.957</strong></td>
      <td><strong>0.675</strong></td>
      <td><strong>0.977</strong></td>
      <td><strong>0.722</strong></td>
    </tr>
    <tr>
      <td>Virchow 2</td>
      <td>Vision</td>
      <td>ViT-h/14</td>
      <td>0.398</td>
      <td>0.952</td>
      <td>0.620</td>
      <td>0.975</td>
      <td>0.713</td>
    </tr>
    <tr>
      <td>Virchow</td>
      <td>Vision</td>
      <td>ViT-h/14</td>
      <td>0.398</td>
      <td>0.919</td>
      <td>0.544</td>
      <td><strong>0.977</strong></td>
      <td>0.670</td>
    </tr>
    <tr>
      <td colspan="8"></td>
    </tr>
    <tr>
      <td><strong>UNI2-g-preview</strong></td>
      <td>Vision</td>
      <td>ViT-g/14</td>
      <td><strong>0.416</strong></td>
      <td><strong>0.949</strong></td>
      <td><strong>0.690</strong></td>
      <td><strong>0.985</strong></td>
      <td><strong>0.725</strong></td>
    </tr>
    <tr>
      <td>h-optimus</td>
      <td>Vision</td>
      <td>ViT-g/14</td>
      <td>0.415</td>
      <td>0.930</td>
      <td>0.647</td>
      <td>0.970</td>
      <td>0.707</td>
    </tr>
    <tr>
      <td>Prov-GigaPath</td>
      <td>Vision</td>
      <td>ViT-g/14</td>
      <td>0.385</td>
      <td>0.929</td>
      <td>0.593</td>
      <td>0.961</td>
      <td>0.693</td>
    </tr>
    <tr>
      <td colspan="8"></td>
    </tr>
    <tr>
      <td>CONCH</td>
      <td>Vision-language</td>
      <td>ViT-b/16</td>
      <td>0.371</td>
      <td>0.941</td>
      <td>0.556</td>
      <td>0.967</td>
      <td>0.685</td>
    </tr>
    <tr>
      <td>MUSK</td>
      <td>Vision-language</td>
      <td>ViT-l/16</td>
      <td>0.346</td>
      <td>0.913</td>
      <td>0.464</td>
      <td>0.954</td>
      <td>0.666</td>
    </tr>
  </tbody>
</table>

### Slide Benchmarks

<table>
  <thead>
    <tr>
      <th>Model name</th>
      <th>Pretraining</th>
      <th>Model size</th>
      <th>EBRAINS (30 classes, Public)</th>
      <th>PANDA (5 classes, Public)</th>
      <th>IHC ER / PR Assess. (6 classes, Internal)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UNI</td>
      <td>Vision</td>
      <td>ViT-l/16</td>
      <td>0.682</td>
      <td>0.944</td>
      <td>0.776</td>
    </tr>
    <tr>
      <td colspan="6"></td>
    </tr>
    <tr>
      <td><strong>UNI2-h</strong></td>
      <td>Vision</td>
      <td>ViT-h/14</td>
      <td><strong>0.711</strong></td>
      <td><strong>0.946</strong></td>
      <td>0.794</td>
    </tr>
    <tr>
      <td>Virchow 2</td>
      <td>Vision</td>
      <td>ViT-h/14</td>
      <td>0.691</td>
      <td>0.931</td>
      <td><strong>0.808</strong></td>
    </tr>
    <tr>
      <td>Virchow</td>
      <td>Vision</td>
      <td>ViT-h/14</td>
      <td>0.681</td>
      <td><strong>0.946</strong></td>
      <td>0.756</td>
    </tr>
    <tr>
      <td colspan="6"></td>
    </tr>
    <tr>
      <td><strong>UNI2-g-preview</strong></td>
      <td>Vision</td>
      <td>ViT-g/14</td>
      <td><strong>0.746</strong></td>
      <td><strong>0.953</strong></td>
      <td><strong>0.795</strong></td>
    </tr>
    <tr>
      <td>h-optimus</td>
      <td>Vision</td>
      <td>ViT-g/14</td>
      <td>0.726</td>
      <td><strong>0.953</strong></td>
      <td>0.761</td>
    </tr>
    <tr>
      <td>Prov-GigaPath</td>
      <td>Vision</td>
      <td>ViT-g/14</td>
      <td>0.687</td>
      <td>0.944</td>
      <td>0.775</td>
    </tr>
    <tr>
      <td colspan="6"></td>
    </tr>
    <tr>
      <td>CONCH</td>
      <td>Vision-language</td>
      <td>ViT-b/16</td>
      <td>0.689</td>
      <td>0.934</td>
      <td>0.794</td>
    </tr>
    <tr>
      <td>MUSK</td>
      <td>Vision-language</td>
      <td>ViT-l/16</td>
      <td>0.660</td>
      <td>0.923</td>
      <td>0.764</td>
    </tr>
  </tbody>
</table>

In each task, for each model, we sweep over 3 learning rates (1e-5, 5e-5, 1e-4) and report the test performance corresponding to the best performing model on the validation set.

For all assessments, all models are evaluated using the global representation (e.g. CLS token) without test time augmentation.

## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/mahmoodlab/UNI.git
cd UNI
```
Then create a conda env and install the dependencies:
```shell
conda create -n UNI python=3.10 -y
conda activate UNI
pip install -e .
```


### 1. Getting access
Request access to the model weights from the Huggingface model page using links provided in the [Model Weights](#model-weights) section. You will need to login to Huggingface to download the model weights. 


### 2. Downloading weights + Creating model
Following authentication (using ```huggingface_hub```), the pretrained checkpoints and image transforms for UNI can be directly loaded using the [timm](https://huggingface.co//github/hub/en/timm) library. This method automatically downloads the model weights to the [huggingface_hub cache](https://huggingface.co//github/huggingface_hub/en/guides/manage-cache) in your home directory, which ```timm``` will automatically find when using the commands below:

```python
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# pretrained=True needed to load UNI weights (and download weights for the first time)
# using UNI2-h as example
timm_kwargs = {
   'img_size': 224, 
   'patch_size': 14, 
   'depth': 24,
   'num_heads': 24,
   'init_values': 1e-5, 
   'embed_dim': 1536,
   'mlp_ratio': 2.66667*2,
   'num_classes': 0, 
   'no_embed_class': True,
   'mlp_layer': timm.layers.SwiGLUPacked, 
   'act_layer': torch.nn.SiLU, 
   'reg_tokens': 8, 
   'dynamic_img_size': True
  }
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
```

You can also download the model weights to a specified checkpoint location in your local directory. The ```timm``` library is still used for defining the model architecture (e.g. custom ViT-H/14). Pretrained weights and image transforms for UNI need to be manually loaded and defined.
```python
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "../assets/ckpts/uni2-h/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
timm_kwargs = {
   'model_name': 'vit_giant_patch14_224',
   'img_size': 224, 
   'patch_size': 14, 
   'depth': 24,
   'num_heads': 24,
   'init_values': 1e-5, 
   'embed_dim': 1536,
   'mlp_ratio': 2.66667*2,
   'num_classes': 0, 
   'no_embed_class': True,
   'mlp_layer': timm.layers.SwiGLUPacked, 
   'act_layer': torch.nn.SiLU, 
   'reg_tokens': 8, 
   'dynamic_img_size': True
  }
model = timm.create_model(**timm_kwargs)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
 [
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
 ]
)
model.eval()
```

The function `get_encoder` performs the commands above, downloading in the checkpoint in the `./assets/ckpts/` relative path of this GitHub repository.
```python
from uni import get_encoder
model, transform = get_encoder(enc_name='uni2-h', device=device)
```

### 3. Running Inference

You can use the UNI pretrained encoder to extract features from histopathology ROIs, as follows:

```python
from PIL import Image
image = Image.open("uni.jpg")
image = transform(image).unsqueeze(dim=0) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
with torch.inference_mode():
 feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1, 1536]
```

These pre-extracted features can then be used ROI classification (via linear probing), slide classification (via multiple instance learning), and other machine learning settings.


## Overview of specific usages
We provide high-level functions for loading the model and using it for inference. For model loading, the function `get_encoder` performs the commands above in Step 2, downloading in the checkpoint in the `./assets/ckpts/` relative path of this GitHub repository.
```python
from uni import get_encoder
model, transform = get_encoder(enc_name='uni2-h', device=device)
```

For inference:
```python
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
```
Refer to the notebooks below for detailed examples.

### More detailed starter code for loading / using the model:
See [**./notebooks/uni_walkthrough.ipynb**](notebooks/uni_walkthrough.ipynb) to get started with loading and using the model to create embeddings, and example code for extracting ROI features and performing ROI classification / retrieval.

## License and Terms of Tuse

ⓒ Mahmood Lab. The models and associated code are released under the [CC-BY-NC-ND 4.0]((https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the UNI models and their derivatives, which include models trained on outputs from the UNI models or datasets created from the UNI models, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading the models, you agree not to distribute, publish or reproduce a copy of the models. If another user within your organization wishes to use the UNI models, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying models. If you are a commercial entity, please contact the corresponding author or Mass General Brigham Innovation Office.


## Acknowledgements
The project was built on top of amazing repositories such as [ViT](https://github.com/google-research/big_vision), [DINOv2](https://github.com/facebookresearch/dinov2), [LGSSL](https://github.com/mbanani/lgssl),  and [Timm](https://github.com/huggingface/pytorch-image-models/) (ViT model implementation). We thank the authors and developers for their contribution. 


## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41591-024-02857-3):

Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. Towards a general-purpose foundation model for computational pathology. Nat Med (2024). https://doi.org/10.1038/s41591-024-02857-3

```
@article{chen2024uni,
  title={Towards a General-Purpose Foundation Model for Computational Pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}
```

<img src=.github/joint_logo.jpg> 
