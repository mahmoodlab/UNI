# UNI 

## Towards a General-Purpose Foundation Model for Computational Pathology
*Nature Medicine* <img src=".github/uni.jpg" width="300px" align="right" />

[Journal Link](https://www.nature.com/articles/s41591-024-02857-3) | [Open Access Read Link](https://rdcu.be/dBMgh) | [Download Models](#model-weights) | [Download Pre-extracted Embeddings](#pre-extracted-embeddings) | [Cite](#reference) 

### Updates
- 3/20/2025: [One year overview of UNI & CONCH](https://www.linkedin.com/posts/faisalmmd_its-been-one-year-since-we-release-uni-and-activity-7308523636250820608-NedR?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAtTgDUBogopLVJVJOF9wEPZNmx4mbyt4OI) written by our team with updated table of research applications.
- 3/6/2025: [Blog Post from Meta AI](https://ai.meta.com/blog/mahmood-lab-human-pathology-dinov2/) on our development of UNI using DINOv2.
- **01/14/2025: Release of UNI 2 trained on over 200 million pathology H&E and IHC images sampled from 350+ thousand diverse whole slide images. [UNI 2 model weights](https://huggingface.co/MahmoodLab/UNI2-h), benchmark results and [25k+ pre-extracted WSI embeddings from TCGA,CPTAC, and PANDA](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features) are released.**
- 12/17/2024: [Research Highlight from Nature Medicine](https://www.nature.com/articles/s43018-024-00837-7) on UNI & CONCH for clinical oncology
- 03/19/2024: UNI is published! Model weights and initial benchmark results are released.

Unfamiliar with UNI? Please refer to the original README ([here](./README_old.md)) for more details or refer to the accompanying Nature Medicine study ([here](https://www.nature.com/articles/s41591-024-02857-3)).


## Model weights
| Model Name    | Release Date | Model Architecture | Download Link            |
|---------------------|--------------|---------------------|-------------------------------------------------------------|
| UNI2-h      |   01/2025        | ViT-h/14-reg8               | [HF Link](https://huggingface.co/MahmoodLab/UNI2-h) |
| UNI          |   03/2024        | ViT-l/16                 | [HF Link](https://huggingface.co/MahmoodLab/uni)  |

## Research Applications using UNI & CONCH
<details>
  <summary>
    <b>Last Updated 3/20/2025</b>
  </summary>

| Paper Name   | Year | Publication  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [A self-supervised framework for learning whole slide representations](https://arxiv.org/abs/2402.06188)                                             | 2024 | arXiv:2402.06188                                                   |
| [Honeybee: a scalable modular framework for creating multimodal oncology datasets with foundational embedding models](https://arxiv.org/abs/2405.07460) | 2024 | arXiv:2405.07460                                                   |
| [Combining graph neural network and mamba to capture local and global tissue spatial relationships in whole slide images](https://arxiv.org/abs/2406.04377) | 2024 | arXiv:2406.04377                                                   |
| [STimage-1K4M: A histopathology image-gene expression dataset for spatial transcriptomics](https://arxiv.org/abs/2406.06393)                         | 2024 | arXiv:2406.06393                                                   |
| [Embedding-based multimodal learning on pan-squamous cell carcinomas for improved survival outcomes](https://arxiv.org/abs/2406.08521)               | 2024 | arXiv:2406.08521                                                   |
| [A clinical benchmark of public self-supervised pathology foundation models](https://arxiv.org/abs/2407.06508v1)                                     | 2024 | arXiv:2407.06508v1                                                |
| [Path-SAM2: Transfer SAM2 for digital pathology semantic segmentation](https://arxiv.org/abs/2408.03651)                                             | 2024 | arXiv:2408.03651                                                   |
| [Benchmarking foundation models as feature extractors for weakly-supervised computational pathology](https://arxiv.org/abs/2408.15823)               | 2024 | arXiv:2408.15823                                                   |
| [Pediatric brain tumor classification using digital histopathology and deep learning: evaluation of SOTA methods on a multi-center Swedish cohort](https://arxiv.org/abs/2409.01330) | 2024 | arXiv:2409.01330                                                   |
| [Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval](https://arxiv.org/abs/2409.09430) | 2024 | arXiv:2409.09430                                                   |
| [Evaluating Deep Regression Models for WSI-Based Gene-Expression Prediction](https://arxiv.org/abs/2410.00945)                                       | 2024 | arXiv:2410.00945                                                   |
| [Deep Learning for Fetal Inflammatory Response Diagnosis in the Umbilical Cord](https://arxiv.org/abs/2411.09767)                                    | 2024 | arXiv:2411.09767                                                   |
| [Diagnostic Text-guided Representation Learning in Hierarchical Classification for Pathological Whole Slide Image](https://arxiv.org/abs/2411.10709) | 2024 | arXiv:2411.10709                                                   |
| [Leveraging Computational Pathology AI for Noninvasive Optical Imaging Analysis Without Retraining](https://arxiv.org/abs/2411.11613)                | 2024 | arXiv:2411.11613                                                   |
| [FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification](https://arxiv.org/abs/2411.14743)             | 2024 | arXiv:2411.14743                                                   |
| [RankByGene: Gene-Guided Histopathology Representation Learning Through Cross-Modal Ranking Consistency](https://arxiv.org/abs/2411.15076)           | 2024 | arXiv:2411.15076                                                   |
| [ST-Align: A Multimodal Foundation Model for Image-Gene Alignment in Spatial Transcriptomics](https://arxiv.org/abs/2411.16793)                     | 2024 | arXiv:2411.16793                                                   |
| [Multimodal Outer Arithmetic Block Dual Fusion of Whole Slide Images and Omics Data for Precision Oncology](https://arxiv.org/abs/2411.17418)        | 2024 | arXiv:2411.17418                                                   |
| [Multimodal whole slide foundation model for pathology](https://arxiv.org/abs/2411.19666)                                                            | 2024 | arXiv:2411.19666                                                   |
| [GCUNet: A GNN-Based Contextual Learning Network for Tertiary Lymphoid Structure Semantic Segmentation in Whole Slide Image](https://arxiv.org/abs/2412.06129) | 2024 | arXiv:2412.06129                                                   |
| [A multimodal ensemble approach for clear cell renal cell carcinoma treatment outcome prediction](https://arxiv.org/abs/2412.07136)                 | 2024 | arXiv:2412.07136                                                   |
| [From Histopathology Images to Cell Clouds: Learning Slide Representations with Hierarchical Cell Transformer](https://arxiv.org/abs/2412.16715)     | 2024 | arXiv:2412.16715                                                   |
| [Vision-language models do not understand negation](https://arxiv.org/abs/2501.09425)                                                                | 2025 | arXiv:2501.09425                                                   |
| [Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images](https://arxiv.org/abs/2501.14056)          | 2025 | arXiv:2501.14056                                                   |
| [Molecular-driven Foundation Model for Oncologic Pathology](https://arxiv.org/abs/2501.16652)                                                        | 2025 | arXiv:2501.16652                                                   |
| [Dynamic Hypergraph Representation for Bone Metastasis Cancer Analysis](https://arxiv.org/abs/2501.16787)                                            | 2025 | arXiv:2501.16787                                                   |
| [Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions](https://arxiv.org/abs/2502.19293)             | 2025 | arXiv:2502.19293                                                   |
| [DELST: Dual Entailment Learning for Hyperbolic Image-Gene Pretraining in Spatial Transcriptomics](https://arxiv.org/abs/2503.00804)                 | 2025 | arXiv:2503.00804                                                   |
| [Explainable Classifier for Malignant Lymphoma Subtyping via Cell Graph and Image Fusion](https://arxiv.org/abs/2503.00925)                          | 2025 | arXiv:2503.00925                                                   |
| [CrossFusion: A Multi-Scale Cross-Attention Convolutional Fusion Model for Cancer Survival Prediction](https://arxiv.org/abs/2503.02064)             | 2025 | arXiv:2503.02064                                                   |
| [Adaptive Prototype Learning for Multimodal Cancer Survival Analysis](https://arxiv.org/abs/2503.04643)                                              | 2025 | arXiv:2503.04643                                                   |
| [ecPath detects ecDNA in tumors from histopathology images](https://www.biorxiv.org/content/10.1101/2024.11.13.623494v1.abstract)                    | 2024 | bioRxiv:2024.11.13.623494v1                                        |
| [Contrastive Learning for Omics-guided Whole-slide Visual Embedding Representation](https://www.biorxiv.org/content/10.1101/2025.01.12.632280.abstract) | 2025 | bioRxiv:2025.01.12.632280                                          |
| [Multi-modal Disentanglement of Spatial Transcriptomics and Histopathology Imaging](https://www.biorxiv.org/content/10.1101/2025.02.19.638201v1)     | 2025 | bioRxiv:2025.02.19.638201v1                                       |
| [High-Parameter Spatial Multi-Omics through Histology-Anchored Integration](https://www.biorxiv.org/content/10.1101/2025.02.23.639721v1)             | 2025 | bioRxiv:2025.02.23.639721v1                                       |
| [Weakly-supervised deep learning models enable HER2-low prediction from H&E stained slides](https://breast-cancer-research.biomedcentral.com/articles/10.1186/s13058-024-01863-0) | 2024 | Breast Cancer Research                                            |
| [2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image](https://arxiv.org/abs/2412.00678)  | 2025 | Computer Vision & Pattern Recognition (CVPR)                       |
| [Transcriptomics-guided slide representation learning in computational pathology](https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Transcriptomics-guided_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html) | 2024 | Computer Vision & Pattern Recognition (CVPR)                       |
| [Morphological prototyping for unsupervised slide representation learning in computational pathology](https://openaccess.thecvf.com/content/CVPR2024/html/Song_Morphological_Prototyping_for_Unsupervised_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html) | 2024 | Computer Vision & Pattern Recognition (CVPR)                       |
| [Development and validation of novel deep learning-based models for cancer histopathology image](https://openarchive.ki.se/articles/thesis/Development_and_validation_of_novel_deep_learning-_based_models_for_cancer_histopathology_image/27291567) | 2024 | Doctoral dissertation (Karolinska Institutet)                      |
| [Multistain pretraining for slide representation learning in pathology](https://eccv.ecva.net/virtual/2024/poster/429)                               | 2024 | European Conference on Computer Vision (ICCV)                      |
| [Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology](https://openreview.net/forum?id=trj2Jq8riA) | 2025 | International Conference on Learning Representations (ICLR)        |
| [Multimodal prototyping for cancer survival prediction](https://proceedings.mlr.press/v235/song24b.html)                                            | 2024 | International Conference on Machine Learning (ICML)                |
| [High-resolution spatial transcriptomics from histology images using histosge](https://arxiv.org/abs/2407.20518)                                     | 2024 | International Conference on Bioinformatics and Biomedicine (BIBM)  |
| [Multi-resolution histopathology patch graphs for ovarian cancer subtyping](https://link.springer.com/chapter/10.1007/978-3-031-83243-7_7)           | 2024 | International Workshop on Graphs in Biomedical Image Analysis      |
| [Bridging Classification and Segmentation in Osteosarcoma Assessment via Foundation and Discrete Diffusion Models](https://arxiv.org/abs/2501.01932) | 2025 | International Symposium on Biomedical Imaging (ISBI)               |
| [1250 H&E-based cell prediction multi-classification models to capture morphologically distinct subpopulations of CD8+ T cells](https://jitc.bmj.com/content/12/Suppl_2/A1399) | 2024 | Journal for ImmunoTherapy of Cancer                                |
| [Liver fibrosis classification on trichrome histology slides using weakly supervised learning in children and young adults](https://www.sciencedirect.com/science/article/pii/S2153353924000555) | 2025 | Journal of Pathology Informatics                                   |
| [Winners of the 2024 Tuberculosis Detection Competition](https://www.linkedin.com/posts/zsoltbedohazi_winners-of-the-2024-tuberculosis-detection-activity-7186281385572065280-zpOq) | 2024 | LinkedIn post                                                      |
| [Model-based cleaning of the QUILT-1M pathology dataset for text-conditional image synthesis](https://openreview.net/forum?id=m7wYKrUjzV)             | 2024 | Medical Imaging with Deep Learning                                 |
| [Generating highly accurate pathology reports from gigapixel whole slide images with HistoGPT](https://www.medrxiv.org/content/10.1101/2024.03.15.24304211v2) | 2024 | medRxiv:2024.03.15.24304211v2                                     |
| [HIBRID: Histology and ct-DNA based Risk-stratification with Deep Learning](https://www.medrxiv.org/content/10.1101/2024.07.23.24310822.abstract)      | 2024 | medRxiv:2024.07.23.24310822                                       |
| ["SurvivMIL: A Multimodal, Multiple Instance Learning Pipeline for Survival Outcome of Neuroblastoma Patients"](https://proceedings.mlr.press/v254/naidoo24a.html) | 2024 | MICCAI Workshop on Computational Pathology with Multimodal Data (COMPAYL) |
| [Early Fusion of H&E and IHC Histology Images for Pediatric Brain Tumor Classification](https://openreview.net/forum?id=PHtzsqDi0n)                  | 2024 | MICCAI Workshop on Computational Pathology with Multimodal Data (COMPAYL) |
| [Fluoroformer: Scaling multiple instance learning to multiplexed images via attention-based channel fusion](https://arxiv.org/abs/2411.08975)        | 2024 | ML4H symposium                                                     |
| [Harnessing transcriptional regulation of alternative end-joining to predict cancer treatment](https://academic.oup.com/narcancer/article/7/1/zcaf007/8063268) | 2025 | NAR Cancer                                                         |
| [A multimodal generative AI copilot for human pathology](https://www.nature.com/articles/s41586-024-07618-3)                                          | 2024 | Nature                                                             |
| [Digital profiling of gene expression from histology images with linearized attention](https://www.nature.com/articles/s41467-024-54182-5)           | 2024 | Nature Communications                                             |
| [Demographic bias in misdiagnosis by computational pathology models](https://www.nature.com/articles/s41591-024-02885-z)                             | 2024 | Nature Medicine                                                    |
| [Hest-1k: A dataset for spatial transcriptomics and histology image analysis](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60a899cc31f763be0bde781a75e04458-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | Advanced in Neural Information Processing Systems                  |
| [Rethinking Transformer for Long Contextual Histopathology Whole Slide Image Analysis](https://openreview.net/forum?id=f3oHNyqd83)                   | 2024 | Advanced in Neural Information Processing Systems                  |
| [Leveraging tumor heterogeneity: Heterogeneous graph representation learning for cancer survival prediction in whole slide images](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760341adc5632de3f1cf2e8d22215a93-Abstract-Conference.html) | 2024 | Advanced in Neural Information Processing Systems                  |
| [Going Beyond H&E and Oncology: How Do Histopathology Foundation Models Perform for Multi-stain IHC and Immunology?](https://arxiv.org/abs/2410.21560) | 2024 | NeurIPS Workshop on Advancements In Medical Foundation Models      |
| [Histopathology and proteomics are synergistic for high-grade serous ovarian cancer platinum response prediction](https://www.nature.com/articles/s41698-025-00808-w) | 2025 | npj Precision Oncology                                             |
| [Deep learning for predicting prognostic consensus molecular subtypes in cervical cancer from histology images](https://www.nature.com/articles/s41698-024-00778-5) | 2025 | npj Precision Oncology                                             |
| [Integrated multicenter deep learning system for prognostic prediction in bladder cancer](https://www.nature.com/articles/s41698-024-00731-6)        | 2024 | npj Precision Oncology                                             |
| [Predicting the tumor microenvironment composition and immunotherapy response in non-small cell lung cancer from digital histopathology images](https://www.nature.com/articles/s41698-024-00765-w) | 2024 | npj Precision Oncology                                             |
| [Artificial intelligence-based morphologic classification and molecular characterization of neuroblastic tumors from digital histopathology](https://www.nature.com/articles/s41698-024-00745-0) | 2024 | npj Precision Oncology                                             |
| [Deep Learning-Enabled Integration of Histology and Transcriptomics for Tissue Spatial Profile Analysis](https://spj.science.org/doi/10.34133/research.0568) | 2025 | spj Research                                                       |
| [Validation of histopathology foundation models through whole slide image retrieval](https://www.nature.com/articles/s41598-025-88545-9)             | 2025 | Scientific Reports                                                 |
| [Deep Learning Framework for Classifying Whole-slide Multiplex Immunofluorescence Images to Predict Immunotherapy Response in Melanoma Patients](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173496563.35713571) | 2024 | TechRxiv:10.36227/techrxiv.173496563.35713571                      |
| [Deep learning-based lymph node metastasis status predicts prognosis from muscle-invasive bladder cancer histopathology](https://link.springer.com/article/10.1007/s00345-025-05440-8) | 2025 | World Journal of Urology                                           |
</details>

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
