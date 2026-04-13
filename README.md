# ReIB: Reversible Information Bottleneck for Unsupervised Semantic Segmentation

ReIB is an unsupervised semantic segmentation framework based on the **Reversible Information Bottleneck** principle. It leverages pretrained DINOv3 features and introduces a reconstruction-guided contrastive learning objective to learn compact yet reversible representations, enabling high-quality dense clustering without any pixel-level annotations.



## Environment

**Python:** 3.12  
**CUDA:** 12.8

### Core Dependencies

| Package | Version |
|---|---|
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| pytorch-lightning | 2.5.6 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |
| optuna | 4.6.0 |
| numpy | 2.1.2 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| Pillow | 11.3.0 |
| seaborn | 0.13.2 |
| wandb | 0.23.0 |

### Installation

```bash
conda create -n ReIB python=3.12
conda activate ReIB
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install pytorch-lightning==2.5.6 hydra-core==1.3.2 omegaconf==2.3.0
pip install optuna numpy scikit-learn scipy Pillow seaborn wandb
```

## Pretrained Backbones

Download pretrained DINOv3 weights and place them under `../Pretrained_Models/`:

* **DINOv3-S** (ViT-S/16): [`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQAYX3GOvRW-TqblUVXtEZF-AXRuz5w5JFLVEf-287QBcfI?e=XS9vRy)
* **DINOv3-B** (ViT-B/16): [`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQDhyLd0zpL0QZUe39IYf_QFAbm_Uq-HwdoQJZbEauxd5cs?e=RYEI7Y)
## Training

Edit `configs/train_config.yml` to set your dataset path and model configuration, then run:

```bash
python train_segmentation.py
```

To run hyperparameter search with Optuna:

```bash
python hyperparameter_search_epoch.py
```

## Evaluation

| Dataset | Variant | Download Link |
| :--- | :--- | :--- |
| **COCO-Stuff** | Small | [Download Link](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQAmotg1_dY7TZaNmGDqfZ12AWBeal-0oFoC2E2q1f-Ztts?e=BwKR1G) |
| | Base | [Download Link](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQAmotg1_dY7TZaNmGDqfZ12AWBeal-0oFoC2E2q1f-Ztts?e=BwKR1G) |
| **Cityscapes** | Small | [Download Link](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQCScnTtTCJkTbwM--8H06EhAWY_-4DqpnjrHl5Z5UKVJNQ?e=sWFo8e) |
| | Base | [Download Link](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQD1moSgGF9aQooctOt1xOCUAasRp3UJllT0PGAX6t1BXgw?e=Qg6V6M) |
| **Potsdam-3** | - | [Download Link](https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQC14axtwOqaR47uAChbTxQgAZVkawyy9jbceZ3sOajQTgI?e=PgG7yN) |

```bash
python eval_segmentation.py
```


