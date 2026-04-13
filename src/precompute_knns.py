from data import ContrastiveSegDataset
from modules import *
import os
from os.path import join
import hydra
import numpy as np
import torch.multiprocessing
import torch.multiprocessing
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
# from pytorch_lightning.utilities.seed import seed_everything  # Old import path
from tqdm import tqdm


def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(version_base="1.1", config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    # 使用两张卡：GPU 0 和 GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join(pytorch_data_dir, "nns"), exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    image_sets = ["val", "train"]
    # Only process the dataset specified in config
    dataset_names = [cfg.dataset_name]  # Use dataset from config file
    crop_types = [cfg.crop_type]  # Use crop_type from config file

    # Uncomment these lines to run on custom datasets
    #dataset_names = ["directory"]
    #crop_types = [None]

    res = cfg.res  # Use resolution from config file
    n_batches = 128  # 增加批次数以减少每批的显存占用

    if cfg.arch == "dino":
        from modules import DinoFeaturizer, LambdaLayer
        no_ap_model = torch.nn.Sequential(
            DinoFeaturizer(20, cfg),  # dim doesent matter
            LambdaLayer(lambda p: p[0]),
        ).cuda()
    else:
        cut_model = load_model(cfg.model_type, join(cfg.output_root, "data")).cuda()
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).cuda()
    # 使用DataParallel在两张卡上并行
    par_model = torch.nn.DataParallel(no_ap_model, device_ids=[0, 1])

    for crop_type in crop_types:
        for image_set in image_sets:
            for dataset_name in dataset_names:
                nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name

                feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
                    cfg.model_type, nice_dataset_name, image_set, crop_type, res))

                if not os.path.exists(feature_cache_file):
                    print("{} not found, computing".format(feature_cache_file))
                    dataset = ContrastiveSegDataset(
                        pytorch_data_dir=pytorch_data_dir,
                        dataset_name=dataset_name,
                        crop_type=crop_type,
                        image_set=image_set,
                        transform=get_transform(res, False, "center"),
                        target_transform=get_transform(res, True, "center"),
                        cfg=cfg,
                    )

                    # 使用两张卡，batch size可以提高到128
                    loader = DataLoader(dataset, 128, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

                    with torch.no_grad():
                        normed_feats = get_feats(par_model, loader)
                        all_nns = []
                        step = normed_feats.shape[0] // n_batches
                        print(f"Total features: {normed_feats.shape}, batch size: {step}")
                        # 不把所有特征都放到GPU，而是每次只处理需要的batch
                        for i in tqdm(range(0, normed_feats.shape[0], step)):
                            batch_feats = normed_feats[i:i + step, :].cuda()
                            # 分批计算相似度，避免一次性加载所有特征到GPU
                            chunk_size = 10000  # 每次只加载10000个特征到GPU
                            batch_similarities = []
                            for j in range(0, normed_feats.shape[0], chunk_size):
                                chunk_feats = normed_feats[j:j + chunk_size, :].cuda()
                                chunk_sims = torch.einsum("nf,mf->nm", batch_feats, chunk_feats)
                                batch_similarities.append(chunk_sims.cpu())
                                del chunk_feats, chunk_sims
                                torch.cuda.empty_cache()
                            
                            pairwise_sims = torch.cat(batch_similarities, dim=1)
                            all_nns.append(torch.topk(pairwise_sims, 30)[1])
                            del pairwise_sims, batch_feats, batch_similarities
                            torch.cuda.empty_cache()
                        
                        nearest_neighbors = torch.cat(all_nns, dim=0)

                        np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                        print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
