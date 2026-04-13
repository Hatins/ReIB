from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels
from pytorch_lightning import seed_everything
import numpy as np
import csv


def _coerce_cfg(cfg_obj: object) -> DictConfig:
    if isinstance(cfg_obj, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg_obj, resolve=False))
    return OmegaConf.create(cfg_obj)


def _load_model_using_cfg_from_checkpoint(model_path: str) -> LitUnsupervisedSegmenter:
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    hp = ckpt.get('hyper_parameters', {})
    if 'cfg' not in hp or 'n_classes' not in hp:
        raise RuntimeError(
            f"Checkpoint missing required hyper_parameters: got keys={list(hp.keys())}. "
            "Expected at least ['n_classes', 'cfg']."
        )

    cfg = _coerce_cfg(hp['cfg'])
    state_dict = ckpt.get('state_dict', {})

    # Backward-compat: some older checkpoints stored cfg.dino_feat_type='fea' but
    # the actual weights correspond to single-layer features (384/768).
    w = state_dict.get('net.cluster1.0.weight', None)
    if w is not None:
        in_ch = int(w.shape[1])
        model_type = cfg.get('model_type', None)
        base_dim = 384 if model_type == 'vit_small' else 768 if model_type == 'vit_base' else None
        if base_dim is not None:
            old_feat_type = cfg.get('dino_feat_type', None)
            if in_ch == base_dim and old_feat_type == 'fea':
                cfg['dino_feat_type'] = 'ori'
                print(
                    f"[eval] Adjusted dino_feat_type: {old_feat_type} -> {cfg['dino_feat_type']} "
                    f"to match checkpoint features (in_channels={in_ch})."
                )
            elif in_ch == base_dim * 3 and old_feat_type != 'fea':
                cfg['dino_feat_type'] = 'fea'
                print(
                    f"[eval] Adjusted dino_feat_type: {old_feat_type} -> {cfg['dino_feat_type']} "
                    f"to match checkpoint features (in_channels={in_ch})."
                )

    model = LitUnsupervisedSegmenter(int(hp['n_classes']), cfg)
    model.load_state_dict(state_dict, strict=True)
    return model

# 设置 multiprocessing start method 为 spawn，避免 PyTorch 2.x 与 CUDA 的死锁问题
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # 如果已经设置过，忽略错误
torch.multiprocessing.set_sharing_strategy('file_system')

def plot_cm(histogram, label_cmap, cfg):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]
    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)
    colors = [label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


@hydra.main(version_base="1.1", config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    import os
    import torch
    
    # Set seed for reproducible clustering colors
    seed_everything(seed=42)
    
    gpu_id = cfg.get('gpu_id', 0)
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU: {gpu_id}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Pretrained weights path from checkpoint will be: ../Pretrained_Models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    print(f"Absolute path: {os.path.abspath('../Pretrained_Models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')}")
    
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/SCON_results/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "picie"), exist_ok=True)
    
    if cfg.get('save_features', False):
        os.makedirs(join(result_dir, "feats"), exist_ok=True)
        os.makedirs(join(result_dir, "code"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = _load_model_using_cfg_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

        run_picie = cfg.run_picie and model.cfg.dataset_name == "cocostuff27"
        if run_picie:
            picie_state = torch.load("../saved_models/picie_and_probes.pth")
            picie = picie_state["model"].cuda()
            picie_cluster_probe = picie_state["cluster_probe"].module.cuda()
            picie_cluster_metrics = picie_state["cluster_metrics"]

        loader_crop = "center"
        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            cfg=model.cfg,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True, collate_fn=flexible_collate)

        model.eval().cuda()

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            if run_picie:
                par_picie = torch.nn.DataParallel(picie)
        else:
            par_model = model.net
            if run_picie:
                par_picie = picie

        num_save = cfg.get('num_save_images', 10)
        all_good_images = list(range(num_save))
        batch_nums = torch.tensor([n // (cfg.batch_size * 2) for n in all_good_images])
        batch_offsets = torch.tensor([n % (cfg.batch_size * 2) for n in all_good_images])

        saved_data = defaultdict(list)
        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()

                    feats, code = par_model(img)

                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                    linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
                    cluster_probs = model.cluster_probe(code, 2, log_probs=True)

                    if cfg.run_crf:
                        linear_preds = batched_crf(pool, img, linear_probs).argmax(1).cuda()
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                    else:
                        linear_preds = linear_probs.argmax(1)
                        cluster_preds = cluster_probs.argmax(1)

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

                    if run_picie:
                        picie_preds = picie_cluster_metrics.map_clusters(
                            picie_cluster_probe(par_picie(img), None)[1].argmax(1).cpu())

                    if i in batch_nums:
                        matching_offsets = batch_offsets[torch.where(batch_nums == i)]
                        for offset in matching_offsets:
                            saved_data["linear_preds"].append(linear_preds.cpu()[offset].unsqueeze(0))
                            saved_data["cluster_preds"].append(cluster_preds.cpu()[offset].unsqueeze(0))
                            saved_data["label"].append(label.cpu()[offset].unsqueeze(0))
                            saved_data["img"].append(img.cpu()[offset].unsqueeze(0))

                            if cfg.get('save_features', False):
                                feats_single = feats[offset:offset+1]
                                feats_vis = F.interpolate(feats_single, scale_factor=2, mode='bilinear', align_corners=False)
                                feats_interp = F.interpolate(feats_vis, label.shape[-2:], mode='bilinear', align_corners=False)
                                saved_data["feats"].append(feats_interp.cpu())
                                saved_data["code"].append(code.cpu()[offset].unsqueeze(0))
                            if run_picie:
                                saved_data["picie_preds"].append(picie_preds.cpu()[offset].unsqueeze(0))
        saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}

        tb_metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }

        print("")
        print(model_path)
        print(tb_metrics)
        
        per_image_metrics = []
        for img_idx in range(len(saved_data["cluster_preds"])):
            mapped_pred = model.test_cluster_metrics.map_clusters(saved_data["cluster_preds"][img_idx])
            gt = saved_data["label"][img_idx]
            mask = (gt >= 0) & (gt < model.n_classes)
            pred_masked = mapped_pred[mask]
            gt_masked = gt[mask]
            
            if mask.sum() > 0:
                gt_classes = torch.unique(gt_masked)
                ious = []
                for c in gt_classes:
                    pred_c = (pred_masked == c)
                    gt_c = (gt_masked == c)
                    intersection = (pred_c & gt_c).sum().float()
                    union = (pred_c | gt_c).sum().float()
                    if union > 0:
                        ious.append((intersection / union).item())
                
                miou = sum(ious) / len(ious) if ious else 0.0
            else:
                miou = 0.0
            
            per_image_metrics.append({
                'image_idx': img_idx,
                'mIoU': miou * 100
            })
        
        csv_path = join(result_dir, "per_image_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_idx', 'mIoU'])
            writer.writeheader()
            writer.writerows(per_image_metrics)
        print(f"\nPer-image metrics saved to: {csv_path}")
        print(f"Total images evaluated: {len(per_image_metrics)}")
        if per_image_metrics:
            mious = [m['mIoU'] for m in per_image_metrics]
            print(f"mIoU range: {min(mious):.2f} - {max(mious):.2f}")
            print(f"Mean mIoU: {sum(mious)/len(mious):.2f}")

        if cfg.run_prediction:
            n_rows = 3
        else:
            n_rows = 2

        if run_picie:
            n_rows += 1

        if cfg.dark_mode:
            plt.style.use('dark_background')

        for good_images in batch_list(range(len(all_good_images)), 10):
            fig, ax = plt.subplots(n_rows, len(good_images), figsize=(len(good_images) * 3, n_rows * 3))
            for i, img_num in enumerate(good_images):
                plot_img = (prep_for_plot(saved_data["img"][img_num]) * 255).numpy().astype(np.uint8)
                plot_label = (model.label_cmap[saved_data["label"][img_num]]).astype(np.uint8)
                Image.fromarray(plot_img).save(join(join(result_dir, "img", str(img_num) + ".jpg")))
                Image.fromarray(plot_label).save(join(join(result_dir, "label", str(img_num) + ".png")))
                
                if cfg.get('save_features', False):
                    from sklearn.decomposition import PCA
                    
                    feats_np = saved_data["feats"][img_num].numpy()
                    feats_reshaped = feats_np.reshape(feats_np.shape[0], -1).T
                    pca = PCA(n_components=3)
                    feats_pca = pca.fit_transform(feats_reshaped)
                    feats_pca = feats_pca.reshape(feats_np.shape[1], feats_np.shape[2], 3)
                    feats_pca = (feats_pca - feats_pca.min()) / (feats_pca.max() - feats_pca.min() + 1e-8)
                    feats_pca = (feats_pca * 255).astype(np.uint8)
                    Image.fromarray(feats_pca).save(join(result_dir, "feats", str(img_num) + ".png"))
                    
                    code_np = saved_data["code"][img_num].numpy()
                    code_reshaped = code_np.reshape(code_np.shape[0], -1).T
                    pca_code = PCA(n_components=3)
                    code_pca = pca_code.fit_transform(code_reshaped)
                    code_pca = code_pca.reshape(code_np.shape[1], code_np.shape[2], 3)
                    code_pca = (code_pca - code_pca.min()) / (code_pca.max() - code_pca.min() + 1e-8)
                    code_pca = (code_pca * 255).astype(np.uint8)
                    Image.fromarray(code_pca).save(join(result_dir, "code", str(img_num) + ".png"))

                ax[0, i].imshow(plot_img)
                ax[1, i].imshow(plot_label)
                if cfg.run_prediction:
                    plot_cluster = (model.label_cmap[
                        model.test_cluster_metrics.map_clusters(
                            saved_data["cluster_preds"][img_num])]) \
                        .astype(np.uint8)
                    Image.fromarray(plot_cluster).save(join(join(result_dir, "cluster", str(img_num) + ".png")))
                    ax[2, i].imshow(plot_cluster)
                if run_picie:
                    picie_img = model.label_cmap[saved_data["picie_preds"][img_num]].astype(np.uint8)
                    ax[3, i].imshow(picie_img)
                    Image.fromarray(picie_img).save(join(join(result_dir, "picie", str(img_num) + ".png")))

            ax[0, 0].set_ylabel("Image", fontsize=26)
            ax[1, 0].set_ylabel("Label", fontsize=26)
            if cfg.run_prediction:
                ax[2, 0].set_ylabel("STEGO\n(Ours)", fontsize=26)
            if run_picie:
                ax[3, 0].set_ylabel("PiCIE\n(Baseline)", fontsize=26)

            remove_axes(ax)
            plt.tight_layout()
            plt.show()
            plt.clf()

        plot_cm(model.test_cluster_metrics.histogram, model.label_cmap, model.cfg)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    prep_args()
    my_app()
