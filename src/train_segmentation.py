from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.meshgrid.*')

torch.multiprocessing.set_sharing_strategy('file_system')

def ssim_loss(x, y, window_size=11, size_average=True):
    """SSIM Loss for feature reconstruction"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    channel = x.size(1)
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(x.device)
    
    mu1 = F.conv2d(x, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            dino_version = cfg.get("dino_version", "v1")
            if dino_version == "v3":
                self.net = Dinov3Featurizer(dim, cfg)
            else:
                self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.train_cluster_probe = ClusterLookup(dim, n_classes)

        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        
        # ===== SCON新增：软聚类探测器和重建网络 =====
        self.soft_cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        
        if cfg.get('dino_feat_type', 'feat') == 'fea':
            recon_feature_dim = self.net.n_feats // 3
        else:
            recon_feature_dim = self.net.n_feats
        
        # 根据config选择重建网络类型
        recon_type = cfg.get('recon_type', 'shallow')  # 'shallow' or 'deep'
        if recon_type == 'deep':
            self.reconstruction_net = DeepReconstructionNet(
                n_classes=n_classes + cfg.extra_clusters,
                feature_dim=recon_feature_dim,
                hidden_dim=cfg.get('recon_hidden_dim', 128)
            )
        else:
            self.reconstruction_net = ReconstructionNet(
                n_classes=n_classes + cfg.extra_clusters,
                feature_dim=recon_feature_dim,
                hidden_dim=cfg.get('recon_hidden_dim', 128)
            )

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.validation_step_outputs = []  # 存储验证步骤的输出
        self.save_hyperparameters()
        
        # ===== SCON2新增：加载预训练分割模型 =====
        if hasattr(cfg, 'pretrained_seg') and cfg.pretrained_seg is not None:
            self.load_pretrained_segmentation(cfg.pretrained_seg)
    
    def load_pretrained_segmentation(self, checkpoint_path):
        """加载预训练的分割模型权重（来自STEGO或SCON）"""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Pretrained checkpoint not found at {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (将使用随机初始化): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys (将被忽略): {unexpected_keys[:5]}...")
        
        print(f"\n✓ Successfully loaded pretrained segmentation weights from:")
        print(f"  {checkpoint_path}")
        print(f"  - Loaded keys: {len(state_dict) - len(missing_keys)}")
        print(f"  - Missing keys: {len(missing_keys)}")
        print(f"{'='*60}\n")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        net_optim, linear_probe_optim, cluster_probe_optim, soft_cluster_optim, recon_optim = self.optimizers()

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()
        soft_cluster_optim.zero_grad()
        recon_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        feats, code = self.net(img)
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)
        log_args = dict(sync_dist=False, rank_zero_only=True)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            if self.cfg.correspondence_weight > 0:
                signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            if self.cfg.correspondence_weight > 0:
                signal_pos = feats_pos

        loss = 0

        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)
        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            if self.cfg.correspondence_weight > 0:
                salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            if self.cfg.correspondence_weight > 0:
                salience_pos = None

        # ===== STEGO对比学习损失 =====
        stego_loss = 0
        if self.cfg.correspondence_weight > 0:
            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd,
            ) = self.contrastive_corr_loss_fn(
                signal, signal_pos,
                salience, salience_pos,
                code, code_pos,
            )

            # WandB不支持add_histogram，使用wandb.Histogram代替
            if should_log_hist:
                import wandb
                self.logger.experiment.log({
                    "intra_cd": wandb.Histogram(pos_intra_cd.detach().cpu().numpy()),
                    "inter_cd": wandb.Histogram(pos_inter_cd.detach().cpu().numpy()),
                    "neg_cd": wandb.Histogram(neg_inter_cd.detach().cpu().numpy())
                }, step=self.global_step)
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_intra', pos_intra_loss, **log_args)
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            stego_loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight

        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log('loss/rec', rec_loss, **log_args)
            stego_loss += self.cfg.rec_weight * rec_loss

        if self.cfg.aug_alignment_weight > 0:
            orig_feats_aug, orig_code_aug = self.net(img_aug)
            downsampled_coord_aug = resize(
                coord_aug.permute(0, 3, 1, 2),
                orig_code_aug.shape[2]).permute(0, 2, 3, 1)
            aug_alignment = -torch.einsum(
                "bkhw,bkhw->bhw",
                norm(sample(code, downsampled_coord_aug)),
                norm(orig_code_aug)
            ).mean()
            self.log('loss/aug_alignment', aug_alignment, **log_args)
            stego_loss += self.cfg.aug_alignment_weight * aug_alignment

        if self.cfg.crf_weight > 0:
            crf = self.crf_loss_fn(
                resize(img, 56),
                norm(resize(code, 56))
            ).mean()
            self.log('loss/crf', crf, **log_args)
            stego_loss += self.cfg.crf_weight * crf

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        
        # ===== SCON新增：重建损失 =====
        recon_loss_total = 0
        soft_alpha = self.cfg.get('soft_cluster_alpha', 1.0)
        _, soft_cluster_probs = self.soft_cluster_probe(code, alpha=soft_alpha)
        reconstructed_feats = self.reconstruction_net(soft_cluster_probs)

        if self.cfg.get('dino_feat_type', 'feat') == 'fea':
            # feats是拼接的3层特征，取最后一段作为重建目标
            target_feats = feats[:, -self.reconstruction_net.feature_dim:, :, :]
        else:
            target_feats = feats
        
        # 根据配置选择重建损失类型
        recon_loss_type = self.cfg.get('recon_loss_type', 'ssim')
        if recon_loss_type == 'mse':
            recon_loss = F.mse_loss(reconstructed_feats, target_feats)
        elif recon_loss_type == 'cosine':
            recon_loss = 1 - F.cosine_similarity(reconstructed_feats, target_feats, dim=1).mean()
        elif recon_loss_type == 'relaxed_cosine':
            cos_sim = F.cosine_similarity(reconstructed_feats, target_feats, dim=1)
            margin = 0.1
            recon_loss = torch.clamp(1 - cos_sim - margin, min=0).mean()
        elif recon_loss_type == 'ssim':
            recon_loss = ssim_loss(reconstructed_feats, target_feats)
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
        
        recon_loss_total = recon_loss
        self.log('loss/reconstruction', recon_loss, **log_args)
        
        # ===== 组合损失：使用权重控制STEGO loss和重建loss =====
        stego_weight = self.cfg.get('stego_loss_weight', 1.0)
        recon_weight = self.cfg.get('recon_loss_weight', 1.0)
        
        loss += stego_weight * stego_loss
        loss += recon_weight * recon_loss_total
        
        self.log('loss/stego_total', stego_loss, **log_args)
        self.log('loss/stego_weighted', stego_weight * stego_loss, **log_args)
        self.log('loss/recon_weighted', recon_weight * recon_loss_total, **log_args)
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()
        soft_cluster_optim.step()
        recon_optim.step()

        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)
        
        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # WandB logger 不需要手动重置文件
            # Make a new tfevent file
            try:
                if hasattr(self.logger.experiment, 'close'):
                    self.logger.experiment.close()
                if hasattr(self.logger.experiment, '_get_file_writer'):
                    self.logger.experiment._get_file_writer()
            except Exception as e:
                print(f"Warning: Could not reset logger: {e}")

        # 为optuna输出训练进度
        import os
        if 'OPTUNA_EXPERIMENT_NAME' in os.environ and self.global_step % 30 == 0:
            print(f"Training: global_step={self.global_step}", flush=True)

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute()
        }
        # WandB Logger只接受一个参数（字典形式）
        self.logger.log_hyperparams(dict(self.cfg))

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)
            # 保存原始小尺寸code用于重建
            code_original = code
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            output = {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}
            
            if batch_idx == 0:
                # DINO特征可视化 - 只对前n_images进行插值以避免内存溢出
                feats_small = feats[:self.cfg.n_images]
                feats_vis = F.interpolate(feats_small, scale_factor=2, mode='bilinear', align_corners=False)
                feats_interp = F.interpolate(feats_vis, label.shape[-2:], mode='bilinear', align_corners=False)
                output['feats'] = feats_interp.detach().cpu()
                output['code'] = code[:self.cfg.n_images].detach().cpu()
                
                # ===== SCON新增：计算重建特征用于可视化 =====
                soft_alpha = self.cfg.get('soft_cluster_alpha', 1.0)
                _, soft_cluster_probs = self.soft_cluster_probe(code_original, alpha=soft_alpha)
                reconstructed = self.reconstruction_net(soft_cluster_probs)
                
                # 提取重建目标特征
                if self.cfg.get('dino_feat_type', 'feat') == 'fea':
                    target_feats = feats[:, -self.reconstruction_net.feature_dim:, :, :]
                else:
                    target_feats = feats
                
                # 只对需要可视化的n_images进行插值
                reconstructed_small = reconstructed[:self.cfg.n_images]
                soft_cluster_probs_small = soft_cluster_probs[:self.cfg.n_images]
                target_feats_small = target_feats[:self.cfg.n_images]
                
                # 插值到目标大小
                reconstructed_interp = F.interpolate(reconstructed_small, label.shape[-2:], mode='bilinear', align_corners=False)
                soft_cluster_probs_interp = F.interpolate(soft_cluster_probs_small, label.shape[-2:], mode='bilinear', align_corners=False)
                target_feats_vis = F.interpolate(target_feats_small, scale_factor=2, mode='bilinear', align_corners=False)
                target_feats_interp = F.interpolate(target_feats_vis, label.shape[-2:], mode='bilinear', align_corners=False)
                
                output['reconstructed'] = reconstructed_interp.detach().cpu()
                output['soft_cluster_probs'] = soft_cluster_probs_interp.detach().cpu()
                output['target_feats'] = target_feats_interp.detach().cpu()
            
            self.validation_step_outputs.append(output)
            return output

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml and len(self.validation_step_outputs) > 0:
                output = {k: v.detach().cpu() for k, v in self.validation_step_outputs[0].items()}

                if 'feats' in output and 'code' in output:
                    # 有重建特征时显示9行
                    fig, ax = plt.subplots(9, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 9 * 3))
                    for i in range(self.cfg.n_images):
                        ax[0, i].imshow(prep_for_plot(output["img"][i]))
                        ax[1, i].imshow(self.label_cmap[output["label"][i]])
                        
                        feats_norm = torch.norm(output["feats"][i], dim=0).numpy()
                        feats_norm = (feats_norm - feats_norm.min()) / (feats_norm.max() - feats_norm.min() + 1e-8)
                        im1 = ax[2, i].imshow(feats_norm, cmap='viridis')
                        if i == self.cfg.n_images - 1:
                            plt.colorbar(im1, ax=ax[2, i], fraction=0.046, pad=0.04)
                        
                        code_norm = torch.norm(output["code"][i], dim=0).numpy()
                        code_norm = (code_norm - code_norm.min()) / (code_norm.max() - code_norm.min() + 1e-8)
                        im2 = ax[3, i].imshow(code_norm, cmap='magma')
                        if i == self.cfg.n_images - 1:
                            plt.colorbar(im2, ax=ax[3, i], fraction=0.046, pad=0.04)
                        
                        # ===== SCON新增：显示软聚类概率、目标特征和重建特征 =====
                        if 'soft_cluster_probs' in output:
                            cluster_map = output["soft_cluster_probs"][i].argmax(0).numpy()
                            im_cluster = ax[4, i].imshow(cluster_map, cmap='tab20', vmin=0, vmax=26)
                            if i == self.cfg.n_images - 1:
                                plt.colorbar(im_cluster, ax=ax[4, i], fraction=0.046, pad=0.04)
                        
                        if 'target_feats' in output:
                            target_norm = torch.norm(output["target_feats"][i], dim=0).numpy()
                            target_norm = (target_norm - target_norm.min()) / (target_norm.max() - target_norm.min() + 1e-8)
                            im_target = ax[5, i].imshow(target_norm, cmap='cividis')
                            if i == self.cfg.n_images - 1:
                                plt.colorbar(im_target, ax=ax[5, i], fraction=0.046, pad=0.04)
                        
                        if 'reconstructed' in output:
                            recon_norm = torch.norm(output["reconstructed"][i], dim=0).numpy()
                            recon_norm = (recon_norm - recon_norm.min()) / (recon_norm.max() - recon_norm.min() + 1e-8)
                            im3 = ax[6, i].imshow(recon_norm, cmap='plasma')
                            if i == self.cfg.n_images - 1:
                                plt.colorbar(im3, ax=ax[6, i], fraction=0.046, pad=0.04)
                        
                        ax[7, i].imshow(self.label_cmap[output["linear_preds"][i]])
                        ax[8, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                    ax[0, 0].set_ylabel("Image", fontsize=16)
                    ax[1, 0].set_ylabel("Label", fontsize=16)
                    ax[2, 0].set_ylabel("DINO Features", fontsize=16)
                    ax[3, 0].set_ylabel("Code Features", fontsize=16)
                    ax[4, 0].set_ylabel("Soft Clusters", fontsize=16)
                    ax[5, 0].set_ylabel("Target Feats", fontsize=16)
                    ax[6, 0].set_ylabel("Reconstructed", fontsize=16)
                    ax[7, 0].set_ylabel("Linear Probe", fontsize=16)
                    ax[8, 0].set_ylabel("Cluster Probe", fontsize=16)
                else:
                    fig, ax = plt.subplots(4, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
                    for i in range(self.cfg.n_images):
                        ax[0, i].imshow(prep_for_plot(output["img"][i]))
                        ax[1, i].imshow(self.label_cmap[output["label"][i]])
                        ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
                        ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                    ax[0, 0].set_ylabel("Image", fontsize=16)
                    ax[1, 0].set_ylabel("Label", fontsize=16)
                    ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
                    ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger.experiment.log, "plot_labels", self.global_step)

                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues")
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    names = get_class_labels(self.cfg.dataset_name)
                    if self.cfg.extra_clusters:
                        names = names + ["Extra"]
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=14)
                    ax.yaxis.set_ticklabels(names, fontsize=14)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    # ax.yaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    # ax.xaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot(self.logger.experiment.log, "conf_matrix", self.global_step)

                    all_bars = torch.cat([
                        self.cluster_metrics.histogram.sum(0).cpu(),
                        self.cluster_metrics.histogram.sum(1).cpu()
                    ], axis=0)
                    ymin = max(all_bars.min() * .8, 1)
                    ymax = all_bars.max() * 1.2

                    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
                    ax[0].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(0).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[0].set_ylim(ymin, ymax)
                    ax[0].set_title("Label Frequency")
                    ax[0].set_yscale('log')
                    ax[0].tick_params(axis='x', labelrotation=90)

                    ax[1].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(1).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[1].set_ylim(ymin, ymax)
                    ax[1].set_title("Cluster Frequency")
                    ax[1].set_yscale('log')
                    ax[1].tick_params(axis='x', labelrotation=90)

                    plt.tight_layout()
                    add_plot(self.logger.experiment.log, "label frequency", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)
                
                # 为optuna超参搜索输出mIoU和Accuracy指标
                import os
                if 'OPTUNA_EXPERIMENT_NAME' in os.environ:
                    cluster_miou = tb_metrics.get('test/cluster/mIoU', 0.0)
                    cluster_acc = tb_metrics.get('test/cluster/Accuracy', 0.0)
                    print(f"OPTUNA_METRIC: step={self.global_step}, mIoU={cluster_miou:.4f}, Acc={cluster_acc:.4f}", flush=True)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()
            
            # 清空验证输出列表
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)
        
        # ===== SCON新增：软聚类和重建网络的优化器 =====
        soft_cluster_lr = self.cfg.get('soft_cluster_lr', 5e-3)
        recon_lr = self.cfg.get('recon_lr', 5e-3)
        soft_cluster_optim = torch.optim.Adam(list(self.soft_cluster_probe.parameters()), lr=soft_cluster_lr)
        recon_optim = torch.optim.Adam(list(self.reconstruction_net.parameters()), lr=recon_lr)

        return net_optim, linear_probe_optim, cluster_probe_optim, soft_cluster_optim, recon_optim


@hydra.main(version_base="1.1", config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    # 从配置文件读取GPU设置，也支持从环境变量覆盖
    import os
    gpu_ids = os.environ.get('GPU_IDS', cfg.get('gpu_ids', '0'))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    # 设置使用的GPU数量（应该与CUDA_VISIBLE_DEVICES中的卡数一致）
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    
    OmegaConf.set_struct(cfg, False)
    
    # 支持从环境变量读取optuna超参数 - STEGO原有参数
    if 'OPTUNA_POS_INTER_WEIGHT' in os.environ:
        cfg.pos_inter_weight = float(os.environ['OPTUNA_POS_INTER_WEIGHT'])
    if 'OPTUNA_POS_INTRA_WEIGHT' in os.environ:
        cfg.pos_intra_weight = float(os.environ['OPTUNA_POS_INTRA_WEIGHT'])
    if 'OPTUNA_NEG_INTER_WEIGHT' in os.environ:
        cfg.neg_inter_weight = float(os.environ['OPTUNA_NEG_INTER_WEIGHT'])
    if 'OPTUNA_NEG_INTER_SHIFT' in os.environ:
        cfg.neg_inter_shift = float(os.environ['OPTUNA_NEG_INTER_SHIFT'])
    if 'OPTUNA_POS_INTER_SHIFT' in os.environ:
        cfg.pos_inter_shift = float(os.environ['OPTUNA_POS_INTER_SHIFT'])
    if 'OPTUNA_POS_INTRA_SHIFT' in os.environ:
        cfg.pos_intra_shift = float(os.environ['OPTUNA_POS_INTRA_SHIFT'])
    if 'OPTUNA_DIM' in os.environ:
        cfg.dim = int(os.environ['OPTUNA_DIM'])
    
    # 支持从环境变量读取optuna超参数 - SCON2新增参数
    if 'OPTUNA_RECON_HIDDEN_DIM' in os.environ:
        cfg.recon_hidden_dim = int(os.environ['OPTUNA_RECON_HIDDEN_DIM'])
    if 'OPTUNA_SOFT_CLUSTER_ALPHA' in os.environ:
        cfg.soft_cluster_alpha = float(os.environ['OPTUNA_SOFT_CLUSTER_ALPHA'])
    if 'OPTUNA_STEGO_LOSS_WEIGHT' in os.environ:
        cfg.stego_loss_weight = float(os.environ['OPTUNA_STEGO_LOSS_WEIGHT'])
    if 'OPTUNA_RECON_LOSS_WEIGHT' in os.environ:
        cfg.recon_loss_weight = float(os.environ['OPTUNA_RECON_LOSS_WEIGHT'])
    if 'OPTUNA_RECON_LR' in os.environ:
        cfg.recon_lr = float(os.environ['OPTUNA_RECON_LR'])
    if 'OPTUNA_SOFT_CLUSTER_LR' in os.environ:
        cfg.soft_cluster_lr = float(os.environ['OPTUNA_SOFT_CLUSTER_LR'])
    if 'OPTUNA_LR' in os.environ:
        cfg.lr = float(os.environ['OPTUNA_LR'])
    
    if 'OPTUNA_MAX_STEPS' in os.environ:
        cfg.max_steps = int(os.environ['OPTUNA_MAX_STEPS'])
    if 'OPTUNA_VAL_FREQ' in os.environ:
        cfg.val_freq = int(os.environ['OPTUNA_VAL_FREQ'])
    if 'OPTUNA_EXPERIMENT_NAME' in os.environ:
        cfg.experiment_name = os.environ['OPTUNA_EXPERIMENT_NAME']
    

    
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=100)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    #val_dataset = MaterializedDataset(val_dataset)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    # 使用WandbLogger，指定save_dir到output_root
    tb_logger = WandbLogger(
        name=cfg.log_dir + "_" + cfg.experiment_name,
        project=cfg.get("project_name", "stego-segmentation"),
        entity=cfg.get("entity", None),
        save_dir=cfg.output_root,
        offline=False
    )

    if cfg.submitting_to_aml:
        gpu_args = dict(devices=num_gpus, accelerator='gpu', val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        # 使用多卡时需要DDP策略
        if num_gpus > 1:
            gpu_args = dict(devices=num_gpus, accelerator='gpu', strategy='ddp_find_unused_parameters_true', val_check_interval=cfg.val_freq)
        else:
            gpu_args = dict(devices=num_gpus, accelerator='gpu', val_check_interval=cfg.val_freq)

        # 注释掉过于严格的限制，允许optuna设置任意验证频率
        # if gpu_args["val_check_interval"] > len(train_loader) // 4:
        #     gpu_args.pop("val_check_interval")

    # 支持按epoch评估：当设置环境变量 OPTUNA_EVAL_EVERY_N_EPOCHS 时，改为每N个epoch验证。
    eval_every_n_epochs = None
    if 'OPTUNA_EVAL_EVERY_N_EPOCHS' in os.environ:
        try:
            eval_every_n_epochs = int(os.environ['OPTUNA_EVAL_EVERY_N_EPOCHS'])
        except Exception:
            eval_every_n_epochs = cfg.get('eval_every_n_epochs', 1)
        # 移除按step的验证频率，避免与epoch模式冲突
        if isinstance(gpu_args, dict) and 'val_check_interval' in gpu_args:
            gpu_args.pop('val_check_interval')

    checkpoint_callback = ModelCheckpoint(
        dirpath=join(checkpoint_dir, name),
        filename="step={step}-mIoU={test/cluster/mIoU:.2f}",
        save_top_k=3,
        monitor="test/cluster/mIoU",
        mode="max",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    # 当使用epoch验证时，传入check_val_every_n_epoch参数
    trainer_kwargs = dict(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        max_epochs=-1,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )
    if eval_every_n_epochs is not None:
        trainer_kwargs['check_val_every_n_epoch'] = eval_every_n_epochs
    trainer = Trainer(**trainer_kwargs, **gpu_args)
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
