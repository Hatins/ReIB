import torch
import sys
import argparse
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, PercentilePruner
import subprocess
import re
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm
import shutil
import os
import signal


def create_code_snapshot(work_dir, snapshot_dir, search_name):
    """创建代码快照，确保所有trial使用相同版本的代码"""
    snapshot_path = Path(snapshot_dir) / search_name
    
    # 如果快照已存在，先删除
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    
    # 创建快照目录
    snapshot_path.mkdir(parents=True)
    
    # 复制SCON目录下的src代码
    src_dir = Path(work_dir) / "SCON" / "src"
    snapshot_src = snapshot_path / "SCON" / "src"
    
    print(f"\n📸 Creating code snapshot...")
    print(f"  Source: {src_dir}")
    print(f"  Snapshot: {snapshot_src}")
    
    # 复制整个src目录
    shutil.copytree(src_dir, snapshot_src, 
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.git*', 'wandb'))
    
    print(f"  ✓ Code snapshot created\n")
    return snapshot_path


def objective(trial: Trial, args, base_config, search_name, study, snapshot_dir):
    # Trial 0 使用baseline参数（从配置文件读取当前值）
    if trial.number == 0:
        # 从配置文件读取baseline参数
        baseline_pos_inter = float(base_config.get('pos_inter_weight', 0.25))
        baseline_pos_intra = float(base_config.get('pos_intra_weight', 0.67))
        baseline_neg_inter = float(base_config.get('neg_inter_weight', 0.63))
        baseline_neg_inter_shift = float(base_config.get('neg_inter_shift', 0.46))
        baseline_pos_inter_shift = float(base_config.get('pos_inter_shift', 0.02))
        baseline_pos_intra_shift = float(base_config.get('pos_intra_shift', 0.08))
        baseline_recon_hidden_dim = int(base_config.get('recon_hidden_dim', 256))
        baseline_soft_cluster_alpha = float(base_config.get('soft_cluster_alpha', 5.0))
        baseline_stego_loss_weight = float(base_config.get('stego_loss_weight', 0.02))
        baseline_recon_loss_weight = float(base_config.get('recon_loss_weight', 0.8))
        baseline_recon_lr = float(base_config.get('recon_lr', 0.001))
        baseline_soft_cluster_lr = float(base_config.get('soft_cluster_lr', 0.0003))
        baseline_lr = float(base_config.get('lr', 0.0001))
        
        print(f"\n{'='*60}")
        print(f"Trial 0 - Using baseline from config file:")
        print(f"  pos_inter_weight: {baseline_pos_inter}")
        print(f"  pos_intra_weight: {baseline_pos_intra}")
        print(f"  neg_inter_weight: {baseline_neg_inter}")
        print(f"  neg_inter_shift: {baseline_neg_inter_shift}")
        print(f"  pos_inter_shift: {baseline_pos_inter_shift}")
        print(f"  pos_intra_shift: {baseline_pos_intra_shift}")
        print(f"  recon_hidden_dim: {baseline_recon_hidden_dim}")
        print(f"  soft_cluster_alpha: {baseline_soft_cluster_alpha}")
        print(f"  stego_loss_weight: {baseline_stego_loss_weight}")
        print(f"  recon_loss_weight: {baseline_recon_loss_weight}")
        print(f"  recon_lr: {baseline_recon_lr}")
        print(f"  soft_cluster_lr: {baseline_soft_cluster_lr}")
        print(f"  lr: {baseline_lr}")
        print(f"{'='*60}\n")
        
        # 使用固定值，但通过suggest记录到Optuna
        pos_inter_weight = trial.suggest_float('pos_inter_weight', baseline_pos_inter, baseline_pos_inter)
        pos_intra_weight = trial.suggest_float('pos_intra_weight', baseline_pos_intra, baseline_pos_intra)
        neg_inter_weight = trial.suggest_float('neg_inter_weight', baseline_neg_inter, baseline_neg_inter)
        neg_inter_shift = trial.suggest_float('neg_inter_shift', baseline_neg_inter_shift, baseline_neg_inter_shift)
        pos_inter_shift = trial.suggest_float('pos_inter_shift', baseline_pos_inter_shift, baseline_pos_inter_shift)
        pos_intra_shift = trial.suggest_float('pos_intra_shift', baseline_pos_intra_shift, baseline_pos_intra_shift)
        recon_hidden_dim = baseline_recon_hidden_dim
        trial.set_user_attr('recon_hidden_dim', baseline_recon_hidden_dim)
        soft_cluster_alpha = trial.suggest_float('soft_cluster_alpha', baseline_soft_cluster_alpha, baseline_soft_cluster_alpha)
        stego_loss_weight = trial.suggest_float('stego_loss_weight', baseline_stego_loss_weight, baseline_stego_loss_weight)
        recon_loss_weight = trial.suggest_float('recon_loss_weight', baseline_recon_loss_weight, baseline_recon_loss_weight)
        recon_lr = trial.suggest_float('recon_lr', baseline_recon_lr, baseline_recon_lr, log=True)
        soft_cluster_lr = trial.suggest_float('soft_cluster_lr', baseline_soft_cluster_lr, baseline_soft_cluster_lr, log=True)
        lr = trial.suggest_float('lr', baseline_lr, baseline_lr, log=True)
    else:
        # 搜索范围
        pos_inter_weight = trial.suggest_float('pos_inter_weight', 0.0, 1.0)
        pos_intra_weight = trial.suggest_float('pos_intra_weight', 0.0, 1.0)
        neg_inter_weight = trial.suggest_float('neg_inter_weight', 0.0, 1.0)
        neg_inter_shift = trial.suggest_float('neg_inter_shift', 0.0, 1.0)
        pos_inter_shift = trial.suggest_float('pos_inter_shift', 0.0, 1.0)
        pos_intra_shift = trial.suggest_float('pos_intra_shift', 0.0, 1.0)
        recon_hidden_dim = trial.suggest_categorical('recon_hidden_dim', [64, 128, 256, 384, 512, 675])
        soft_cluster_alpha = trial.suggest_float('soft_cluster_alpha', 0.1, 8.0)
        stego_loss_weight = trial.suggest_float('stego_loss_weight', 0.001, 1.0)
        recon_loss_weight = trial.suggest_float('recon_loss_weight', 0.001, 1.0)
        recon_lr = trial.suggest_float('recon_lr', 1e-4, 1e-2, log=True)
        soft_cluster_lr = trial.suggest_float('soft_cluster_lr', 1e-4, 1e-2, log=True)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"  pos_inter_weight: {pos_inter_weight:.4f}")
    print(f"  pos_intra_weight: {pos_intra_weight:.4f}")
    print(f"  neg_inter_weight: {neg_inter_weight:.4f}")
    print(f"  neg_inter_shift: {neg_inter_shift:.4f}")
    print(f"  pos_inter_shift: {pos_inter_shift:.4f}")
    print(f"  pos_intra_shift: {pos_intra_shift:.4f}")
    print(f"  recon_hidden_dim: {recon_hidden_dim}")
    print(f"  soft_cluster_alpha: {soft_cluster_alpha:.4f}")
    print(f"  stego_loss_weight: {stego_loss_weight:.4f}")
    print(f"  recon_loss_weight: {recon_loss_weight:.4f}")
    print(f"  recon_lr: {recon_lr:.2e}")
    print(f"  soft_cluster_lr: {soft_cluster_lr:.2e}")
    print(f"  lr: {lr:.2e}")
    print(f"{'='*60}\n")
    
    # 设置环境变量
    os.environ['OPTUNA_POS_INTER_WEIGHT'] = str(pos_inter_weight)
    os.environ['OPTUNA_POS_INTRA_WEIGHT'] = str(pos_intra_weight)
    os.environ['OPTUNA_NEG_INTER_WEIGHT'] = str(neg_inter_weight)
    os.environ['OPTUNA_NEG_INTER_SHIFT'] = str(neg_inter_shift)
    os.environ['OPTUNA_POS_INTER_SHIFT'] = str(pos_inter_shift)
    os.environ['OPTUNA_POS_INTRA_SHIFT'] = str(pos_intra_shift)
    os.environ['OPTUNA_RECON_HIDDEN_DIM'] = str(recon_hidden_dim)
    os.environ['OPTUNA_SOFT_CLUSTER_ALPHA'] = str(soft_cluster_alpha)
    os.environ['OPTUNA_STEGO_LOSS_WEIGHT'] = str(stego_loss_weight)
    os.environ['OPTUNA_RECON_LOSS_WEIGHT'] = str(recon_loss_weight)
    os.environ['OPTUNA_RECON_LR'] = str(recon_lr)
    os.environ['OPTUNA_SOFT_CLUSTER_LR'] = str(soft_cluster_lr)
    os.environ['OPTUNA_LR'] = str(lr)
    os.environ['OPTUNA_MAX_STEPS'] = str(args.max_steps)
    os.environ['OPTUNA_EXPERIMENT_NAME'] = f"{search_name}/trial_{trial.number}"
    os.environ['OPTUNA_EVAL_EVERY_N_EPOCHS'] = str(args.eval_every_n_epochs)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['OPTUNA_SILENT_TRAIN'] = '1'
    
    print(f"  Settings: max_steps={args.max_steps}, eval_every_n_epochs={args.eval_every_n_epochs}")
    
    # 使用快照目录中的代码
    snapshot_path = Path(snapshot_dir) / search_name
    train_script = snapshot_path / "SCON" / "src" / "train_segmentation.py"
    cmd = ['python', str(train_script), '--config-name=train_config_potsdam.yml']
    
    print(f"Training trial {trial.number}...")
    import time
    start_time = time.time()
    last_step = 0
    mious = []
    accs = []
    reported_steps = set()
    
    # 创建进度条
    pbar = tqdm(total=args.max_steps, 
                desc=f"Trial {trial.number}", 
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] {postfix}',
                ncols=100)
    
    try:
        process = subprocess.Popen(cmd, cwd=args.work_dir, stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  preexec_fn=os.setsid)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_str = output.strip()
                
                # 捕获训练进度
                if 'Training:' in output_str or 'global_step=' in output_str:
                    step_match = re.search(r'global_step=(\d+)', output_str)
                    if step_match:
                        current_step = int(step_match.group(1))
                        if current_step > last_step:
                            pbar.update(current_step - last_step)
                            last_step = current_step
                
                # 捕获验证结果
                metric_match = re.search(r'OPTUNA_METRIC: step=(\d+), mIoU=([\d.]+), Acc=([\d.]+)', output_str)
                if metric_match:
                    step = int(metric_match.group(1))
                    miou = float(metric_match.group(2))
                    acc = float(metric_match.group(3))
                    mious.append(miou)
                    accs.append(acc)
                    
                    # 计算组合指标：mIoU和Acc的加权平均 (各占50%)
                    combined_metric = 0.5 * miou + 0.5 * acc
                    max_miou = max(mious)
                    max_acc = max(accs)
                    max_combined = max([0.5 * m + 0.5 * a for m, a in zip(mious, accs)])
                    
                    pbar.set_postfix_str(f"mIoU={miou:.4f}(max={max_miou:.4f}), Acc={acc:.4f}(max={max_acc:.4f}), Combined={combined_metric:.4f}")
                    tqdm.write(f"  ✓ Step {step}: mIoU={miou:.4f}, Acc={acc:.4f}, Combined={combined_metric:.4f} (Max Combined={max_combined:.4f})")
                    
                    # 报告组合指标用于剪枝判断
                    if step not in reported_steps:
                        trial.report(combined_metric, step)
                        reported_steps.add(step)
                        
                        # 检查是否应该剪枝
                        if trial.should_prune():
                            tqdm.write(f"  ✂ Trial {trial.number} pruned at step {step} (Combined={combined_metric:.4f})")
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            pbar.close()
                            raise optuna.TrialPruned()
        
        pbar.close()
        if process.returncode != 0:
            print(f"Training failed with return code: {process.returncode}")
            return 0.0
    except subprocess.TimeoutExpired:
        pbar.close()
        print(f"\nTraining timeout")
        return 0.0
    except optuna.TrialPruned:
        raise
    except Exception as e:
        pbar.close()
        print(f"\nError: {e}")
        return 0.0
    
    if not mious:
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} COMPLETED")
        print(f"  Result: No valid evaluations found")
        print(f"  Max Combined Value: 0.0000")
        print(f"{'='*60}\n")
        max_combined = 0.0
    else:
        max_miou = max(mious)
        max_acc = max(accs)
        # 计算最大组合指标
        max_combined = max([0.5 * m + 0.5 * a for m, a in zip(mious, accs)])
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} COMPLETED")
        print(f"  Max mIoU: {max_miou:.4f}")
        print(f"  Max Acc: {max_acc:.4f}")
        print(f"  Max Combined (0.5*mIoU + 0.5*Acc): {max_combined:.4f}")
        print(f"  Evaluations: {len(mious)}")
        print(f"  mIoU scores: {[f'{m:.4f}' for m in mious]}")
        print(f"  Acc scores: {[f'{a:.4f}' for a in accs]}")
        print(f"{'='*60}\n")
    
    # 显示当前最佳结果
    try:
        best_trial = study.best_trial
        print(f"\n{'='*60}")
        print(f"CURRENT BEST TRIAL: #{best_trial.number}")
        print(f"  Best Combined Score (0.5*mIoU + 0.5*Acc): {best_trial.value:.4f}")
        print(f"  Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.4f}")
        if 'recon_hidden_dim' in best_trial.user_attrs:
            print(f"    recon_hidden_dim: {best_trial.user_attrs['recon_hidden_dim']}")
        print(f"{'='*60}\n")
    except ValueError:
        pass
    
    # 清理当前trial的所有checkpoints（默认开启）
    if not args.keep_checkpoints and max_combined > 0:
        cleanup_trial_checkpoints(args, search_name, trial.number)
    
    return max_combined


def cleanup_trial_checkpoints(args, search_name, trial_number):
    """清理trial的所有checkpoints"""
    import shutil
    
    checkpoint_dir = Path(args.work_dir) / "SCON" / "checkpoints" / search_name / f"trial_{trial_number}"
    
    if not checkpoint_dir.exists():
        return
    
    # 找到所有checkpoint文件
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    
    if len(checkpoint_files) == 0:
        return
    
    # 删除所有checkpoint文件
    deleted_count = 0
    deleted_size = 0
    for ckpt_file in checkpoint_files:
        file_size = ckpt_file.stat().st_size / (1024 * 1024)  # MB
        ckpt_file.unlink()
        deleted_count += 1
        deleted_size += file_size
    
    # 删除空目录
    try:
        checkpoint_dir.rmdir()
    except:
        pass
    
    if deleted_count > 0:
        print(f"  🧹 Cleaned up {deleted_count} checkpoints, freed {deleted_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=5400)
    parser.add_argument('--eval_every_n_epochs', type=int, default=3, help='Evaluate every N epochs')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--keep_checkpoints', action='store_true', help='Keep checkpoints after each trial')
    parser.add_argument('--n_startup_trials', type=int, default=5, help='Number of trials before pruning starts')
    parser.add_argument('--n_warmup_steps', type=int, default=0, help='Number of validation steps before pruning can happen')
    
    args = parser.parse_args()
    
    # 固定使用potsdam配置
    config_dir = 'SCON/src/configs'
    config_name = 'train_config_potsdam.yml'
    
    base_config_path = Path(config_dir) / config_name
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 创建搜索名称
    dino_version = base_config.get('dino_version', 'v3')
    model_type = base_config.get('model_type', 'vit_small')
    dino_patch_size = base_config.get('dino_patch_size', 16)
    dataset_name = 'potsdam'
    
    # 创建代码快照目录
    snapshot_dir = Path(args.work_dir) / "SCON" / "optuna_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    # 提取model名称
    model_name = 'small' if 'small' in model_type else 'base'
    
    search_name = f"search_{dataset_name}_scon_dino{dino_version}_{model_name}_patch{dino_patch_size}_epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建代码快照
    snapshot_path = create_code_snapshot(args.work_dir, snapshot_dir, search_name)
    
    # 创建独立的数据库文件
    db_dir = Path(args.work_dir) / "SCON" / "optuna_db"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / f"{search_name}.db"
    storage = f"sqlite:///{db_path}"
    
    print(f"\n{'='*60}")
    print(f"Starting Hyperparameter Search - SCON Potsdam")
    print(f"  Search Name: {search_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  DINO Version: {dino_version}")
    print(f"  Model Type: {model_type} (patch_size={dino_patch_size})")
    print(f"  Trials: {args.n_trials}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Eval Every N Epochs: {args.eval_every_n_epochs}")
    print(f"  Pruning: Enabled (startup_trials={args.n_startup_trials}, warmup_steps={args.n_warmup_steps})")
    print(f"  Database: {db_path}")
    print(f"{'='*60}\n")
    
    # 创建study
    study = optuna.create_study(
        study_name=search_name,
        storage=storage,
        load_if_exists=False,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=PercentilePruner(
            percentile=50.0,
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=1
        )
    )
    
    study.optimize(lambda trial: objective(trial, args, base_config, search_name, study, snapshot_dir), n_trials=args.n_trials)
    
    # 搜索完成后，清理快照
    print(f"\n🧹 Cleaning up code snapshot...")
    snapshot_path = Path(snapshot_dir) / search_name
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
        print(f"  ✓ Snapshot removed: {snapshot_path}")
    
    print(f"\n{'='*60}")
    print("Best trial:")
    print(f"  Combined Score (0.5*mIoU + 0.5*Acc): {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value:.4f}")
    if 'recon_hidden_dim' in study.best_trial.user_attrs:
        print(f"    recon_hidden_dim: {study.best_trial.user_attrs['recon_hidden_dim']}")
    print(f"{'='*60}\n")
    
    # 保存结果
    result_filename = f"best_params_{dataset_name}_scon_dino{dino_version}_{model_name}_patch{dino_patch_size}_epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    result_path = Path(config_dir) / result_filename
    with open(result_path, 'w') as f:
        f.write(f"Search: {search_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"DINO Version: {dino_version}\n")
        f.write(f"Model Type: {model_type} (patch_size={dino_patch_size})\n")
        f.write(f"Evaluation: Every {args.eval_every_n_epochs} epoch(s)\n\n")
        f.write(f"Optimization Metric: Combined (0.5*mIoU + 0.5*Acc)\n")
        f.write(f"Best Combined Score: {study.best_trial.value:.4f}\n\n")
        f.write("Best Parameters (Full Precision):\n")
        for key, value in study.best_trial.params.items():
            if isinstance(value, int):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
        if 'recon_hidden_dim' in study.best_trial.user_attrs:
            f.write(f"recon_hidden_dim: {study.best_trial.user_attrs['recon_hidden_dim']}\n")
    
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
