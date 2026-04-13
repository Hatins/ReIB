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


def create_code_snapshot(work_dir, snapshot_dir, search_name):
    snapshot_path = Path(snapshot_dir) / search_name
    
    # 如果快照已存在，先删除
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    
    # 创建快照目录
    snapshot_path.mkdir(parents=True)
    
    # 复制SCON2目录下的src代码
    src_dir = Path(work_dir) / "SCON2" / "src"
    snapshot_src = snapshot_path / "SCON2" / "src"
    
    print(f"\n📸 Creating code snapshot...")
    print(f"  Source: {src_dir}")
    print(f"  Snapshot: {snapshot_src}")
    
    # 复制整个src目录
    shutil.copytree(src_dir, snapshot_src, 
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.git*', 'wandb'))
    
    # 也需要复制配置文件
    config_src = Path(work_dir) / "SCON2" / "src" / "configs"
    config_dst = snapshot_path / "SCON2" / "src" / "configs"
    if config_src.exists() and not config_dst.exists():
        shutil.copytree(config_src, config_dst)
    
    print(f"  ✓ Code snapshot created\n")
    return snapshot_path

def objective(trial: Trial, args, base_config, search_name, study, snapshot_dir):
    # 从配置文件读取dim，不再搜索（保持与配置文件一致）
    dim = int(base_config.get('dim', 128))
    
    # Trial 0 使用baseline参数（从配置文件读取当前值）
    if trial.number == 0:
        # STEGO原有参数
        baseline_pos_inter = base_config.get('pos_inter_weight', 0.25)
        baseline_pos_intra = base_config.get('pos_intra_weight', 0.67)
        baseline_neg_inter = base_config.get('neg_inter_weight', 0.63)
        baseline_neg_inter_shift = base_config.get('neg_inter_shift', 0.46)
        baseline_pos_inter_shift = base_config.get('pos_inter_shift', 0.02)
        baseline_pos_intra_shift = base_config.get('pos_intra_shift', 0.08)
        
        # SCON2新增参数
        baseline_recon_hidden_dim = int(base_config.get('recon_hidden_dim', 128))
        baseline_soft_cluster_alpha = float(base_config.get('soft_cluster_alpha', 1.0))
        baseline_stego_loss_weight = float(base_config.get('stego_loss_weight', 1.0))
        baseline_recon_loss_weight = float(base_config.get('recon_loss_weight', 1.0))
        baseline_recon_lr = float(base_config.get('recon_lr', 5e-3))
        baseline_soft_cluster_lr = float(base_config.get('soft_cluster_lr', 5e-3))
        
        print(f"\n{'='*60}")
        print(f"Trial 0 - Using baseline from config file:")
        print(f"  STEGO Parameters:")
        print(f"    pos_inter_weight: {baseline_pos_inter}")
        print(f"    pos_intra_weight: {baseline_pos_intra}")
        print(f"    neg_inter_weight: {baseline_neg_inter}")
        print(f"    neg_inter_shift: {baseline_neg_inter_shift}")
        print(f"    pos_inter_shift: {baseline_pos_inter_shift}")
        print(f"    pos_intra_shift: {baseline_pos_intra_shift}")
        print(f"    dim: {dim} (fixed)")
        print(f"  SCON2 Reconstruction Parameters:")
        print(f"    recon_hidden_dim: {baseline_recon_hidden_dim}")
        print(f"    soft_cluster_alpha: {baseline_soft_cluster_alpha}")
        print(f"    stego_loss_weight: {baseline_stego_loss_weight}")
        print(f"    recon_loss_weight: {baseline_recon_loss_weight}")
        print(f"    recon_lr: {baseline_recon_lr}")
        print(f"    soft_cluster_lr: {baseline_soft_cluster_lr}")
        print(f"{'='*60}\n")
        
        # STEGO参数
        pos_inter_weight = trial.suggest_float('pos_inter_weight', baseline_pos_inter, baseline_pos_inter)
        pos_intra_weight = trial.suggest_float('pos_intra_weight', baseline_pos_intra, baseline_pos_intra)
        neg_inter_weight = trial.suggest_float('neg_inter_weight', baseline_neg_inter, baseline_neg_inter)
        neg_inter_shift = trial.suggest_float('neg_inter_shift', baseline_neg_inter_shift, baseline_neg_inter_shift)
        pos_inter_shift = trial.suggest_float('pos_inter_shift', baseline_pos_inter_shift, baseline_pos_inter_shift)
        pos_intra_shift = trial.suggest_float('pos_intra_shift', baseline_pos_intra_shift, baseline_pos_intra_shift)
        
        # SCON2参数 - Trial 0 直接使用baseline值，避免与其他trial的categorical分布冲突
        recon_hidden_dim = baseline_recon_hidden_dim
        trial.set_user_attr('recon_hidden_dim', baseline_recon_hidden_dim)
        soft_cluster_alpha = trial.suggest_float('soft_cluster_alpha', baseline_soft_cluster_alpha, baseline_soft_cluster_alpha)
        stego_loss_weight = trial.suggest_float('stego_loss_weight', baseline_stego_loss_weight, baseline_stego_loss_weight)
        recon_loss_weight = trial.suggest_float('recon_loss_weight', baseline_recon_loss_weight, baseline_recon_loss_weight)
        recon_lr = trial.suggest_float('recon_lr', baseline_recon_lr, baseline_recon_lr, log=True)
        soft_cluster_lr = trial.suggest_float('soft_cluster_lr', baseline_soft_cluster_lr, baseline_soft_cluster_lr, log=True)
    else:
        # STEGO参数
        pos_inter_weight = trial.suggest_float('pos_inter_weight', 0.0, 1.0)
        pos_intra_weight = trial.suggest_float('pos_intra_weight', 0.0, 1.0)
        neg_inter_weight = trial.suggest_float('neg_inter_weight', 0.0, 1.0)
        neg_inter_shift = trial.suggest_float('neg_inter_shift', 0.0, 1.0)
        pos_inter_shift = trial.suggest_float('pos_inter_shift', 0.0, 1.0)
        pos_intra_shift = trial.suggest_float('pos_intra_shift', 0.0, 1.0)
        
        # SCON2参数
        recon_hidden_dim = trial.suggest_categorical('recon_hidden_dim', [64, 128, 256, 384])
        soft_cluster_alpha = trial.suggest_float('soft_cluster_alpha', 0.1, 5.0)
        stego_loss_weight = trial.suggest_float('stego_loss_weight', 0.001, 1.0)
        recon_loss_weight = trial.suggest_float('recon_loss_weight', 0.001, 1.0)
        recon_lr = trial.suggest_float('recon_lr', 1e-4, 1e-2, log=True)
        soft_cluster_lr = trial.suggest_float('soft_cluster_lr', 1e-4, 1e-2, log=True)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"  STEGO Parameters:")
    print(f"    pos_inter_weight: {pos_inter_weight:.4f}")
    print(f"    pos_intra_weight: {pos_intra_weight:.4f}")
    print(f"    neg_inter_weight: {neg_inter_weight:.4f}")
    print(f"    neg_inter_shift: {neg_inter_shift:.4f}")
    print(f"    pos_inter_shift: {pos_inter_shift:.4f}")
    print(f"    pos_intra_shift: {pos_intra_shift:.4f}")
    print(f"    dim: {dim} (fixed)")
    print(f"  SCON2 Reconstruction Parameters:")
    print(f"    recon_hidden_dim: {recon_hidden_dim}")
    print(f"    soft_cluster_alpha: {soft_cluster_alpha:.4f}")
    print(f"    stego_loss_weight: {stego_loss_weight:.4f}")
    print(f"    recon_loss_weight: {recon_loss_weight:.4f}")
    print(f"    recon_lr: {recon_lr:.6f}")
    print(f"    soft_cluster_lr: {soft_cluster_lr:.6f}")
    print(f"{'='*60}\n")
    
    import os
    # STEGO参数
    os.environ['OPTUNA_POS_INTER_WEIGHT'] = str(pos_inter_weight)
    os.environ['OPTUNA_POS_INTRA_WEIGHT'] = str(pos_intra_weight)
    os.environ['OPTUNA_NEG_INTER_WEIGHT'] = str(neg_inter_weight)
    os.environ['OPTUNA_NEG_INTER_SHIFT'] = str(neg_inter_shift)
    os.environ['OPTUNA_POS_INTER_SHIFT'] = str(pos_inter_shift)
    os.environ['OPTUNA_POS_INTRA_SHIFT'] = str(pos_intra_shift)
    os.environ['OPTUNA_DIM'] = str(dim)
    
    # SCON2参数
    os.environ['OPTUNA_RECON_HIDDEN_DIM'] = str(recon_hidden_dim)
    os.environ['OPTUNA_SOFT_CLUSTER_ALPHA'] = str(soft_cluster_alpha)
    os.environ['OPTUNA_STEGO_LOSS_WEIGHT'] = str(stego_loss_weight)
    os.environ['OPTUNA_RECON_LOSS_WEIGHT'] = str(recon_loss_weight)
    os.environ['OPTUNA_RECON_LR'] = str(recon_lr)
    os.environ['OPTUNA_SOFT_CLUSTER_LR'] = str(soft_cluster_lr)
    
    os.environ['OPTUNA_MAX_STEPS'] = str(args.max_steps)
    os.environ['OPTUNA_EXPERIMENT_NAME'] = f"{search_name}/trial_{trial.number}"
    os.environ['OPTUNA_VAL_FREQ'] = str(args.eval_interval)
    os.environ['WANDB_MODE'] = 'disabled'
    
    print(f"  Settings: max_steps={args.max_steps}, val_freq={args.eval_interval}")
    
    # 使用快照目录中的代码
    snapshot_path = Path(snapshot_dir) / search_name
    train_script = snapshot_path / "SCON2" / "src" / "train_segmentation.py"
    cmd = ['python', str(train_script)]
    
    print(f"Training trial {trial.number}...")
    import time
    start_time = time.time()
    last_step = 0
    mious = []  # 直接从stdout收集mIoU
    reported_steps = set()  # 记录已经报告过的steps，避免重复
    
    # 为每个trial创建日志文件
    log_dir = Path(args.work_dir) / "SCON2" / "logs" / "optuna_trials"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{search_name}_trial_{trial.number}.log"
    
    # 创建进度条
    pbar = tqdm(total=args.max_steps, 
                desc=f"Trial {trial.number}", 
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] {postfix}',
                ncols=100)
    
    error_logs = []  # 收集错误信息
    all_output = []  # 收集所有输出用于调试
    
    try:
        process = subprocess.Popen(cmd, cwd=args.work_dir, stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        with open(log_file, 'w') as f:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_str = output.strip()
                    all_output.append(output_str)
                    f.write(output_str + '\n')
                    f.flush()
                    
                    # 捕获错误信息
                    if any(keyword in output_str.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'cuda out of memory']):
                        error_logs.append(output_str)
                        tqdm.write(f"  ⚠️  {output_str}")
                    
                    # 捕获训练进度 - 匹配Training输出或其他格式
                    # 优先匹配 "Training: global_step=XXX"
                    if 'Training:' in output_str or 'global_step=' in output_str:
                        step_match = re.search(r'global_step=(\d+)', output_str)
                        if step_match:
                            current_step = int(step_match.group(1))
                            if current_step > last_step:
                                pbar.update(current_step - last_step)
                                last_step = current_step
                    
                    metric_match = re.search(r'OPTUNA_METRIC: step=(\d+), mIoU=([\d.]+)', output_str)
                    if metric_match:
                        step = int(metric_match.group(1))
                        miou = float(metric_match.group(2))
                        mious.append(miou)
                        max_miou = max(mious)
                        pbar.set_postfix_str(f"mIoU: latest={miou:.4f}, max={max_miou:.4f}")
                        tqdm.write(f"  ✓ Validation at step {step}: mIoU={miou:.4f}, Max={max_miou:.4f}")
                        
                        # 报告中间值用于剪枝判断（避免重复报告同一step）
                        if step not in reported_steps:
                            trial.report(miou, step)
                            reported_steps.add(step)
                            
                            # 检查是否应该剪枝
                            if trial.should_prune():
                                tqdm.write(f"  ✂ Trial {trial.number} pruned at step {step} (mIoU={miou:.4f})")
                                process.terminate()
                                pbar.close()
                                raise optuna.TrialPruned()
        
        pbar.close()
        
        if process.returncode != 0:
            print(f"\n{'='*60}")
            print(f"❌ Trial {trial.number} FAILED")
            print(f"  Return code: {process.returncode}")
            print(f"  Log file: {log_file}")
            if error_logs:
                print(f"\n  Error messages captured:")
                for i, err in enumerate(error_logs[-10:], 1):  # 显示最后10个错误
                    print(f"    {i}. {err}")
            else:
                print(f"\n  No explicit error messages found. Last 10 lines of output:")
                for line in all_output[-10:]:
                    print(f"    {line}")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Training process failed with return code {process.returncode}")
            
    except optuna.TrialPruned:
        # 这是正常的剪枝，直接重新抛出
        raise
    except subprocess.TimeoutExpired:
        pbar.close()
        print(f"\n{'='*60}")
        print(f"⏱️  Trial {trial.number} TIMEOUT")
        print(f"  Exceeded maximum time limit")
        print(f"  Log file: {log_file}")
        print(f"{'='*60}\n")
        raise RuntimeError("Training timeout")
    except Exception as e:
        pbar.close()
        print(f"\n{'='*60}")
        print(f"💥 Trial {trial.number} EXCEPTION")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {str(e)}")
        print(f"  Log file: {log_file}")
        if error_logs:
            print(f"\n  Error messages from training:")
            for i, err in enumerate(error_logs[-10:], 1):
                print(f"    {i}. {err}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise
    
    if not mious:
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} COMPLETED")
        print(f"  Result: No valid evaluations found")
        print(f"  Max Value: 0.0000")
        print(f"{'='*60}\n")
        max_miou = 0.0
    else:
        max_miou = max(mious)
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} COMPLETED")
        print(f"  Max Value: {max_miou:.4f}")
        print(f"  Evaluations: {len(mious)}")
        print(f"  Individual scores: {[f'{m:.4f}' for m in mious]}")
        print(f"{'='*60}\n")
    
    # 显示当前最佳结果（需要至少有一个完成的trial）
    try:
        best_trial = study.best_trial
        print(f"\n{'='*60}")
        print(f"CURRENT BEST TRIAL: #{best_trial.number}")
        print(f"  Best mIoU: {best_trial.value:.4f}")
        print(f"  Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.4f}")
        # 如果dim是固定的，额外显示
        if 'dim_fixed' in best_trial.user_attrs and best_trial.user_attrs['dim_fixed']:
            # dim不在params中，从user_attrs获取
            if 'actual_dim' in best_trial.user_attrs:
                print(f"    dim: {best_trial.user_attrs['actual_dim']} (fixed)")
        print(f"{'='*60}\n")
    except ValueError:
        # 第一个trial还没完成时会触发这个异常
        pass
    
    # 清理当前trial的所有checkpoints（默认开启）
    if not args.keep_checkpoints and max_miou > 0:
        cleanup_trial_checkpoints(args, search_name, trial.number)
    
    # 记录实际使用的dim值（固定为128）
    trial.set_user_attr('actual_dim', dim)
    
    return max_miou

def cleanup_trial_checkpoints(args, search_name, trial_number):
    """清理trial的所有checkpoints"""
    import shutil
    
    checkpoint_dir = Path(args.work_dir) / "SCON2" / "checkpoints" / search_name / f"trial_{trial_number}"
    
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
    parser.add_argument('--n_trials', type=int, default=300)
    parser.add_argument('--max_steps', type=int, default=60000)
    parser.add_argument('--eval_interval', type=int, default=600)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--config_dir', type=str, default='SCON2/src/configs')
    parser.add_argument('--config_name', type=str, default='train_config.yml', help='Config file name (e.g., train_config.yml)')
    parser.add_argument('--keep_checkpoints', action='store_true', help='Keep checkpoints after each trial (by default, all checkpoints are deleted to save disk space)')
    parser.add_argument('--n_startup_trials', type=int, default=5, help='Number of trials before pruning starts')
    parser.add_argument('--n_warmup_steps', type=int, default=3, help='Number of intermediate reports before pruning can happen (e.g., 2 means pruning starts from the 3rd validation)')
    
    args = parser.parse_args()
    
    base_config_path = Path(args.config_dir) / args.config_name
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 创建搜索名称
    dino_version = base_config.get('dino_version', 'v1')
    model_type = base_config.get('model_type', 'vit_small')
    dino_patch_size = base_config.get('dino_patch_size', 8)
    
    # 创建代码快照目录（放在SCON2文件夹下）
    snapshot_dir = Path(args.work_dir) / "SCON2" / "optuna_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    # 提取model名称（small/base）
    model_name = 'small' if 'small' in model_type else 'base'
    
    search_name = f"search_scon2_dino{dino_version}_{model_name}_patch{dino_patch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建代码快照（冻结代码版本）
    snapshot_path = create_code_snapshot(args.work_dir, snapshot_dir, search_name)
    
    # 创建独立的数据库文件（放在SCON2文件夹下），确保不同搜索使用不同的存储
    db_dir = Path(args.work_dir) / "SCON2" / "optuna_db"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / f"{search_name}.db"
    storage = f"sqlite:///{db_path}"
    
    print(f"\n{'='*60}")
    print(f"Starting Hyperparameter Search")
    print(f"  Search Name: {search_name}")
    print(f"  DINO Version: {dino_version}")
    print(f"  Model Type: {model_type} (patch_size={dino_patch_size})")
    print(f"  Trials: {args.n_trials}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Eval Interval: {args.eval_interval}")
    print(f"  Pruning: Enabled (startup_trials={args.n_startup_trials}, warmup_steps={args.n_warmup_steps})")
    print(f"  Database: {db_path}")
    print(f"{'='*60}\n")
    

    study = optuna.create_study(
        study_name=search_name,      # 使用唯一的study名称
        storage=storage,              # 使用独立的SQLite数据库
        load_if_exists=False,         # 不加载已存在的study，确保每次都是新的
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=PercentilePruner(
            percentile=50.0,         # 50%分位数（中位数，接近平均值）
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=1         # 每次验证都检查是否剪枝
        )
    )
    
    # 定义objective wrapper来处理异常
    def objective_wrapper(trial):
        try:
            return objective(trial, args, base_config, search_name, study, snapshot_dir)
        except optuna.TrialPruned:
            # 正常剪枝，重新抛出
            raise
        except Exception as e:
            # 记录失败的trial
            print(f"\n⚠️  Trial {trial.number} failed with exception: {e}")
            print(f"   Trial will be marked as FAILED and Optuna will continue with next trial\n")
            # 抛出异常让Optuna记录这个trial为FAIL状态
            raise
    
    study.optimize(objective_wrapper, n_trials=args.n_trials, catch=(Exception,))
    
    # 搜索完成后，清理快照
    print(f"\n🧹 Cleaning up code snapshot...")
    snapshot_path = Path(snapshot_dir) / search_name
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
        print(f"  ✓ Snapshot removed: {snapshot_path}")
    
    # 统计所有trials的状态
    print(f"\n{'='*60}")
    print("Search Summary:")
    all_trials = study.trials
    completed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"  Total trials: {len(all_trials)}")
    print(f"  Completed: {len(completed_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Failed: {len(failed_trials)}")
    
    if failed_trials:
        print(f"\n  Failed trials:")
        for t in failed_trials:
            print(f"    Trial {t.number}: {t.params if hasattr(t, 'params') else 'N/A'}")
            # 尝试找到对应的日志文件
            log_file = Path(args.work_dir) / "SCON2" / "logs" / "optuna_trials" / f"{search_name}_trial_{t.number}.log"
            if log_file.exists():
                print(f"      Log: {log_file}")
    
    print(f"{'='*60}\n")
    
    if len(completed_trials) > 0:
        print(f"\n{'='*60}")
        print("Best trial:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print("  Params:")
        for key, value in study.best_trial.params.items():
            # 如果有actual_dim属性，显示实际使用的dim
            if key == 'dim' and 'actual_dim' in study.best_trial.user_attrs:
                actual_dim = study.best_trial.user_attrs['actual_dim']
                print(f"    {key}: {value:.4f} (actual: {actual_dim})")
            else:
                print(f"    {key}: {value:.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("⚠️  No completed trials found!")
        print("   All trials either failed or were pruned.")
        print(f"{'='*60}\n")
        return  # 没有完成的trial就不保存结果文件
    
    # 文件名包含dino版本、模型类型和patch大小
    result_filename = f"best_params_dino{dino_version}_{model_name}_patch{dino_patch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    result_path = Path(args.config_dir) / result_filename
    with open(result_path, 'w') as f:
        f.write(f"Search: {search_name}\n")
        f.write(f"DINO Version: {dino_version}\n")
        f.write(f"Model Type: {model_type} (patch_size={dino_patch_size})\n\n")
        
        # 添加搜索统计信息
        f.write(f"Search Statistics:\n")
        f.write(f"  Total trials: {len(all_trials)}\n")
        f.write(f"  Completed: {len(completed_trials)}\n")
        f.write(f"  Pruned: {len(pruned_trials)}\n")
        f.write(f"  Failed: {len(failed_trials)}\n\n")
        
        f.write(f"Best mIoU: {study.best_trial.value:.4f}\n\n")
        f.write("Best Parameters:\n")
        for key, value in study.best_trial.params.items():
            # 保存实际使用的dim
            if key == 'dim' and 'actual_dim' in study.best_trial.user_attrs:
                actual_dim = study.best_trial.user_attrs['actual_dim']
                f.write(f"{key}: {actual_dim}  # searched value was {value:.4f}, but fixed to {actual_dim}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
    
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()
