"""
Stability-aware CLAM training script.

This script enables training CLAM models with stability-aware features:
1. Use stability metrics directly as features
2. Filter FM features based on stability

Separate from main.py to avoid modifying core CLAM code.

Example usage:

# Mode 1: Use spatial ICC as features
python main_stability.py \
    --stability_mode features \
    --stability_metric icc \
    --stability_category spatial \
    --stability_dir /path/to/stability_features \
    --data_root_dir /path/to/fm_features \
    --task task_kidney_grade \
    --exp_code kidney_spatial_icc

# Mode 2: Filter features by high overall ICC (top 80%)
python main_stability.py \
    --stability_mode filter \
    --filter_metric icc \
    --filter_category overall \
    --filter_direction high \
    --filter_percentile 80 \
    --stability_dir /path/to/stability_features \
    --data_root_dir /path/to/fm_features \
    --task task_kidney_grade \
    --exp_code kidney_filtered_icc80

# Mode 3: Multi-model with filtering
python main_stability.py \
    --stability_mode filter \
    --filter_metric icc \
    --filter_percentile 80 \
    --models uni_v1,conch_v15 \
    --data_dirs ./features_uni_v1,./features_conch_v15 \
    --stability_dirs ./stability_uni_v1,./stability_conch_v15 \
    --task task_kidney_grade \
    --exp_code multi_model_filtered
"""

from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Internal imports
from utils.file_utils import save_pkl
from utils.core_utils import train
from dataset_modules.dataset_stability import (
    StabilityAwareDataset,
    StabilitySplit,
    StabilityFilter,
    MultiModelStabilityDataset,
)
from dataset_modules.dataset_generic import Generic_MIL_Dataset


# Model embed_dim registry
EMBED_DIM_REGISTRY = {
    'uni_v1': 1024, 'uni_v2': 1536,
    'conch_v1': 512, 'conch_v15': 768,
    'virchow': 2560, 'virchow2': 2560,
    'gigapath': 1536, 'hoptimus0': 1536, 'hoptimus1': 1536,
    'phikon': 768, 'phikon_v2': 1024,
    'resnet50': 1024, 'ctranspath': 768,
    'musk': 1024, 'hibou_l': 1024,
}

VALID_STABILITY_IDS = None


def get_embed_dim(model_name: str) -> int:
    """Get embedding dimension for a model."""
    return EMBED_DIM_REGISTRY.get(model_name, 1024)


def parse_model_configs(args):
    """Parse multi-model configuration from args."""
    if args.models is None:
        # Single model mode (backward compatible)
        return [{
            'name': args.patch_encoder or 'unknown',
            'data_dir': args.data_root_dir,
            'stability_dir': args.stability_dir,
            'embed_dim': args.embed_dim,
        }]
    
    models = args.models.split(',')
    data_dirs = args.data_dirs.split(',') if args.data_dirs else [args.data_root_dir] * len(models)
    stability_dirs = args.stability_dirs.split(',') if args.stability_dirs else [args.stability_dir] * len(models)
    
    if len(models) != len(data_dirs) or len(models) != len(stability_dirs):
        raise ValueError("Number of models, data_dirs, and stability_dirs must match")
    
    return [
        {
            'name': model,
            'data_dir': data_dir,
            'stability_dir': stability_dir,
            'embed_dim': get_embed_dim(model),
        }
        for model, data_dir, stability_dir in zip(models, data_dirs, stability_dirs)
    ]


def create_stability_dataset(args, csv_path: str, label_dict: dict, model_configs: list):
    """Create stability-aware dataset based on args."""
    
    common_kwargs = {
        'csv_path': csv_path,
        'shuffle': False,
        'seed': args.seed,
        'print_info': True,
        'label_dict': label_dict,
        'patient_strat': True,
        'ignore': [],
    }
    
    if len(model_configs) > 1:
        # Multi-model mode
        dataset = MultiModelStabilityDataset(
            model_configs=model_configs,
            stability_mode=args.stability_mode,
            stability_metric=args.stability_metric or args.filter_metric,
            stability_category=args.stability_category or args.filter_category,
            filter_direction=args.filter_direction,
            filter_percentile=args.filter_percentile,
            filter_weighting=args.filter_weighting,
            **common_kwargs
        )
    else:
        # Single model mode
        config = model_configs[0]
        dataset = StabilityAwareDataset(
            stability_dir=config['stability_dir'],
            stability_mode=args.stability_mode,
            stability_metric=args.stability_metric or args.filter_metric,
            stability_category=args.stability_category or args.filter_category,
            filter_direction=args.filter_direction,
            filter_percentile=args.filter_percentile,
            filter_weighting=args.filter_weighting,
            data_dir=config['data_dir'],
            **common_kwargs
        )
    
    return dataset


def create_split(
    slide_data,
    args,
    model_configs: list,
    stability_filter=None,
):
    """Create a split dataset with stability awareness."""
    config = model_configs[0]  # Use first model config for single-model mode
    
    return StabilitySplit(
        slide_data=slide_data,
        stability_dir=config['stability_dir'],
        stability_mode=args.stability_mode,
        stability_metric=args.stability_metric or args.filter_metric,
        stability_category=args.stability_category or args.filter_category,
        filter_direction=args.filter_direction,
        filter_percentile=args.filter_percentile,
        filter_weighting=args.filter_weighting,
        data_dir=config['data_dir'],
        num_classes=args.n_classes,
        stability_filter=stability_filter,
    )


def list_stability_ids(stability_dir: str) -> set:
    """List slide IDs with available stability H5 files."""
    ids = set()
    if not stability_dir or not os.path.isdir(stability_dir):
        return ids
    for entry in os.scandir(stability_dir):
        if entry.is_file() and entry.name.endswith('.h5'):
            ids.add(os.path.splitext(entry.name)[0])
    return ids


def get_valid_stability_ids(model_configs: list) -> set:
    """Get valid slide IDs present in all required stability dirs."""
    stability_dirs = [config['stability_dir'] for config in model_configs]
    valid_ids = None
    for stability_dir in stability_dirs:
        ids = list_stability_ids(stability_dir)
        if valid_ids is None:
            valid_ids = ids
        else:
            valid_ids = valid_ids.intersection(ids)
    return valid_ids if valid_ids is not None else set()


def filter_ids_by_stability(ids_list, valid_ids: set):
    """Filter a list of slide IDs by valid stability IDs."""
    filtered = [sid for sid in ids_list if sid in valid_ids]
    dropped = len(ids_list) - len(filtered)
    return filtered, dropped


def main(args):
    """Main training function."""
    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    
    # Set up fold range
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    
    # Parse model configs
    model_configs = parse_model_configs(args)
    print(f"\nUsing {len(model_configs)} model(s):")
    for config in model_configs:
        print(f"  - {config['name']}: {config['data_dir']}")
    
    # Tracking metrics
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    valid_ids = VALID_STABILITY_IDS or set()
    
    for i in folds:
        seed_torch(args.seed)
        
        # Load splits
        splits_path = os.path.join(args.split_dir, f'splits_{i}.csv')
        all_splits = pd.read_csv(splits_path, dtype=dataset.slide_data['slide_id'].dtype)
        
        # Get training slide IDs for filter computation
        train_ids = all_splits['train'].dropna().tolist()
        if valid_ids:
            train_ids, dropped_train = filter_ids_by_stability(train_ids, valid_ids)
            val_ids_raw = all_splits['val'].dropna().tolist()
            test_ids_raw = all_splits['test'].dropna().tolist()
            val_ids, dropped_val = filter_ids_by_stability(val_ids_raw, valid_ids)
            test_ids, dropped_test = filter_ids_by_stability(test_ids_raw, valid_ids)
            print(
                f"Fold {i}: dropped missing stability files "
                f"(train={dropped_train}, val={dropped_val}, test={dropped_test})"
            )
        else:
            val_ids = all_splits['val'].dropna().tolist()
            test_ids = all_splits['test'].dropna().tolist()
        
        # Initialize filter using training data
        if args.stability_mode == 'filter':
            stability_filter = StabilityFilter(
                stability_dir=model_configs[0]['stability_dir'],
                slide_ids=train_ids,
                metric=args.filter_metric,
                category=args.filter_category,
                direction=args.filter_direction,
                percentile=args.filter_percentile,
            )
            print(f"\nFold {i}: Using {stability_filter.n_features_kept}/{stability_filter.n_features_total} features")
            
            # Update embed_dim
            if not args.filter_weighting:
                args.embed_dim = stability_filter.n_features_kept
            else:
                args.embed_dim = stability_filter.n_features_total
        else:
            stability_filter = None
            # For features mode, embed_dim is the feature dimension
            # Will be set after loading a sample
        
        # Create splits with stability awareness
        train_data = dataset.slide_data[dataset.slide_data['slide_id'].isin(train_ids)].reset_index(drop=True)
        val_data = dataset.slide_data[dataset.slide_data['slide_id'].isin(val_ids)].reset_index(drop=True)
        test_data = dataset.slide_data[dataset.slide_data['slide_id'].isin(test_ids)].reset_index(drop=True)
        
        train_split = create_split(train_data, args, model_configs, stability_filter)
        val_split = create_split(val_data, args, model_configs, stability_filter)
        test_split = create_split(test_data, args, model_configs, stability_filter)
        
        datasets = (train_split, val_split, test_split)
        
        # Train
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        
        # Save results
        filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)
    
    # Save summary
    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc
    })
    
    if len(folds) != args.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Mean Test AUC: {np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}")
    print(f"Mean Val AUC:  {np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}")
    print(f"Mean Test Acc: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}")
    print("="*60)


# =============================================================================
# Argument Parser
# =============================================================================

parser = argparse.ArgumentParser(description='Stability-aware CLAM training')

# ─────────────────────────────────────────────────────────────────────────────
# Stability-specific arguments
# ─────────────────────────────────────────────────────────────────────────────
stability_group = parser.add_argument_group('Stability Settings')
stability_group.add_argument('--stability_mode', type=str, required=True,
    choices=['features', 'filter'],
    help='Use stability metrics AS features or to FILTER FM features')
stability_group.add_argument('--stability_dir', type=str, default=None,
    help='Directory containing stability H5 files')

# Mode 1: Stability as features
features_group = parser.add_argument_group('Stability as Features (mode=features)')
features_group.add_argument('--stability_metric', type=str, default=None,
    choices=['icc', 'variance', 'norm_range'],
    help='Which stability metric to use as features')
features_group.add_argument('--stability_category', type=str, default=None,
    choices=['overall', 'spatial', 'color', 'noise'],
    help='Which augmentation category to use')

# Mode 2: Feature filtering
filter_group = parser.add_argument_group('Feature Filtering (mode=filter)')
filter_group.add_argument('--filter_metric', type=str, default='icc',
    choices=['icc', 'variance', 'norm_range'],
    help='Metric to base filtering on')
filter_group.add_argument('--filter_category', type=str, default='overall',
    choices=['overall', 'spatial', 'color', 'noise'],
    help='Category to base filtering on')
filter_group.add_argument('--filter_direction', type=str, default='high',
    choices=['high', 'low'],
    help='Keep high-stability or low-stability features')
filter_group.add_argument('--filter_percentile', type=float, default=80.0,
    help='Percentile threshold (e.g., 80 = keep top 20%%)')
filter_group.add_argument('--filter_weighting', action='store_true', default=False,
    help='Use soft weighting instead of hard masking')

# Multi-model support
multimodel_group = parser.add_argument_group('Multi-Model Support')
multimodel_group.add_argument('--models', type=str, default=None,
    help='Comma-separated list of model names (e.g., uni_v1,conch_v15)')
multimodel_group.add_argument('--data_dirs', type=str, default=None,
    help='Comma-separated list of feature directories')
multimodel_group.add_argument('--stability_dirs', type=str, default=None,
    help='Comma-separated list of stability directories')
multimodel_group.add_argument('--patch_encoder', type=str, default=None,
    help='Single patch encoder name (for single-model mode)')

# ─────────────────────────────────────────────────────────────────────────────
# Standard CLAM arguments (same as main.py)
# ─────────────────────────────────────────────────────────────────────────────
parser.add_argument('--data_root_dir', type=str, default=None,
    help='Data directory')
parser.add_argument('--embed_dim', type=int, default=1024,
    help='Feature embedding dimension')
parser.add_argument('--max_epochs', type=int, default=200,
    help='Maximum number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument('--label_frac', type=float, default=1.0,
    help='Fraction of training labels')
parser.add_argument('--reg', type=float, default=1e-5,
    help='Weight decay')
parser.add_argument('--seed', type=int, default=1,
    help='Random seed')
parser.add_argument('--k', type=int, default=10,
    help='Number of folds')
parser.add_argument('--k_start', type=int, default=-1,
    help='Start fold')
parser.add_argument('--k_end', type=int, default=-1,
    help='End fold')
parser.add_argument('--results_dir', default='./results',
    help='Results directory')
parser.add_argument('--split_dir', type=str, default=None,
    help='Directory containing split files')
parser.add_argument('--log_data', action='store_true', default=False,
    help='Log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False,
    help='Debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False,
    help='Enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam',
    help='Optimizer')
parser.add_argument('--drop_out', type=float, default=0.25,
    help='Dropout rate')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
    help='Slide-level classification loss')
parser.add_argument('--model_type', type=str, 
    choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
    help='Model type')
parser.add_argument('--exp_code', type=str, required=True,
    help='Experiment code')
parser.add_argument('--weighted_sample', action='store_true', default=False,
    help='Enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
    help='Model size')
parser.add_argument('--task', type=str, required=True,
    choices=[
        'task_1_tumor_vs_normal',
        'task_2_tumor_subtyping',
        'task_kidney_grade',
        'task_prostate_grade',
        'task_rectal_stage'
    ],
    help='Which task to run')

# CLAM-specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
    help='Disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
    help='Instance-level clustering loss')
parser.add_argument('--subtyping', action='store_true', default=False,
    help='Subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
    help='Weight coefficient for bag-level loss')
parser.add_argument('--B', type=int, default=8,
    help='Number of positive/negative patches to sample for CLAM')

args = parser.parse_args()

# =============================================================================
# Setup
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

# =============================================================================
# Dataset Setup (Task-specific)
# =============================================================================

print('\n' + '='*60)
print('STABILITY-AWARE CLAM TRAINING')
print('='*60)
print(f'Mode: {args.stability_mode}')
if args.stability_mode == 'features':
    print(f'Metric: {args.stability_metric}')
    print(f'Category: {args.stability_category}')
else:
    print(f'Filter Metric: {args.filter_metric}')
    print(f'Filter Category: {args.filter_category}')
    print(f'Filter Direction: {args.filter_direction}')
    print(f'Filter Percentile: {args.filter_percentile}')
    print(f'Filter Weighting: {args.filter_weighting}')
print('='*60 + '\n')

print('Load Dataset')

# Parse model configs
model_configs = parse_model_configs(args)

# Task-specific dataset creation
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    label_dict = {'normal_tissue': 0, 'tumor_tissue': 1}
    csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv'

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    label_dict = {'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2}
    csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv'

elif args.task == 'task_kidney_grade':
    args.n_classes = 2
    label_dict = {0: 0, 1: 0, 2: 1, 3: 1}
    csv_path = '/mnt/vstor/Data7/bxf169/KidneyCancerPathology/kirc_splits/pre_split/grade_labels.csv'

elif args.task == 'task_prostate_grade':
    args.n_classes = 2
    label_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1}
    csv_path = '/mnt/vstor/Data7/bxf169/ProstateCancerPathology/gleason_grade_clam.csv'

elif args.task == 'task_rectal_stage':
    args.n_classes = 2
    label_dict = {1: 0, 2: 0, 3: 1, 4: 1}
    csv_path = '/mnt/vstor/Data7/bxf169/RectalCancerPathology/rectal_stage_clam.csv'

else:
    raise NotImplementedError(f"Task {args.task} not implemented")

# Create dataset
dataset = create_stability_dataset(args, csv_path, label_dict, model_configs)

# Filter slide_data to only those with available stability files
VALID_STABILITY_IDS = get_valid_stability_ids(model_configs)
if VALID_STABILITY_IDS:
    before_count = len(dataset.slide_data)
    dataset.slide_data = dataset.slide_data[
        dataset.slide_data['slide_id'].isin(VALID_STABILITY_IDS)
    ].reset_index(drop=True)
    after_count = len(dataset.slide_data)
    dropped = before_count - after_count
    print(f"Filtered slide_data by stability files: kept={after_count}, dropped={dropped}")
else:
    print("Warning: no stability files found for filtering; proceeding without filtering.")

# Setup results directory
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, f'{args.exp_code}_s{args.seed}')
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Setup split directory
if args.split_dir is None:
    args.split_dir = os.path.join('splits', f'{args.task}_{int(args.label_frac*100)}')
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print(f'split_dir: {args.split_dir}')
assert os.path.isdir(args.split_dir), f"Split directory not found: {args.split_dir}"

# Save settings
settings = {
    'stability_mode': args.stability_mode,
    'stability_metric': args.stability_metric,
    'stability_category': args.stability_category,
    'filter_metric': args.filter_metric,
    'filter_category': args.filter_category,
    'filter_direction': args.filter_direction,
    'filter_percentile': args.filter_percentile,
    'filter_weighting': args.filter_weighting,
    'num_splits': args.k,
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs,
    'results_dir': args.results_dir,
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    'use_drop_out': args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt,
    'split_dir': args.split_dir,
}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B
    })

with open(os.path.join(args.results_dir, f'experiment_{args.exp_code}.txt'), 'w') as f:
    print(settings, file=f)

print("\n################# Settings ###################")
for key, val in settings.items():
    print(f"{key}:  {val}")

# =============================================================================
# Run Training
# =============================================================================

if __name__ == "__main__":
    results = main(args)
    print("\nFinished!")
    print("End script")
