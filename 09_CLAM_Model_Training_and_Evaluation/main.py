from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # Survival mode uses different metrics
    survival_mode = hasattr(args, 'survival_mode') and args.survival_mode

    if survival_mode:
        all_test_cindex = []
        all_val_cindex = []
        all_test_hr = []
        all_test_pvalue = []
    else:
        all_test_auc = []
        all_val_auc = []
        all_test_acc = []
        all_val_acc = []

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        for ds in (train_dataset, val_dataset, test_dataset):
            ds.load_from_h5(True)     # sets use_h5 = True, file_ext = '.h5',
                              # dir_name = 'h5_files'

        datasets = (train_dataset, val_dataset, test_dataset)
        results, metric1, metric2, metric3, metric4 = train(datasets, i, args)

        if survival_mode:
            # metrics: test_cindex, val_cindex, hazard_ratio, p_value
            all_test_cindex.append(metric1)
            all_val_cindex.append(metric2)
            all_test_hr.append(metric3)
            all_test_pvalue.append(metric4)
        else:
            # metrics: test_auc, val_auc, test_acc, val_acc
            all_test_auc.append(metric1)
            all_val_auc.append(metric2)
            all_test_acc.append(metric3)
            all_val_acc.append(metric4)

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # Create summary DataFrame based on mode
    if survival_mode:
        final_df = pd.DataFrame({
            'folds': folds,
            'test_cindex': all_test_cindex,
            'val_cindex': all_val_cindex,
            'test_hazard_ratio': all_test_hr,
            'test_pvalue': all_test_pvalue
        })
    else:
        final_df = pd.DataFrame({
            'folds': folds,
            'test_auc': all_test_auc,
            'val_auc': all_val_auc,
            'test_acc': all_test_acc,
            'val_acc': all_val_acc
        })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument(
    '--task',
    type=str,
    choices=[
        'task_1_tumor_vs_normal',
        'task_2_tumor_subtyping',
        'task_kidney_grade' ,
        'task_prostate_grade',
        'task_rectal_stage'
    ],
    help='Which predefined dataset/task to run'
)
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

### Survival mode arguments
parser.add_argument('--survival_mode', action='store_true', default=False,
                    help='Enable survival prediction mode (Cox regression)')
parser.add_argument('--time_col', type=str, default='time',
                    help='Column name for time-to-event in CSV (for survival mode)')
parser.add_argument('--event_col', type=str, default='event',
                    help='Column name for event indicator in CSV (for survival mode)')
parser.add_argument('--cox_l1_reg', type=float, default=0.0,
                    help='L1 regularization coefficient for Cox loss')
parser.add_argument('--cox_l2_reg', type=float, default=0.0,
                    help='L2 regularization coefficient for Cox loss')
parser.add_argument('--survival_csv', type=str, default=None,
                    help='Path to survival CSV file (required for survival mode)')
parser.add_argument('--survival_batch_size', type=int, default=None,
                    help='Batch size for survival mode training (default: None, auto-calculate as 25%% of dataset)')
parser.add_argument('--survival_batch_fraction', type=float, default=0.25,
                    help='Fraction of dataset to use as batch size for survival mode (default: 0.25)')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
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
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

if args.survival_mode:
    settings.update({'survival_mode': True,
                     'time_col': args.time_col,
                     'event_col': args.event_col,
                     'cox_l1_reg': args.cox_l1_reg,
                     'cox_l2_reg': args.cox_l2_reg,
                     'survival_batch_size': args.survival_batch_size,
                     'survival_batch_fraction': args.survival_batch_fraction})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
# ──────────────────────────────────────────────────────────────────
elif args.task == 'task_kidney_grade':        # ★ NEW BLOCK ★
    """
    Two-class grade prediction -- G1→0, G2→0, G3→1, G4→1
    Expects:
      • dataset_csv/grade_labels.csv     (case_id,slide_id,label,…)
      • args.data_root_dir/<feature_dir> (your slide-level .h5 patches)
    """
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path  = '//mnt/vstor/Data7/bxf169/KidneyCancerPathology/kirc_splits/pre_split/grade_labels.csv',
        data_dir  = args.data_root_dir,
        shuffle   = False, seed = args.seed, print_info = True,
        label_dict = {0:0, 1:0, 2:1, 3:1},
        # label_dict = {0:0, 1:0, 2:1, 3:1},
        patient_strat = True,
        ignore = []    # keep all slides
    )
    dataset.load_from_h5(True)
    # no subtyping-specific assertion needed
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
elif args.task == 'task_prostate_grade':        # ★ NEW BLOCK ★
    """
    Two-class grade prediction -- 1-6→0, 7-10→1
    """
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path  = '//mnt/vstor/Data7/bxf169/ProstateCancerPathology/gleason_grade_clam.csv',
        data_dir  = args.data_root_dir,
        shuffle   = False, seed = args.seed, print_info = True,
        label_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:1, 9:1, 10:1},
        patient_strat = True,
        ignore = []    # keep all slides
    )
    dataset.load_from_h5(True)
    # no subtyping-specific assertion needed
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
elif args.task == 'task_rectal_stage':        # ★ NEW BLOCK ★
    """
    Two-class grade prediction -- 1→0, 2→0, 3→1, 4→1
    """
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path  = '//mnt/vstor/Data7/bxf169/RectalCancerPathology/rectal_stage_clam.csv',
        data_dir  = args.data_root_dir,
        shuffle   = False, seed = args.seed, print_info = True,
        label_dict = {1:0, 2:0, 3:1, 4:1},
        # label_dict = {0:0, 1:0, 2:1, 3:1},
        patient_strat = True,
        ignore = []    # keep all slides
    )
    dataset.load_from_h5(True)
    # no subtyping-specific assertion needed
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# Survival Mode Task
# ──────────────────────────────────────────────────────────────────
elif args.survival_mode:
    """
    Survival prediction mode using Cox proportional hazards.
    Requires:
      - --survival_csv: Path to CSV with columns: case_id, slide_id, time_col, event_col
      - --time_col: Column name for time-to-event (default: 'time')
      - --event_col: Column name for event indicator (default: 'event')
      - --data_root_dir: Directory containing feature files
    """
    if args.survival_csv is None:
        raise ValueError("--survival_csv is required for survival mode")

    #region agent log
    import json
    try:
        with open('/mnt/vstor/Data7/bxf169/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"main.py:325","message":"Setting n_classes for survival mode","data":{"n_classes_before":getattr(args,'n_classes',None)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    #endregion
    args.n_classes = 1  # Survival outputs single risk score
    #region agent log
    try:
        with open('/mnt/vstor/Data7/bxf169/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"main.py:326","message":"After setting n_classes","data":{"n_classes_after":args.n_classes},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    #endregion

    dataset = Generic_MIL_Dataset(
        csv_path=args.survival_csv,
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={},  # Not used in survival mode
        patient_strat=True,
        ignore=[],
        survival_mode=True,
        time_col=args.time_col,
        event_col=args.event_col
    )
    dataset.load_from_h5(True)

    # Validate survival mode configuration
    if args.model_type not in ['clam_sb', 'clam_mb']:
        raise ValueError("Survival mode only supports clam_sb and clam_mb models")

# ──────────────────────────────────────────────────────────────────

else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    if args.task is not None:
        # Classification mode: use task-based split directory
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        # Survival mode: use exp_code-based split directory (or require explicit --split_dir)
        # For survival mode, splits should be provided via --split_dir or use default
        if hasattr(args, 'survival_mode') and args.survival_mode:
            # Use exp_code to create default split directory name
            if args.exp_code:
                args.split_dir = os.path.join('splits', args.exp_code)
            else:
                raise ValueError("For survival mode, either --split_dir must be provided or --exp_code must be set")
        else:
            raise ValueError("--task or --split_dir must be provided")
else:
    # If split_dir is provided, check if it's an absolute path
    if os.path.isabs(args.split_dir):
        # Use absolute path as-is
        args.split_dir = args.split_dir
    else:
        # Relative path: prepend 'splits' directory
        args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


