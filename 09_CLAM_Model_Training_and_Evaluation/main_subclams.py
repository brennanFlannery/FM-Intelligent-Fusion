from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset_modules.dataset_subclam import SubCLAM_Dataset
from models.model_subclam import SubCLAM
from utils.file_utils import save_pkl
from utils.utils import calculate_error, get_optim, print_network, make_weights_for_balanced_classes_split

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Accuracy_Logger(object):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, y_hat, y):
        y_hat = int(y_hat)
        y = int(y)
        self.data[y]["count"] += 1
        self.data[y]["correct"] += (y_hat == y)

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        if count == 0:
            return None, correct, count
        acc = float(correct) / count
        return acc, correct, count


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


def collate_subclam(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    cluster_ids = torch.from_numpy(batch[0][2]).long()
    coords = batch[0][3]
    return [img, label, cluster_ids, coords]


def get_split_loader_subclam(split_dataset, training=False, weighted=False):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(split_dataset)
    else:
        sampler = SequentialSampler(split_dataset)
    loader = DataLoader(split_dataset, batch_size=1, sampler=sampler, collate_fn=collate_subclam, **kwargs)
    return loader


def train_loop_subclam(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.0
    train_error = 0.0

    print('\n')
    for batch_idx, (data, label, cluster_ids, coords) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        cluster_ids = cluster_ids.to(device)

        logits, y_prob, y_hat, _, _ = model(data, cluster_ids)
        acc_logger.log(y_hat, label)

        loss = loss_fn(logits, label)
        loss_value = loss.item()
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(
                batch_idx, loss_value, label.item(), data.size(0)))

        error = calculate_error(y_hat, label)
        train_error += error

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate_subclam(cur, epoch, model, loader, n_classes, writer=None, loss_fn=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.0
    val_error = 0.0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.inference_mode():
        for batch_idx, (data, label, cluster_ids, coords) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            cluster_ids = cluster_ids.to(device)

            logits, y_prob, y_hat, _, _ = model(data, cluster_ids)
            acc_logger.log(y_hat, label)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            prob[batch_idx] = y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        aucs = []
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve((labels == class_idx).astype(int), prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    return val_loss, auc


def summary_subclam(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.0

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    with torch.inference_mode():
        for batch_idx, (data, label, cluster_ids, coords) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            cluster_ids = cluster_ids.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            logits, y_prob, y_hat, _, _ = model(data, cluster_ids)
            acc_logger.log(y_hat, label)

            probs = y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(y_hat, label)
            test_error += error

    test_error /= len(loader)
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve((all_labels == class_idx).astype(int), all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def train(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {
        "dropout": args.drop_out,
        'n_classes': args.n_classes,
        "embed_dim": args.embed_dim,
        "n_clusters": args.n_clusters,
        "size_arg": args.model_size,
    }
    model = SubCLAM(**model_dict)
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader_subclam(train_split, training=True, weighted=args.weighted_sample)
    val_loader = get_split_loader_subclam(val_split, training=False)
    test_loader = get_split_loader_subclam(test_split, training=False)
    print('Done!')

    last_val_auc = None
    for epoch in range(args.max_epochs):
        train_loop_subclam(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        _, last_val_auc = validate_subclam(cur, epoch, model, val_loader, args.n_classes, writer, loss_fn)

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    results_dict, test_error, test_auc, acc_logger = summary_subclam(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return results_dict, test_auc, last_val_auc, 1 - test_error, None


def main(args):
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

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        )

        for ds in (train_dataset, val_dataset, test_dataset):
            ds.load_from_h5(True)

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

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


parser = argparse.ArgumentParser(description='Configurations for Sub-CLAM Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--k', type=int, default=10, help='number of folds')
parser.add_argument('--k_start', type=int, default=-1, help='start fold')
parser.add_argument('--k_end', type=int, default=-1, help='end fold')
parser.add_argument('--results_dir', default='./results', help='results directory')
parser.add_argument('--split_dir', type=str, default=None, help='split directory')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model')
parser.add_argument('--n_clusters', type=int, default=5, help='number of tissue clusters')
parser.add_argument('--cluster_cache_dir', type=str, required=True, help='cluster cache directory')
parser.add_argument('--exp_code', type=str, required=True, help='experiment code for saving results')
parser.add_argument(
    '--task',
    type=str,
    choices=[
        'task_1_tumor_vs_normal',
        'task_2_tumor_subtyping',
        'task_kidney_grade',
        'task_prostate_grade',
        'task_rectal_stage'
    ],
    help='Which predefined dataset/task to run'
)

args = parser.parse_args()
seed_torch(args.seed)

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = SubCLAM_Dataset(
        csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
        patient_strat=False,
        ignore=[],
        cluster_cache_dir=args.cluster_cache_dir,
    )

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = SubCLAM_Dataset(
        csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
        patient_strat=False,
        ignore=[],
        cluster_cache_dir=args.cluster_cache_dir,
    )

elif args.task == 'task_kidney_grade':
    args.n_classes = 2
    dataset = SubCLAM_Dataset(
        csv_path='//mnt/vstor/Data7/bxf169/KidneyCancerPathology/kirc_splits/pre_split/grade_labels.csv',
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={0: 0, 1: 0, 2: 1, 3: 1},
        patient_strat=True,
        ignore=[],
        cluster_cache_dir=args.cluster_cache_dir,
    )

elif args.task == 'task_prostate_grade':
    args.n_classes = 2
    dataset = SubCLAM_Dataset(
        csv_path='//mnt/vstor/Data7/bxf169/ProstateCancerPathology/gleason_grade_clam.csv',
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1},
        patient_strat=True,
        ignore=[],
        cluster_cache_dir=args.cluster_cache_dir,
    )

elif args.task == 'task_rectal_stage':
    args.n_classes = 2
    dataset = SubCLAM_Dataset(
        csv_path='//mnt/vstor/Data7/bxf169/RectalCancerPathology/rectal_stage_clam.csv',
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={1: 0, 2: 0, 3: 1, 4: 1},
        patient_strat=True,
        ignore=[],
        cluster_cache_dir=args.cluster_cache_dir,
    )

else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    if args.task is not None:
        args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
    else:
        raise ValueError("--task or --split_dir must be provided")
else:
    if not os.path.isabs(args.split_dir):
        args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    settings = {
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
        'model_size': args.model_size,
        'n_clusters': args.n_clusters,
        'cluster_cache_dir': args.cluster_cache_dir,
        'weighted_sample': args.weighted_sample,
        'opt': args.opt,
    }
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    main(args)
    print("finished!")
