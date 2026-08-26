import os
import json
import time
import random
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(2, min(4, os.cpu_count() or 2)))
DEVICE = torch.device('cpu')
ROOT = Path('exp5_data')
OUT = Path('.github/exp5-results')
OUT.mkdir(parents=True, exist_ok=True)

weights = MobileNet_V2_Weights.DEFAULT
val_transform = weights.transforms()
train_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainval_ds = OxfordIIITPet(ROOT, split='trainval', target_types='category', transform=val_transform, download=True)
test_ds = OxfordIIITPet(ROOT, split='test', target_types='category', transform=val_transform, download=True)
classes = trainval_ds.classes
labels = np.array(trainval_ds._labels, dtype=np.int64)
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.20, random_state=SEED, stratify=labels)

base = mobilenet_v2(weights=weights)
base.classifier = nn.Identity()
base.eval().to(DEVICE)
for p in base.parameters():
    p.requires_grad = False

@torch.inference_mode()
def extract_features(dataset, indices=None, batch_size=64):
    ds = Subset(dataset, indices.tolist()) if indices is not None else dataset
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    feats, ys = [], []
    for x, y in loader:
        feats.append(base(x.to(DEVICE)).cpu())
        ys.append(torch.as_tensor(y).long().cpu())
    return torch.cat(feats), torch.cat(ys)

t0 = time.perf_counter()
all_features, all_y = extract_features(trainval_ds, None, 64)
feature_extraction_train_time = time.perf_counter() - t0
X_train = all_features[torch.as_tensor(train_idx)]
y_train = all_y[torch.as_tensor(train_idx)]
X_val = all_features[torch.as_tensor(val_idx)]
y_val = all_y[torch.as_tensor(val_idx)]

class Head(nn.Module):
    def __init__(self, dropout=0.0, use_bn=False):
        super().__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.bn = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 37)
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.bn(self.fc1(x)))))

def initialize_head(model, method):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == 'Zero':
                nn.init.zeros_(m.weight)
            elif method == 'Random':
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif method == 'Xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'He':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def make_optimizer(name, params, lr, weight_decay=0.0):
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == 'Momentum':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == 'RMSProp':
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

def evaluate_head(model, X, y, batch_size=256):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss) * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            n += len(yb)
    return total_loss / n, correct / n

def train_head(Xtr, ytr, Xv, yv, epochs=10, batch_size=32, lr=0.001, optimizer='Adam', init='He', dropout=0.0, use_bn=False, weight_decay=0.0, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Head(dropout=dropout, use_bn=use_bn)
    initialize_head(model, init)
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    hist = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(yb)
            correct += int((logits.detach().argmax(1) == yb).sum())
            n += len(yb)
        vl, va = evaluate_head(model, Xv, yv)
        hist['train_loss'].append(total_loss / n)
        hist['train_accuracy'].append(correct / n)
        hist['val_loss'].append(vl)
        hist['val_accuracy'].append(va)
    elapsed = time.perf_counter() - start
    return model, hist, elapsed

def hist_summary(hist, elapsed):
    best = float(max(hist['val_accuracy']))
    best_epoch = int(np.argmax(hist['val_accuracy']) + 1)
    target = 0.99 * best
    conv_epoch = next((i + 1 for i, v in enumerate(hist['val_accuracy']) if v >= target), len(hist['val_accuracy']))
    return {'final_loss': float(hist['train_loss'][-1]), 'best_val_accuracy': best, 'best_epoch': best_epoch, 'epoch_to_converge': int(conv_epoch), 'time_s': float(elapsed)}

results = {'seed': SEED, 'dataset': {'name': 'Oxford-IIIT Pet', 'classes': classes, 'trainval_count': int(len(trainval_ds)), 'train_count': int(len(train_idx)), 'validation_count': int(len(val_idx)), 'test_count': int(len(test_ds)), 'image_size': '224 x 224 x 3', 'feature_extraction_train_time_s': float(feature_extraction_train_time)}}

initialization = {}
for method in ['Zero', 'Random', 'Xavier', 'He']:
    model, hist, elapsed = train_head(X_train, y_train, X_val, y_val, epochs=10, batch_size=32, lr=0.001, optimizer='Adam', init=method, dropout=0.0, use_bn=False, seed=SEED + ['Zero','Random','Xavier','He'].index(method))
    initialization[method] = {'history': hist, **hist_summary(hist, elapsed)}
results['initialization'] = initialization
best_init = max(initialization, key=lambda k: initialization[k]['best_val_accuracy'])
results['best_initialization'] = best_init

regularization = {}
reg_specs = {'No Regularization': dict(dropout=0.0, use_bn=False, weight_decay=0.0), 'L2': dict(dropout=0.0, use_bn=False, weight_decay=1e-4), 'Dropout': dict(dropout=0.5, use_bn=False, weight_decay=0.0), 'Batch Normalization': dict(dropout=0.0, use_bn=True, weight_decay=0.0)}
for i, (name, spec) in enumerate(reg_specs.items()):
    model, hist, elapsed = train_head(X_train, y_train, X_val, y_val, epochs=12, batch_size=32, lr=0.001, optimizer='Adam', init=best_init, seed=100 + i, **spec)
    regularization[name] = {'history': hist, **hist_summary(hist, elapsed), **spec}
results['regularization'] = regularization
best_reg = max(regularization, key=lambda k: regularization[k]['best_val_accuracy'])
results['best_regularization'] = best_reg

optimizers = {}
for i, name in enumerate(['SGD', 'Momentum', 'RMSProp', 'Adam']):
    model, hist, elapsed = train_head(X_train, y_train, X_val, y_val, epochs=12, batch_size=32, lr=0.001, optimizer=name, init=best_init, dropout=0.25, use_bn=True, seed=200 + i)
    optimizers[name] = {'history': hist, **hist_summary(hist, elapsed), 'learning_rate': 0.001}
results['optimizers'] = optimizers
best_optimizer = max(optimizers, key=lambda k: optimizers[k]['best_val_accuracy'])
results['best_optimizer'] = best_optimizer

hyper_rows = []
hyper_runs = {}
def run_hparam(label, lr, batch_size, dropout, optimizer, seed):
    key = f'{lr}_{batch_size}_{dropout}_{optimizer}'
    if key not in hyper_runs:
        model, hist, elapsed = train_head(X_train, y_train, X_val, y_val, epochs=10, batch_size=batch_size, lr=lr, optimizer=optimizer, init=best_init, dropout=dropout, use_bn=True, seed=seed)
        hyper_runs[key] = {'model': model, 'history': hist, 'elapsed': elapsed}
    r = hyper_runs[key]
    summary = hist_summary(r['history'], r['elapsed'])
    hyper_rows.append({'setting': label, 'learning_rate': lr, 'batch_size': batch_size, 'dropout': dropout, 'optimizer': optimizer, 'best_val_accuracy': summary['best_val_accuracy'], 'time_s': summary['time_s']})
    return r
for i, lr in enumerate([0.001, 0.0001]): run_hparam(f'Learning Rate {lr:g}', lr, 32, 0.25, 'Adam', 300 + i)
for i, bs in enumerate([16, 32, 64]): run_hparam(f'Batch Size {bs}', 0.001, bs, 0.25, 'Adam', 310 + i)
for i, dr in enumerate([0.0, 0.25, 0.5]): run_hparam(f'Dropout {dr:g}', 0.001, 32, dr, 'Adam', 320 + i)
for i, opt in enumerate(['SGD', 'Adam']): run_hparam(f'Optimizer {opt}', 0.001, 32, 0.25, opt, 330 + i)
unique_hyper = {}
for row in hyper_rows:
    key = (row['learning_rate'], row['batch_size'], row['dropout'], row['optimizer'])
    if key not in unique_hyper or row['best_val_accuracy'] > unique_hyper[key]['best_val_accuracy']: unique_hyper[key] = row
best_hyper_row = max(unique_hyper.values(), key=lambda r: r['best_val_accuracy'])
best_cfg = {'learning_rate': best_hyper_row['learning_rate'], 'batch_size': best_hyper_row['batch_size'], 'dropout': best_hyper_row['dropout'], 'optimizer': best_hyper_row['optimizer'], 'use_bn': True, 'init': best_init}
results['hyperparameter_study'] = hyper_rows
results['best_hyperparameters'] = best_cfg

feature_model, feature_hist, feature_time = train_head(X_train, y_train, X_val, y_val, epochs=12, batch_size=best_cfg['batch_size'], lr=best_cfg['learning_rate'], optimizer=best_cfg['optimizer'], init=best_cfg['init'], dropout=best_cfg['dropout'], use_bn=True, seed=400)
results['feature_extraction'] = {'history': feature_hist, **hist_summary(feature_hist, feature_time)}

class MobileNetPet(nn.Module):
    def __init__(self, head_state=None, dropout=0.25, use_bn=True):
        super().__init__()
        self.base = mobilenet_v2(weights=weights)
        self.base.classifier = nn.Identity()
        self.head = Head(dropout=dropout, use_bn=use_bn)
        if head_state is not None: self.head.load_state_dict(head_state)
    def forward(self, x): return self.head(self.base(x))

def prepare_partial_model(head_state, dropout):
    model = MobileNetPet(head_state=head_state, dropout=dropout, use_bn=True)
    for p in model.base.parameters(): p.requires_grad = False
    for block in list(model.base.features.children())[-2:]:
        for p in block.parameters(): p.requires_grad = True
    for m in model.base.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    return model

def image_loaders(train_indices, val_indices, batch_size):
    tr_ds = OxfordIIITPet(ROOT, split='trainval', target_types='category', transform=train_aug, download=False)
    va_ds = OxfordIIITPet(ROOT, split='trainval', target_types='category', transform=val_transform, download=False)
    tr_loader = DataLoader(Subset(tr_ds, train_indices.tolist()), batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
    va_loader = DataLoader(Subset(va_ds, val_indices.tolist()), batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)
    return tr_loader, va_loader

def train_full_model(model, tr_loader, va_loader, epochs, lr):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for m in model.base.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in tr_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(yb)
            correct += int((logits.detach().argmax(1) == yb).sum())
            n += len(yb)
        hist['train_loss'].append(total_loss / n)
        hist['train_accuracy'].append(correct / n)
        model.eval()
        vloss, vcorrect, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vloss += float(loss) * len(yb)
                vcorrect += int((logits.argmax(1) == yb).sum())
                vn += len(yb)
        hist['val_loss'].append(vloss / vn)
        hist['val_accuracy'].append(vcorrect / vn)
    return model, hist, time.perf_counter() - start

tr_loader, va_loader = image_loaders(train_idx, val_idx, best_cfg['batch_size'])
ft_lr_trials = {}
for i, lr in enumerate([1e-4, 1e-5]):
    torch.manual_seed(500 + i)
    m = prepare_partial_model(copy.deepcopy(feature_model.state_dict()), best_cfg['dropout'])
    m, h, e = train_full_model(m, tr_loader, va_loader, 1, lr)
    ft_lr_trials[str(lr)] = {'history': h, 'best_val_accuracy': float(max(h['val_accuracy'])), 'time_s': float(e)}
best_ft_lr = float(max(ft_lr_trials, key=lambda k: ft_lr_trials[k]['best_val_accuracy']))
results['fine_tuning_lr_study'] = ft_lr_trials
results['best_fine_tuning_lr'] = best_ft_lr

torch.manual_seed(550)
ft_model = prepare_partial_model(copy.deepcopy(feature_model.state_dict()), best_cfg['dropout'])
ft_model, ft_hist, ft_time = train_full_model(ft_model, tr_loader, va_loader, 3, best_ft_lr)
results['fine_tuning'] = {'history': ft_hist, 'best_val_accuracy': float(max(ft_hist['val_accuracy'])), 'time_s': float(ft_time), 'learning_rate': best_ft_lr}

cv_candidates = [
    {'name': 'C1', **best_cfg, 'strategy': 'Frozen base'},
    {'name': 'C2', 'learning_rate': 0.0005, 'batch_size': 32, 'dropout': 0.25, 'optimizer': 'Adam', 'use_bn': True, 'init': best_init, 'strategy': 'Frozen base'},
    {'name': 'C3', 'learning_rate': best_cfg['learning_rate'], 'batch_size': 64, 'dropout': 0.5, 'optimizer': 'Adam', 'use_bn': True, 'init': best_init, 'strategy': 'Frozen base'},
    {'name': 'C4', 'learning_rate': best_cfg['learning_rate'], 'batch_size': 32, 'dropout': 0.0, 'optimizer': best_cfg['optimizer'], 'use_bn': True, 'init': best_init, 'strategy': 'Frozen base'}]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = []
for cix, cfg in enumerate(cv_candidates):
    folds, times = [], []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        _, h, elapsed = train_head(all_features[tri], all_y[tri], all_features[vai], all_y[vai], epochs=10, batch_size=cfg['batch_size'], lr=cfg['learning_rate'], optimizer=cfg['optimizer'], init=cfg['init'], dropout=cfg['dropout'], use_bn=cfg['use_bn'], seed=600 + cix * 10 + fold)
        folds.append(float(max(h['val_accuracy'])))
        times.append(float(elapsed))
    cv_results.append({**cfg, 'folds': folds, 'mean': float(np.mean(folds)), 'sd': float(np.std(folds)), 'time_s': float(np.sum(times))})
results['cross_validation'] = cv_results
selected_cv = max(cv_results, key=lambda r: (r['mean'], -r['sd']))
results['selected_cv_configuration'] = selected_cv['name']

extra_cfgs = [
    {'name': 'E1', 'learning_rate': 0.0005, 'batch_size': 48, 'dropout': 0.35, 'optimizer': 'Adam', 'use_bn': True, 'init': best_init, 'strategy': 'Frozen base'},
    {'name': 'E2', 'learning_rate': 0.0002, 'batch_size': 64, 'dropout': 0.10, 'optimizer': 'Adam', 'use_bn': True, 'init': best_init, 'strategy': 'Frozen base'}]
extra_results = []
for cix, cfg in enumerate(extra_cfgs):
    folds, times = [], []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        _, h, elapsed = train_head(all_features[tri], all_y[tri], all_features[vai], all_y[vai], epochs=10, batch_size=cfg['batch_size'], lr=cfg['learning_rate'], optimizer=cfg['optimizer'], init=cfg['init'], dropout=cfg['dropout'], use_bn=cfg['use_bn'], seed=700 + cix * 10 + fold)
        folds.append(float(max(h['val_accuracy'])))
        times.append(float(elapsed))
    extra_results.append({**cfg, 'folds': folds, 'mean': float(np.mean(folds)), 'sd': float(np.std(folds)), 'time_s': float(np.sum(times))})
results['additional_exercise'] = extra_results

selected_cfg = next(c for c in cv_candidates if c['name'] == selected_cv['name'])
print('Hyperparameter selection complete; extracting untouched test features now.')
t0 = time.perf_counter()
test_features, test_y = extract_features(test_ds, None, 64)
results['dataset']['feature_extraction_test_time_s'] = float(time.perf_counter() - t0)

def full_feature_test(label, cfg, epochs=12, seed=800):
    m, h, elapsed = train_head(all_features, all_y, all_features[:1], all_y[:1], epochs=epochs, batch_size=cfg['batch_size'], lr=cfg['learning_rate'], optimizer=cfg['optimizer'], init=cfg['init'], dropout=cfg['dropout'], use_bn=cfg.get('use_bn', False), weight_decay=cfg.get('weight_decay', 0.0), seed=seed)
    loss, acc = evaluate_head(m, test_features, test_y)
    return m, {'label': label, 'cv_accuracy': None, 'sd': None, 'test_accuracy': float(acc), 'test_loss': float(loss), 'training_time_s': float(elapsed)}

baseline_cfg = {'learning_rate': 0.001, 'batch_size': 32, 'dropout': 0.0, 'optimizer': 'Adam', 'init': 'He', 'use_bn': False}
best_init_cfg = {**baseline_cfg, 'init': best_init}
reg_spec = reg_specs[best_reg]
best_reg_cfg = {'learning_rate': 0.001, 'batch_size': 32, 'dropout': reg_spec['dropout'], 'optimizer': 'Adam', 'init': best_init, 'use_bn': reg_spec['use_bn'], 'weight_decay': reg_spec['weight_decay']}
best_opt_cfg = {'learning_rate': 0.001, 'batch_size': 32, 'dropout': 0.25, 'optimizer': best_optimizer, 'init': best_init, 'use_bn': True}
selected_test_cfg = {'learning_rate': selected_cfg['learning_rate'], 'batch_size': selected_cfg['batch_size'], 'dropout': selected_cfg['dropout'], 'optimizer': selected_cfg['optimizer'], 'init': selected_cfg['init'], 'use_bn': selected_cfg['use_bn']}
overall = []
for i, (label, cfg) in enumerate([('Baseline', baseline_cfg), ('Best Initialization', best_init_cfg), ('Best Regularization', best_reg_cfg), ('Best Optimizer', best_opt_cfg), ('Best Hyperparameters', selected_test_cfg)]):
    m, row = full_feature_test(label, cfg, 12, 800 + i)
    if label == 'Best Hyperparameters':
        final_head = m
        row['cv_accuracy'] = float(selected_cv['mean'])
        row['sd'] = float(selected_cv['sd'])
    overall.append(row)

final_train_ds = OxfordIIITPet(ROOT, split='trainval', target_types='category', transform=train_aug, download=False)
final_train_loader = DataLoader(final_train_ds, batch_size=selected_cfg['batch_size'], shuffle=True, num_workers=2, persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
final_test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)
final_ft = prepare_partial_model(copy.deepcopy(final_head.state_dict()), selected_cfg['dropout'])
opt = torch.optim.Adam([p for p in final_ft.parameters() if p.requires_grad], lr=best_ft_lr)
loss_fn = nn.CrossEntropyLoss()
start = time.perf_counter()
final_ft_train_hist = {'train_loss': [], 'train_accuracy': []}
for _ in range(2):
    final_ft.train()
    for m in final_ft.base.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()
    tl, tc, tn = 0.0, 0, 0
    for xb, yb in final_train_loader:
        opt.zero_grad(set_to_none=True)
        logits = final_ft(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        tl += float(loss.detach()) * len(yb)
        tc += int((logits.detach().argmax(1) == yb).sum())
        tn += len(yb)
    final_ft_train_hist['train_loss'].append(tl / tn)
    final_ft_train_hist['train_accuracy'].append(tc / tn)
final_ft_time = time.perf_counter() - start

final_ft.eval()
all_pred, all_true = [], []
test_loss_sum, test_n = 0.0, 0
with torch.no_grad():
    for xb, yb in final_test_loader:
        logits = final_ft(xb)
        loss = loss_fn(logits, yb)
        test_loss_sum += float(loss) * len(yb)
        test_n += len(yb)
        all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        all_true.extend(yb.cpu().numpy().tolist())
all_pred = np.array(all_pred)
all_true = np.array(all_true)
acc = accuracy_score(all_true, all_pred)
precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='weighted', zero_division=0)
cm = confusion_matrix(all_true, all_pred)
report = classification_report(all_true, all_pred, target_names=classes, output_dict=True, zero_division=0)
class_acc = np.diag(cm) / cm.sum(axis=1)
best_classes = np.argsort(class_acc)[-5:][::-1].tolist()
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
flat = np.argsort(cm_off.ravel())[::-1]
confusions = []
for idx in flat:
    i, j = np.unravel_index(idx, cm_off.shape)
    if cm_off[i, j] <= 0: break
    confusions.append({'true': classes[i], 'predicted': classes[j], 'count': int(cm_off[i, j])})
    if len(confusions) == 5: break
final_params = int(sum(p.numel() for p in final_ft.parameters()))
final_trainable = int(sum(p.numel() for p in final_ft.parameters() if p.requires_grad))
results['final_model'] = {'mean_cv_accuracy': float(selected_cv['mean']), 'cv_standard_deviation': float(selected_cv['sd']), 'test_accuracy': float(acc), 'test_loss': float(test_loss_sum / test_n), 'precision_weighted': float(precision), 'recall_weighted': float(recall), 'f1_weighted': float(f1), 'training_time_s': float(overall[-1]['training_time_s'] + final_ft_time), 'fine_tuning_time_s': float(final_ft_time), 'number_of_parameters': final_params, 'trainable_parameters_during_fine_tuning': final_trainable, 'confusion_matrix': cm.tolist(), 'classification_report': report, 'best_classified_classes': [{'class': classes[i], 'accuracy': float(class_acc[i])} for i in best_classes], 'most_confused_pairs': confusions, 'final_retrain_history': final_ft_train_hist}
overall.append({'label': 'Fine-Tuned Model', 'cv_accuracy': float(selected_cv['mean']), 'sd': float(selected_cv['sd']), 'test_accuracy': float(acc), 'test_loss': float(test_loss_sum / test_n), 'training_time_s': float(overall[-1]['training_time_s'] + final_ft_time)})
results['overall_results'] = overall
results['batch_norm_numerical'] = {'input': [2,4,6,8], 'mean': 5.0, 'variance': 5.0, 'std': float(np.sqrt(5.0)), 'normalized': [float((x-5.0)/np.sqrt(5.0)) for x in [2,4,6,8]], 'gamma': 1.0, 'beta': 0.0}
results['convolution_dimension_examples'] = [{'N':224,'K':3,'P':1,'S':1,'O':224}, {'N':224,'K':3,'P':1,'S':2,'O':112}, {'N':224,'K':5,'P':0,'S':1,'O':220}]
with open(OUT / 'results.json', 'w') as f: json.dump(results, f, indent=2)
print(json.dumps({'best_init': best_init, 'best_reg': best_reg, 'best_optimizer': best_optimizer, 'best_hyper': best_cfg, 'best_ft_lr': best_ft_lr, 'selected_cv': selected_cv['name'], 'final_test_accuracy': acc}, indent=2))
