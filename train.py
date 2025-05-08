import argparse, yaml, os, random, pathlib, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import InstrumentDataset
from model   import SmallCNN
from tqdm    import tqdm


def set_seed(seed=1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy(logits, y):              
    return (logits.argmax(1) == y).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    accs = []
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            accs.append(accuracy(model(x), y))
    return float(np.mean(accs))

def main(cfg):
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    label_names = sorted(
        d.name for d in pathlib.Path(cfg.data_dir).iterdir() if d.is_dir()
    )
    n_classes = len(label_names)
    print(f'Found {n_classes} classes:', ', '.join(label_names))

    tr_ds = InstrumentDataset(cfg.data_dir, 'train', cfg.sample_rate, cfg.n_mels)
    vl_ds = InstrumentDataset(cfg.data_dir, 'val',   cfg.sample_rate, cfg.n_mels)
    ts_ds = InstrumentDataset(cfg.data_dir, 'test',  cfg.sample_rate, cfg.n_mels)

    dl_kwargs = dict(batch_size=cfg.batch_size, pin_memory=True,
                     num_workers=4, drop_last=False)
    tr_dl = DataLoader(tr_ds, shuffle=True,  **dl_kwargs)
    vl_dl = DataLoader(vl_ds, shuffle=False, **dl_kwargs)
    ts_dl = DataLoader(ts_ds, shuffle=False, **dl_kwargs)

    model  = SmallCNN(n_classes=n_classes).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_f = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(tr_dl, desc=f'E{epoch}/{cfg.epochs}')
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device):
                logits = model(x)
                loss   = loss_f(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            losses.append(loss.item())
            if len(losses) % 10 == 0:
                pbar.set_postfix(loss=np.mean(losses[-10:]).item())

        val_acc = evaluate(model, vl_dl, device)
        print(f'  val acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'state_dict'  : model.state_dict(),
                'n_classes'   : n_classes,
                'label_names' : label_names,
                'config'      : vars(cfg)
            }, 'best_model.pt')
            print('  ✓ checkpoint saved')

    print('\nTesting best model …')
    best = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(best['state_dict'])
    test_acc = evaluate(model, ts_dl, device)
    print(f'Best val acc = {best_acc:.4f} | test acc = {test_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='config.yaml')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs',  type=int)
    parser.add_argument('--lr',      type=float)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if args.epochs is not None: cfg_dict['epochs'] = args.epochs
    if args.lr     is not None: cfg_dict['lr']     = args.lr
    cfg_dict['data_dir'] = args.data_dir
    cfg = argparse.Namespace(**cfg_dict)

    main(cfg)
