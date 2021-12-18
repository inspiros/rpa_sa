import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from rpa_sa import MultiheadNonlocal2d


def test_forward():
    model = MultiheadNonlocal2d(in_channels=8,
                                rpa_mode='relative_position',
                                rpa_kernel_size=5,
                                rpa_interpolation='slerp')

    x = torch.rand(1, 8, 12, 12)
    out = model(x)
    print(out)


class RPASAModel(nn.Module):
    def __init__(self, num_classes):
        super(RPASAModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.nl1 = nn.Sequential(
            MultiheadNonlocal2d(in_channels=32,
                                rpa_mode='relative_position',
                                rpa_kernel_size=7,
                                rpa_interpolation='lerp'),
            nn.BatchNorm2d(32)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.nl2 = nn.Sequential(
            MultiheadNonlocal2d(in_channels=64,
                                rpa_mode='relative_position',
                                rpa_kernel_size=7,
                                rpa_interpolation='lerp'),
            nn.BatchNorm2d(64)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.nl1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.nl2(x)

        x = self.global_pool(x).flatten(1)

        x = self.fc(x)
        return x


def test_train():
    device = "cpu"  # torch.device(f"cuda:{torch.cuda.device_count() - 1}"
    #      if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    cifar10_train = datasets.CIFAR10(root="D:/code/projects/pycharm/multi_stream_videonet/data/cifar10",
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root="D:/code/projects/pycharm/multi_stream_videonet/data/cifar10",
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    train_loader = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=2)

    model = RPASAModel(num_classes=10).to(device)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))
    exit()

    max_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                           max_epochs * len(train_loader),
                                                           eta_min=1e-6)

    # standard training or evaluation loop
    def epoch(epoch_id, loader, model, optim=None, lr_scheduler=None):
        epoch_loss, epoch_acc = 0., 0.
        model.eval() if optim is None else model.train()
        task = 'Training' if optim is not None else 'Testing'
        pbar = tqdm(enumerate(loader), desc=f'[Epoch {epoch_id}] ({task})')
        for batch_id, (X, y) in pbar:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = criterion(yp, y)
            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            batch_acc = yp.argmax(1).eq(y).sum().item() / X.size(0)
            batch_loss = loss.item()
            epoch_acc += batch_acc * X.size(0)
            epoch_loss += batch_loss * X.size(0)

            pbar.set_description(f'[Epoch {epoch_id} - Iter {batch_id}/{len(loader)}] ({task}) '
                                 f'acc={batch_acc:.03f}, loss={batch_loss:.03f}')
        return epoch_acc / len(loader.dataset), epoch_loss / len(loader.dataset)

    print('Training RPA_SA model')
    for epoch_id in range(max_epochs):
        train_err, train_loss = epoch(epoch_id, train_loader, model, optim, scheduler)
        test_err, test_loss = epoch(epoch_id, test_loader, model)
        print(f'[Epoch {epoch_id + 1}/{max_epochs}] '
              f'train_acc={train_err:.03f}, train_loss={train_loss:.03f}, '
              f'test_acc={test_err:.03f}, test_loss={test_loss:.03f}')

    torch.save(model.state_dict(), 'weights/relative_position_slerp.pt')


def test_speed():
    device = "cuda"
    torch.manual_seed(0)

    batch_sz = 1
    in_channels = 256
    height = 84
    width = 48

    x = torch.rand(batch_sz, in_channels, height, width, device=device)
    nl_module = MultiheadNonlocal2d(in_channels=in_channels,
                                    rpa_mode=None).to(device)
    rda_nl_module = MultiheadNonlocal2d(in_channels=in_channels,
                                        rpa_mode='relative_distance',
                                        rpa_kernel_size=6,
                                        rpa_interpolation='nearest',
                                        rpa_zero_init=False).to(device)

    # warmup
    for _ in range(10):
        nl_module(x)

    # nl_module
    for _ in trange(200, desc='Nonlocal Module'):
        nl_module(x)

    # rda_nl_module
    for _ in trange(200, desc='Relative Distance-Aware Nonlocal Module'):
        rda_nl_module(x)


if __name__ == '__main__':
    # test_train()
    test_speed()
