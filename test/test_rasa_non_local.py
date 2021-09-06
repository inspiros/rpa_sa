import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from rpa import MultiheadNonlocal2d


def test_train():
    model = MultiheadNonlocal2d(in_channels=8,
                                rasa_mode='relative_position',
                                rasa_kernel_size=5,
                                rasa_interpolation='slerp')

    x = torch.rand(1, 8, 12, 12)
    out = model(x)
    print(out)


class RASAModel(nn.Module):
    def __init__(self, num_classes, rasa_channels=64):
        super(RASAModel, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, rasa_channels, kernel_size=(3, 3)),
            nn.BatchNorm2d(rasa_channels),
            nn.ReLU()
        )
        self.nl = MultiheadNonlocal2d(in_channels=rasa_channels,
                                      rasa_mode='relative_position',
                                      rasa_kernel_size=(7, 7),
                                      rasa_interpolation='slerp')
        self.bn = nn.BatchNorm2d(rasa_channels)
        self.pool = nn.AvgPool2d(kernel_size=(3, 3))
        self.fc = nn.Linear(rasa_channels * 4 * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn(self.nl(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


def test_train():
    device = torch.device(f"cuda:{torch.cuda.device_count() - 1}"
                          if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    cifar10_train = datasets.CIFAR10(root="D:/code/projects/pycharm/multi_stream_videonet/data/cifar10",
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root="D:/code/projects/pycharm/multi_stream_videonet/data/cifar10",
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    train_loader = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=8)
    test_loader = DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=8)

    model = RASAModel(num_classes=10, ).to(device)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

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

    print('Training RASA model')
    for epoch_id in range(max_epochs):
        train_err, train_loss = epoch(epoch_id, train_loader, model, optim, scheduler)
        test_err, test_loss = epoch(epoch_id, test_loader, model)
        print(f'[Epoch {epoch_id + 1}/{max_epochs}] '
              f'train_acc={train_err:.03f}, train_loss={train_loss:.03f}, '
              f'test_acc={test_err:.03f}, test_loss={test_loss:.03f}')


if __name__ == '__main__':
    test_train()
