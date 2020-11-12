from ntc.ns import *


__all__ = ['main']


xent = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.1)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.fc2(x)
        return x


def forward(model, xs, ys, device):
    xs, ys = xs.to(device), ys.to(device)
    logits = model(xs)
    loss = xent(logits, ys)

    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(ys.view_as(pred)).sum().item()
    return loss, correct


def train(model, device, train_loader, opt, epoch):
    met = AverageMeters()
    model.train()
    acc_count = 0
    ACC_NUM = 16
    opt.zero_grad()

    for i, (xs, ys) in enumerate(train_loader):

        loss, correct = forward(model, xs, ys, device)
        loss.backward()
        acc_count += 1

        if acc_count % ACC_NUM == 0:
            acc_count = 0
            opt.step()
            opt.zero_grad()

        met.update('loss', loss.item())
        met.update('acc', correct, ys.size(0))
    else:
        if acc_count != 0:
            opt.step()

        sayi(f'Epoch {epoch:03d}] Train Loss: {met.avg("loss"):.3f}, Acc: {met.avg("acc"):.3f}')


def test(model, device, test_loader):
    model.eval()
    met = AverageMeters()
    with torch.no_grad():
        for xs, ys in test_loader:
            loss, correct = forward(model, xs, ys, device)

            met.update('loss', loss.item())
            met.update('acc', correct, ys.size(0))

    sayi(f'Test Loss: {met.avg("loss"):.3f}, Acc: {met.avg("acc"):.3f}')


def get_dataloader():
    import torchvision
    from torchvision.transforms import ToTensor
    d_train = torchvision.datasets.MNIST('./mnist/', train=True, transform=ToTensor(), download=True)
    d_test = torchvision.datasets.MNIST('./mnist/', train=False, transform=ToTensor(), download=True)

    l_train = DataLoader(d_train, batch_size=4, shuffle=True, pin_memory=True)
    l_test = DataLoader(d_test, batch_size=256, shuffle=False, pin_memory=True)

    return l_train, l_test


def main():
    device = torch.device("cuda")
    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    train_loader, test_loader = get_dataloader()

    sayi('Start training')
    for epoch in range(1, 100 + 1):
        train(model, device, train_loader, opt, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    set_cuda(0)
    main()
