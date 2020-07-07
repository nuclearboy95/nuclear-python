from ntc.ns import *

__all__ = ['main']


xent = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv4(x)
        x = F.leaky_relu(x, 0.1)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.fc2(x)
        return x


def forward(model, xs, ys, device):
    xs, ys = xs.to(device), ys.to(device)
    logits = model(xs)
    loss = xent(logits, ys.view(-1))

    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(ys.view_as(pred)).sum().item()
    return loss, correct


def train(model, device, train_loader, opt, epoch):
    met = AverageMeters()
    model.train()
    for i, (xs, ys) in enumerate(train_loader):
        opt.zero_grad()

        loss, correct = forward(model, xs, ys, device)
        loss.backward()

        opt.step()

        met.update('loss', loss.item())
        met.update('acc', correct, ys.size(0))

    return met.avgs


def test(model, device, test_loader):
    model.eval()
    met = AverageMeters()
    with torch.no_grad():
        for xs, ys in test_loader:
            loss, correct = forward(model, xs, ys, device)

            met.update('loss', loss.item())
            met.update('acc', correct, ys.size(0))
    return met.avgs


def get_dataloader():
    from tensorflow import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    x_train = np.transpose(x_train, [0, 3, 1, 2])
    x_test = np.transpose(x_test, [0, 3, 1, 2])

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test, dtype=torch.long)

    d_train = TensorDataset(x_train, y_train)
    d_test = TensorDataset(x_test, y_test)

    l_train = DataLoader(d_train, batch_size=64, shuffle=True, pin_memory=True)
    l_test = DataLoader(d_test, batch_size=256, shuffle=False, pin_memory=True)

    return l_train, l_test


def main():
    import ntf
    device = torch.device("cuda")
    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    ntf.tb.set_name('CIFAR10')
    train_loader, test_loader = get_dataloader()

    sayi('Start training')
    for epoch in range(1, 100 + 1):
        d_train = train(model, device, train_loader, opt, epoch)
        d_test = test(model, device, test_loader)

        ntf.tb.add_scalars(d_train, prefix='train', step=epoch)
        ntf.tb.add_scalars(d_test, prefix='test', step=epoch)

        sayi(f'Epoch {epoch:03d}] Train Loss: {d_train["loss"]:.3f}, Acc: {d_train["acc"]:.3f}, Grad Norm: {d_train["grad_norm"]:.3f} | Test Loss: {d_test["loss"]:.3f}, Acc: {d_test["acc"]:.3f}')
