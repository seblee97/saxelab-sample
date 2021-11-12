import torch
import torch.nn as nn


class Runner:
    def __init__(self, net, train_dataloader, test_dataloader, lr, device):

        self._net = net
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._lr = lr
        self._device = device

        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    def train(self, num_epochs):
        for e in range(num_epochs):
            self._train_loop()
            self._test()

    def _train_loop(self):
        size = len(self._train_dataloader.dataset)
        self._net.train()
        for batch, (X, y) in enumerate(self._train_dataloader):
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = self._net(X)
            loss = self._loss_fn(pred, y)

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test(self):
        size = len(self._test_dataloader.dataset)
        num_batches = len(self._test_dataloader)
        self._net.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self._test_dataloader:
                X, y = X.to(self._device), y.to(self._device)
                pred = self._net(X)
                test_loss += self._loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
