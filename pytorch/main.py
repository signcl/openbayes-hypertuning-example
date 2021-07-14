import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import openbayestool
import argparse

parser = argparse.ArgumentParser(description="hypertuning")
parser.add_argument("--input", help="input")
parser.add_argument("--filters", help="filters")
parser.add_argument("--nn", help="nn")
parser.add_argument("--opt", help="opt")
args = parser.parse_args()
print(args.filters, args.nn, args.opt)

dense_num = int(float(args.nn))
# number of convolutional filters to use
nb_filters = int(float(args.filters))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.MNIST(
    root=args.input,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)
test_data = torchvision.datasets.MNIST(
    root=args.input, train=False, transform=torchvision.transforms.ToTensor()
)
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=16)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, nb_filters, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(nb_filters, nb_filters * 2, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(nb_filters * 2, nb_filters * 2, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(nb_filters * 2 * 3 * 3, dense_num),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_num, 10),
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


model = Net()
model.to(device)
print(model)

if args.opt == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
elif args.opt == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
elif args.opt == "Adadelta":
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=0.01, rho=0.9, eps=1e-06, weight_decay=0
    )


loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    print("epoch {}".format(epoch + 1))
    # training-----------------------------
    train_loss = 0.0
    train_acc = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        "Train Loss: {:.6f}, Acc: {:.6f}".format(
            train_loss / (len(train_data)), train_acc / (len(train_data))
        )
    )

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(
            batch_y, volatile=True
        )
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print(
        "Test Loss: {:.6f}, Acc: {:.6f}".format(
            eval_loss / (len(test_data)), eval_acc / (len(test_data))
        )
    )
    openbayestool.log_metric("loss", eval_loss / (len(test_data)))
    openbayestool.log_metric("acc", eval_acc / (len(test_data)))
