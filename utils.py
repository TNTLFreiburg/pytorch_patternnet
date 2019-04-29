import torch
from torch.autograd import Variable
import time
import torchvision
import torchvision.transforms as transforms
import PIL


def load_data_mnist(batchsize):
    # load Cifar10 data set
    # define transform of data files
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # load trainset and define trainloader
    trainset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=2
    )

    # load testset and define testloader
    testset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False, num_workers=2
    )

    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    return trainloader, testloader, classes


def train(net, num_epochs, trainloader, criterion, optimizer, gpu=False, opt_steps=100):
    if gpu:
        net = net.cuda()
    trainloader = trainloader
    criterion = criterion
    optimizer = optimizer

    start_time = time.time()
    if num_epochs == None:
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i <= opt_steps:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(
                        labels.cuda()
                    )
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # net(inputs) calls forward from class Net
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 500 == 499:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f"
                        % (epoch + 1, i + 1, running_loss / 500)
                    )
                    running_loss = 0.0 
            else:
                break
    else:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(
                        labels.cuda()
                    )
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # net(inputs) calls forward from class Net
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 500 == 499:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f"
                        % (epoch + 1, i + 1, running_loss / 500)
                    )
                    running_loss = 0.0

    end_time = time.time()
    time_needed = int(end_time - start_time)
    print("Finished Training in %i seconds" % time_needed)


def accuracy(net, iterator, num_steps=None, device=torch.device("cpu")):

    total = 0.
    correct = 0.
    if num_steps is None:
        for data in iterator:
            inp, labels = data
            inp, labels = inp.to(device), labels.to(device)
            out = net(Variable(inp))
            _, predictions = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum()
    else:
        for _ in range(num_steps):
            inp, labels = next(iterator.__iter__())
            inp, labels = inp.to(device), labels.to(device)
            out = net(Variable(inp))
            _, predictions = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum()

    print("Accuracy: %f" % (int(correct) / int(total)))

    return (int(correct) / int(total))