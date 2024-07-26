import torch
import torchvision
import torchvision.transforms as transforms

def get_mnist_data(batch_size):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # Resize to 32x32
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    return trainloader

def get_fashion_mnist_data(batch_size):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # Resize to 32x32
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    return trainloader

if __name__ == "__main__":
    batch_size = 64

    # Get MNIST data
    mnist_loader = get_mnist_data(batch_size)
    mnist_iter = iter(mnist_loader)
    images, labels = next(mnist_iter)
    print(f"MNIST batch shape: {images.size()}")

    # Get Fashion MNIST data
    fashion_mnist_loader = get_fashion_mnist_data(batch_size)
    fashion_mnist_iter = iter(fashion_mnist_loader)
    images, labels = next(fashion_mnist_iter)
    print(f"Fashion MNIST batch shape: {images.size()}")
