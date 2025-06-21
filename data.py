import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torchvision.transforms.functional as F
from PIL import Image

transform = transforms.Compose(
        [transforms.ToTensor(), #Coverts to Pytorch tensor
        transforms.Normalize((0.5, ), (0.5, )) # with mean 0.5, std 0.5 greyscale image E(0, 1) is coverted to E(-1, 1)
        ])

train_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
val_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sanity_test", type=int, default=1, help="1 to invoke sanity test 0 to stop")

    args = parser.parse_args()
    if args.sanity_test:
        data_iter = iter(train_loader)
        images, labels = next(data_iter)

        img_grid = torchvision.utils.make_grid(images)
        PIL_image = F.to_pil_image(img_grid)
        PIL_image.save("sanity_test.png")
        print('  '.join(classes[labels[j]] for j in range(4)))
