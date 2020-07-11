from torchvision.datasets.folder import default_loader

from .flat_image_folder import FlatImageFolder


class CelebAHQ(FlatImageFolder):
    def __init__(self, root, image_size=1024, transform=None, loader=default_loader, train=True, download=True):
        root += "/celebAHQ/data" + str(image_size) + "x" + str(image_size)
        super().__init__(root, transform, loader)
