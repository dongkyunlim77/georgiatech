import torch
from PIL import Image
from torchvision.transforms import v2


def create_training_transformations():
    """
    In this function, you are going to preprocess and augment training data.
    Use torchvision.transforms.v2 to do these transforms and the order of the transformations matter!

    First, convert the original PIL Images to tensors using v2.ToImage().
    Second, apply random rotation ranging from -10 degrees (clockwise) to 10 degrees (counterclockwise).
    Third, apply random affine transformation with up to 5% translation in width and height.
    Fourth, add random horizontal flip with a probability of 0.5.
    Finally, convert the tensor to dtype torch.float32 and scale values from [0, 255] to [0, 1].

    RETURN: torchvision.transforms.v2.Compose object
    """
    return v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(degrees=(-10,10)),
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])


def create_testing_transformations():
    """
    In this function, you are going to only preprocess testing data.
    Use torchvision.transforms.v2 to do these transforms and the order of the transformations matter!

    Convert the original PIL Images to Tensors
    (Hint): Do not directly use ToTensor() instead use v2.ToImage ,v2.ToDtype, and look at ToTensor documentation

    RETURN: torchvision.transforms.v2.Compose object
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        """
        A dataset that preprocesses and stores transformed images.
        """
        self.transform = transform
        self.data = []
        self.targets = torch.tensor(
            labels, dtype=torch.long
        )  # Convert labels to tensor

        # Apply transformation and store processed images
        for img in images:
            img = Image.fromarray(img.astype("uint8"))  # Convert to PIL image
            if self.transform:
                img = self.transform(img)  # Apply transformation

            self.data.append(img)  # Store transformed image

        # Convert list of transformed images to a single tensor for efficient indexing
        self.data = torch.stack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]  # Return transformed image and label
