"""Dataset and image transforms."""
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class LesionDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(size, train, aug="strong"):
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if not train:
        return transforms.Compose([transforms.Resize((size, size))] + norm)
    if aug == "basic":                                   # what the capstone used
        return transforms.Compose([transforms.Resize((size, size)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10)] + norm)
    big = int(size * 1.25)                               # dermoscopy has no "up": rotate freely
    return transforms.Compose([
        transforms.Resize((big, big)),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.03),
    ] + norm)
