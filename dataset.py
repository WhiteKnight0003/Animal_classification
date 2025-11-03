from torch.utils.data import Dataset

class AnimalDataset(Dataset):
    def __init__(self, images ,labels, transforms=None):
        super().__init__()

        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transforms:
            image = self.transforms(image)
            
        return image, label       