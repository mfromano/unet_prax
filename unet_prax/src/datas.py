from . import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(24*16)
])

TRAIN = MNIST(root='.', train=True, transform=transform)
TEST = MNIST(root='.', train=False, transform=transform)

class MnistData(Dataset):
    def __init__(self, train: bool):
        super().__init__()
        if train:
            self._data = TRAIN
        else:
            self._data = TEST
    
    def _binaryitem(self, idx):
        d = self._data[idx][0]
        return torch.where(d == 0, 0, 1)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        x = self._data[idx][0].to(torch.float32)
        y = self._binaryitem(idx).to(torch.float32)
        return x, y
