from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pickle

class LazyDataset(Dataset):
    def __init__(self, list_of_filenames, batch_size=1):
        super().__init__(list_of_filenames, batch_size=batch_size)
        self.data_files = list_of_filenames

    def __getindex__(self, idx):
        return self.load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

    def load_file(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
