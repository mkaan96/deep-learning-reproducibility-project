from torch.utils.data.dataset import Dataset


class LazyDataset(Dataset):
    def __init__(self, list_of_filenames):
        super().__init__()
        self.data_files = list_of_filenames

    def __getitem__(self, item):
        return self.load_file(self.data_files[item])

    def __len__(self):
        return len(self.data_files)

    def load_file(self, filename):
        return filename
