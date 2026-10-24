from .base import *


class CT(BaseDataset):
    def __init__(self, root, mode, transform=None):
        if mode not in ('train', 'eval'):
            raise ValueError('CT mode must be train or eval')
        self.classes = [0, 1]
        BaseDataset.__init__(self, os.path.join(root, 'CT'), mode, transform)
        folders = {'COVID_tr': 0, 'nonCOVID_tr': 1} if mode == 'train' else {'COVID_ts': 0, 'nonCOVID_ts': 1}
        source = torchvision.datasets.ImageFolder(root=self.root)
        for path, folder_index in source.imgs:
            folder = source.classes[folder_index]
            if folder in folders and not os.path.basename(path).startswith('._'):
                self.I.append(len(self.ys))
                self.ys.append(folders[folder])
                self.im_paths.append(path)
        if set(self.ys) != set(self.classes):
            raise ValueError(f'CT {mode} requires nonempty folders: {list(folders)}')
