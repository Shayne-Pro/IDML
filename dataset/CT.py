from .base import *

class CT(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/CT'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = [0,1]
        elif self.mode == 'eval':
            self.classes = [2,3]
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root)).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes :
                self.ys += [y]  # 标签
                self.I += [index]  # 索引
                self.im_paths.append(os.path.join(self.root, i[0]))  # 图片路径
                index += 1

