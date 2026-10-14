import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CombinedDataset(Dataset):
    def __init__(self, uieb_root, euvp_root, split='train', transform=None):
        self.transform = transform
        self.pairs = []
        # 加载 UIEB
        u_raw, u_ref = os.path.join(uieb_root, 'raw-890'), os.path.join(uieb_root, 'reference-890')
        u_imgs = sorted(os.listdir(u_raw))
        u_split = u_imgs[:800] if split == 'train' else u_imgs[800:]
        for n in u_split:
            self.pairs.append((os.path.join(u_raw, n), os.path.join(u_ref, n)))
        # 加载 EUVP (仅训练集)
        if split == 'train' and os.path.exists(euvp_root):
            e_raw, e_ref = os.path.join(euvp_root, 'trainA'), os.path.join(euvp_root, 'trainB')
            if os.path.exists(e_raw):
                for n in os.listdir(e_raw):
                    self.pairs.append((os.path.join(e_raw, n), os.path.join(e_ref, n)))
        print(f"[{split}] Total pairs: {len(self.pairs)}")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        pA, pB = self.pairs[idx]
        imgA, imgB = Image.open(pA).convert('RGB'), Image.open(pB).convert('RGB')
        if self.transform:
            imgA, imgB = self.transform(imgA), self.transform(imgB)
        return imgA, imgB