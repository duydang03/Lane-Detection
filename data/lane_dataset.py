class LaneSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      image = cv2.imread(self.image_paths[idx])
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      mask = cv2.imread(self.mask_paths[idx], 0)
      mask = (mask > 127).astype("float32")  # Binary mask
      if self.transform:
          augmented = self.transform(image=image, mask=mask)
          image = augmented['image']
          mask = augmented['mask']

      if isinstance(mask, np.ndarray):
          mask = torch.from_numpy(mask)
      mask = mask.unsqueeze(0).float()

      return image, mask
