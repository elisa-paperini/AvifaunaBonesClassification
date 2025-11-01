import torch
import torch.utils.data

class N_AugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps an existing PyTorch ImageFolder dataset to return N augmented 
    versions of a single sample per index.

    Args:
        dataset (torch.utils.data.Dataset): The base dataset (e.g., ImageFolder).
        n_aug (int): The number of augmented views to generate for each image.
    """
    def __init__(self, dataset, n_aug):
        self.dataset = dataset
        self.n_aug = n_aug

    def __len__(self):
        # The length of the wrapped dataset (number of original images)
        return len(self.dataset)

    def __getitem__(self, idx):
        # Save the original transform
        original_transform = self.dataset.transform
        # Temporarily set transform to None to retrieve the PIL image
        self.dataset.transform = None
        pil_img, target = self.dataset[idx]
        # Restore the full transformation
        self.dataset.transform = original_transform
        # Apply the full augmentation N times
        augmented_samples = []
        for _ in range(self.n_aug):
            aug_img = original_transform(pil_img)
            augmented_samples.append(aug_img)
        # Stack the N augmented images into a single tensor (N, C, H, W)
        aug_batch = torch.stack(augmented_samples, dim=0)
        
        return aug_batch, target