"""Image replay buffer for stabilizing GAN training."""
import random
import torch


class ImagePool:
    """Stores previously generated images for discriminator training.

    With 50% probability, returns a previously stored image instead of
    the current one, following Shrivastava et al. (2017).
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        """Return images from the pool, replacing some with new ones."""
        if self.pool_size == 0:
            return images

        result = []
        for img in images:
            img = img.unsqueeze(0)  # (1, C, H, W)
            if len(self.images) < self.pool_size:
                self.images.append(img.clone())
                result.append(img)
            else:
                if random.random() > 0.5:
                    # Return a random old image, store the new one
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = img.clone()
                    result.append(old)
                else:
                    result.append(img)

        return torch.cat(result, dim=0)
