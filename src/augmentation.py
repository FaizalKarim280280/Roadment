import albumentations as A

class Augmentaton:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_augmentation(self, h_flip_prob, blur_prob, blur_limit):
        transform = A.compose([
            A.HorizontalFlip(p = h_flip_prob),
            A.Blur(blur_limit = blur_limit, p = blur_prob)
            # A.RandomCrop(height = self.input_shape[0], width = self.input_shape[1], p = 1)
        ])

        return transform


