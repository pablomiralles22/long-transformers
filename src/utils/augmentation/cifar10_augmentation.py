import random
import torch.nn as nn
import torchvision.transforms.functional as tvF

def apply_op(img, op_name: str, magnitude: float):
    if op_name == "Brightness":
        img = tvF.adjust_brightness(img, magnitude)
    elif op_name == "Color":
        img = tvF.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = tvF.adjust_contrast(img, magnitude)
    elif op_name == "Sharpness":
        img = tvF.adjust_sharpness(img, magnitude)
    elif op_name == "Posterize":
        img = tvF.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = tvF.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = tvF.autocontrast(img)
    elif op_name == "Equalize":
        img = tvF.equalize(img)
    elif op_name == "Invert":
        img = tvF.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class RandomCifar10Augmentator(nn.Module):
    __AUGMENTATIONS = [
        ('Invert', 0.008),
        ('Contrast', 0.032),
        ('Sharpness', 0.11200000000000002),
        ('AutoContrast', 0.2),
        ('Equalize', 0.188),
        ('Posterize', 0.012),
        ('Color', 0.11599999999999999),
        ('Brightness', 0.092),
        ('Solarize', 0.076),
    ]

    __AUGMENTATION_SPACE = {
        'Brightness': (1.0, 1.9),
        'Color': (1.0, 1.9),
        'Contrast': (1.0, 1.9),
        'Sharpness': (1.0, 1.9),
        'Posterize': (0, 7),
        'Solarize': (0, 256),
    }

    def forward(self, img):
        for op_name, prob in self.__AUGMENTATIONS:
            if random.random() < prob:
                if op_name in self.__AUGMENTATION_SPACE:
                    magnitude = random.uniform(*self.__AUGMENTATION_SPACE[op_name])
                else:
                    magnitude = None
                img = apply_op(img, op_name, magnitude)
        return img