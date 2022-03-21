
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision import transforms


class ColorJitter_Hue(transforms.ColorJitter):
    def __init__(self, hue=0.):
        super().__init__(hue=hue)


class ColorJitter_Bright_High(transforms.ColorJitter):
    def __init__(self, brightness=0.):
        super().__init__(brightness=brightness)


class ColorJitter_Bright_Low(transforms.ColorJitter):
    def __init__(self, brightness=0.):
        super().__init__(brightness=brightness)


class ColorJitter_Contr_High(transforms.ColorJitter):
    def __init__(self, contrast=0.):
        super().__init__(contrast=contrast)


class ColorJitter_Contr_Low(transforms.ColorJitter):
    def __init__(self, contrast=0.):
        super().__init__(contrast=contrast)


class ColorJitter_Satur_High(transforms.ColorJitter):
    def __init__(self, saturation=0.):
        super().__init__(saturation=saturation)


class transform_PILtoTensor(torch.nn.Module):
    """
    unlike transforms.ToTensor doesnt change the orders of the channels
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img: PIL.Image):
        """
        Args:
            img (PIL.Image): PIL image to be converted to torch.Tensor

        Returns:
            img (torch.Tensor): tensor image.
        """
        return torch.tensor(np.array(img))


class transform_TensortoNumpy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, img: torch.Tensor):
        """
        Args:
            img (torch.Tensor): Tensor image to be converted to numpy.array

        Returns:
            img (numpy.array): numpy image.
        """
        return img.detach().cpu().numpy()


AUGS_DICT = {

    # region General
    'h_flip': transforms.RandomHorizontalFlip(p=1.),
    'affine': transforms.RandomAffine(degrees=(-40, 40), fill=255),
    'random_erasing': transforms.Compose(
        [transform_PILtoTensor(), transforms.RandomErasing(p=1.), transform_TensortoNumpy()]),
    'hue_rot': transforms.RandomChoice([ColorJitter_Hue((0.2, 0.5)), ColorJitter_Hue((-0.5, -0.2))]),
    'low_bright': ColorJitter_Bright_Low(brightness=(0.2, 0.6)),
    'high_bright': ColorJitter_Bright_High(brightness=(1.4, 1.8)),
    'low_contrast': ColorJitter_Contr_Low(contrast=(0.3, 0.8)),
    'high_contrast': ColorJitter_Contr_High(contrast=(1.8, 3)),
    'grayscale': transforms.RandomGrayscale(p=1.),
    'high_satur': ColorJitter_Satur_High(saturation=(1.8, 3.)),
    'gblurr': transforms.GaussianBlur(kernel_size=5, sigma=(1., 1.)),
    'v_flip': transforms.RandomVerticalFlip(p=1.),
    'crop': transforms.RandomResizedCrop((64, 64), scale=(0.5, 1.0)),

    # region SimSiam
    'simsiam_crop': transforms.RandomResizedCrop((64, 64), scale=(0.6, 1.)),
    'simsiam_cjitter': transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.3),
    'simsiam_grayscale': transforms.RandomGrayscale(p=0.2),
    'simsiam_gblurr': transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=[1., 2.])], p=0.2),
    'simsiam_hflip': transforms.RandomHorizontalFlip(),
    # endregion SimSiam

}

simsiam_augmentation = ['simsiam_crop', 'simsiam_hflip', 'simsiam_cjitter', 'simsiam_grayscale',
                        'simsiam_gblurr']
simsiam_augmentation = transforms.Compose([AUGS_DICT[x] for x in simsiam_augmentation])
AUGS_DICT['simsiam'] = simsiam_augmentation

