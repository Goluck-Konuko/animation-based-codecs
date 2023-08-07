import torch
import torch.nn as nn
from .modules.train_utils import ImagePyramide


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class DiscriminatorModel(nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    def __init__(self, discriminator=None, train_params=None):
        super(DiscriminatorModel, self).__init__()
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, 3)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['target_0'])
        
        if 'enhanced_pred_0' in generated:
            pyramide_generated = self.pyramid(generated['enhanced_pred_0'].detach())
        else:
            pyramide_generated = self.pyramid(generated['prediction_0'].detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(generated['kp_target_0']))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(generated['kp_target_0']))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total
        return loss_values
