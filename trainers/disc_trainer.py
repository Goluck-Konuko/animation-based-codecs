import torch
import torch.nn as nn
import torch.nn.functional as F


class DACDiscriminatorFullModel(nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    def __init__(self, discriminator=None, train_params=None, **kwargs):
        super(DACDiscriminatorFullModel, self).__init__()
        self.discriminator = discriminator
        self.train_params = train_params
        self.disc_type = kwargs['disc_type']
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, 3)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.loss_weights = train_params['loss_weights']

    def compute_multiscale(self, real, decoded, kp_target):
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(decoded)
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_target))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_target))

        loss = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            loss += self.loss_weights['discriminator_gan'] * value.mean()
        return loss


    def _non_saturating_loss(self,D_real_logits, D_gen_logits):
        D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,
            target=torch.ones_like(D_real_logits))
        D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,
            target=torch.zeros_like(D_gen_logits))
        D_loss = D_loss_real + D_loss_gen
        return D_loss

    def compute_patch_disc(self, real, decoded, context, model='disc'):
        disc_out = self.discriminator(torch.cat([real, decoded], dim=1), context)
        loss = self._non_saturating_loss(disc_out.d_real_logits, disc_out.d_gen_logits)
        return loss

    def forward(self, x, generated, model='discriminator'):
        loss = 0.0
        num_targets = len([tgt for tgt in x.keys() if 'target' in tgt])
        for idx in range(num_targets):
            real = x[f'target_{idx}']
            prediction = generated[f'prediction_{idx}'].detach()

            if self.disc_type == 'multi_scale':
                loss += self.compute_multiscale(real, prediction, generated[f'kp_target_{idx}'])
            else:
                loss += self.compute_patch_disc(real, prediction,generated[f'context_{idx}'],'disc')
        loss = (loss/num_targets)*self.train_params['loss_weights']['discriminator_gan']
        return {'disc_gan': loss}
    


class CFTEDiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator,videocompressor, train_params):
        super(CFTEDiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.videocompressor= videocompressor
        
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['target_0'])
        pyramide_generated = self.pyramid(generated['prediction_0'].detach())
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)
        
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values


class MRDiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(MRDiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator

        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['target_0'])
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



class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ], indexing='xy'
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale),mode='bilinear', align_corners=True)

        return out
        

class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

