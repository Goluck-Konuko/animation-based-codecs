import torch
import numpy as np
from models.common.train_utils import Vgg19,ImagePyramide,Transform, detach_kp


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, config, **kwargs):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_params = config['train_params']
        self.scales = self.train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
            self.vgg = self.vgg.cuda()
        self.loss_weights = self.train_params['loss_weights']


    def forward(self, x, **kwargs):
        num_references = len([i for i in x.keys() if 'reference' in i])
        anim_params = {**x, 'num_references':num_references, 'num_targets':1}

        if self.config['model_params']['generator_params']['ref_coder']:
            #check the range of bitrate levels in the image coder and select one randomly
            ref_coder_params = self.config['model_params']['generator_params']['iframe_params']
            rate_idx = np.random.choice(range(ref_coder_params['levels']))
        else:
            rate_idx = 0
        anim_params.update({'rate_idx': rate_idx})

        for idx in range(num_references):
            kp_ref = self.kp_extractor(x[f'reference_{idx}'])
            anim_params.update({f"kp_reference_{idx}": kp_ref})
        kp_target = self.kp_extractor(x['target_0'])
        anim_params.update({f'kp_target_0': kp_target})
        generated = self.generator(**anim_params)
        generated.update({**anim_params})
        loss_values = {}

        pyramide_real = self.pyramid(x['target_0'])
        pyramide_generated = self.pyramid(generated['prediction_0'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_target))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_target))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if self.loss_weights['equivariance_value'] != 0:
            transform = Transform(x['target_0'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['target_0'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            value = torch.abs(kp_target['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

        return loss_values, generated

