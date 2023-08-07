import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
from .modules.train_utils import Vgg19, ImagePyramide, Transform, detach_kp

class TrainerModel(nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    def __init__(self, kp_extractor=None, generator=None, discriminator=None, config=None):
        super(TrainerModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_params = config['train_params']
        self.num_target_frames = config['dataset_params']['num_sources']-1
        self.scales = self.train_params['scales']
        if discriminator is not None:
            self.disc_scales = self.discriminator.scales
        else:
            self.disc_scales = [1]
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = self.train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def downsample(self, frame:torch.Tensor, sf:float=0.5)->torch.Tensor:
        return F.interpolate(frame, scale_factor=(sf, sf),mode='bilinear', align_corners=True)

    def upsample(self, frame: torch.Tensor) -> torch.Tensor:
        return F.interpolate(frame, scale_factor=(2, 2),mode='bilinear', align_corners=True)
        
    def compute_perp_loss(self, real: torch.Tensor,generated: torch.Tensor) -> torch.Tensor:
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(generated)
        value_total = 0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
            for i, _ in enumerate(self.loss_weights['perceptual']):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                value_total += self.loss_weights['perceptual'][i] * value
        return value_total
    
    def forward(self, x:Dict[Any], **kwargs):
        kp_target = self.kp_extractor(x['target_0'])
        anim_params = {'n_target':self.num_target_frames}

        for idx in range(self.num_target_frames):
            kp_target_prev = self.kp_extractor(x[f'target_{idx}'])
            anim_params.update({f'target_{idx}': x[f'target_{idx}'], f'kp_target_{idx}': kp_target_prev})
                    
        kp_reference = self.kp_extractor(x['reference'])
        anim_params.update({'kp_reference': kp_reference})


        generated = self.generator(x['reference'],**anim_params)
        generated.update({'kp_reference': kp_reference, 'kp_target_0': kp_target})
    
        loss_values = {}
        rd_loss = 0.0
        total_distortion = 0.0
        rate = 0.0
        perceptual_loss = 0.0
        enh_perp_loss = 0.0
        perp_distortion = 0.0
        
        for idx in range(self.num_target_frames):
            tgt = self.num_target_frames-1 
            perceptual_loss += self.compute_perp_loss(anim_params[f'target_{idx}'],generated[f'prediction_{idx}'])

            #compute rd loss
            if f'enhanced_pred_{idx}' in generated:
                rd_lambda = self.train_params['rd_lambda'][self.train_params['target_rate'] ]
                perp_distortion = self.compute_perp_loss( anim_params[f'target_{tgt}'],generated[f'enhanced_pred_{tgt}'])
                distortion = F.mse_loss(anim_params[f'target_{tgt}'],generated[f'enhanced_pred_{tgt}'])
                
                ##
                rd_loss += rd_lambda*perp_distortion+generated[f"rate_{tgt}"]
                total_distortion += distortion #
                enh_perp_loss += perp_distortion
                rate += generated[f"rate_{tgt}"]

        loss_values['perceptual'] = perceptual_loss/self.num_target_frames
        if f'enhanced_pred_0' in generated:
            loss_values['rd_loss'] = rd_loss/self.num_target_frames
            generated['distortion'] = total_distortion/self.num_target_frames
            generated['perp_distortion'] = enh_perp_loss/self.num_target_frames
            generated['rate'] = rate/self.num_target_frames

        if 'enhanced_pred_0' in generated:
            pyramide_generated = self.pyramid(generated['enhanced_pred_0'])
        else:
            pyramide_generated = self.pyramid(generated['prediction_0'])
        pyramide_real = self.pyramid(x['target_0'])

        if self.loss_weights['generator_gan'] != 0 and self.discriminator != None:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_target))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_target))

            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                if not torch.isnan(value):
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
                        if not torch.isnan(value) and not torch.isinf(value):
                            value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['target_0'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['target_0'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_target['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.config['model_params']['common_params']['estimate_jacobian'] and self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_target = torch.inverse(kp_target['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_target, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
            
        return loss_values, generated


