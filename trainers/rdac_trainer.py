import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.common.train_utils import Vgg19,ImagePyramide,Transform, detach_kp

class GeneratorFullModel(nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    def __init__(self, kp_detector=None, generator=None, discriminator=None, config=None):
        super(GeneratorFullModel, self).__init__()
        self.kp_detector = kp_detector
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_params = config['train_params']
        self.num_target_frames = config['dataset_params']['num_sources']-1
        self.scales = self.train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.mse = nn.MSELoss()

        self.loss_weights = self.train_params['loss_weights']
        self.vgg = Vgg19()

    def compute_perp_loss(self, real,generated):
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

    
    def compute_ms_gan_loss(self, real, generated, kp_target):
        gen_gan, feature_matching = 0,0
        
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(generated)
                
        if self.loss_weights['generator_gan'] != 0 and self.discriminator != None:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_target))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_target))

            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                if not torch.isnan(value):
                    value_total += self.loss_weights['generator_gan'] * value
            gen_gan = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                feature_matching += value_total
        return gen_gan, feature_matching


    def gram_matrix(self, frame):
        # get the batch size, channels, height, and width of the image
        (bs, ch, h, w) = frame.size()
        f = frame.view(bs, ch, w * h)
        G = f.bmm(f.transpose(1, 2)) / (ch * h * w)
        return G
    
    def compute_style_loss(self, real, generated):
        #we want the style of the generated frame to match the real image
        style_fts = self.vgg(real)
        gen_fts = self.vgg(generated)

        # Get the gram matrices
        style_gram = [self.gram_matrix(fmap) for fmap in style_fts]
        gen_gram = [self.gram_matrix(fmap) for fmap in gen_fts]
        style_loss = 0.0
        for idx, gram in enumerate(style_gram):
            style_loss += self.mse(gen_gram[idx], gram)
        return style_loss

    def non_saturating_loss(self,D_gen_logits):
        G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,
            target=torch.ones_like(D_gen_logits))
        return G_loss  


    def forward(self, x, **kwargs):
        anim_params = {**x}
        anim_params.update({'num_targets':self.num_target_frames})
        #get reference frame KP
        kp_reference = self.kp_detector(x['reference'])
        anim_params.update({"kp_reference":kp_reference})
        if kwargs['variable_bitrate']:
            rate_idx = np.random.choice(range(kwargs['bitrate_levels']))
            rd_lambda_value = self.config['train_params']['rd_lambda'][rate_idx]
        else:
            rate_idx =  self.config['train_params']['target_rate']
            rd_lambda_value  = self.config['train_params']['rd_lambda'][rate_idx]

        anim_params.update({f'rd_lambda_value': rd_lambda_value , f'rate_idx':rate_idx})

        kp_targets ={}
        for idx in range(self.num_target_frames):
            kp_target_prev = self.kp_detector(x[f'target_{idx}'])
            kp_targets.update({f'kp_target_{idx}': kp_target_prev})
        
        anim_params.update(**kp_targets)
        B,C,H,W = x['reference'].shape
        
        generated = self.generator(**anim_params)
        generated.update({'kp_reference': kp_reference, **kp_targets})
        loss_values = {}

        rd_loss = 0.0
        rate = 0.0
        perceptual_loss = 0.0
        enh_perp_loss = 0.0
        ref_perp_loss = 0.0
        style_loss = 0.0
        gen_gan = 0.0
        feature_matching = 0.0
        equivariance_value = 0.0

        for idx in range(self.num_target_frames):
            tgt = self.num_target_frames-1 
            
            target = x[f'target_{idx}']
            # print("Target: ", target.shape)
            animated_frame = generated[f'prediction_{idx}']
            perceptual_loss += self.compute_perp_loss(target,animated_frame)
            if f'bpp_{idx}' in generated:
                perceptual_loss += 10*generated[f'bpp_{idx}']
                
            if self.loss_weights['style_loss'] != 0:
                style_loss += self.compute_style_loss(target, animated_frame)*self.loss_weights['style_loss']
    

            if f'sr_prediction_{idx}' in generated:
                #this loss applies to the refinement network
                ref_prediction = generated[f'sr_prediction_{idx}']
                ref_perp_loss += self.compute_perp_loss(target,ref_prediction)
            
            
            if self.loss_weights['style_loss'] != 0:
                style_loss += self.compute_style_loss(target, animated_frame)*self.loss_weights['style_loss']


            if f'enhanced_prediction_{idx}' in generated:
                #perceptual loss after residual coding
                enh_prediction = generated[f'enhanced_prediction_{idx}']
                enh_perp = self.compute_perp_loss(target,enh_prediction)
                
                rd_loss += anim_params[f'rd_lambda_value'] * enh_perp + generated[f"rate_{idx}"] #+ 10
                rate += generated[f"rate_{tgt}"]
                enh_perp_loss += enh_perp

            #compute the gan losses
            if self.discriminator.disc_type == 'multi_scale':
                gan_value, ft_matching_value = self.compute_ms_gan_loss(target, animated_frame, kp_targets[f"kp_target_{idx}"])       
                gen_gan += gan_value 
                feature_matching += ft_matching_value
            else:
                disc_out = self.discriminator(torch.cat([target, animated_frame], dim=1), generated[f'context_{idx}'])
                gen_gan += self.non_saturating_loss(disc_out.d_gen_logits)

            if self.loss_weights['equivariance_value'] != 0:
                transform = Transform(target.shape[0], **self.train_params['transform_params'])
                transformed_frame = transform.transform_frame(target)
                transformed_kp = self.kp_detector(transformed_frame)

                ## Value loss part
                if self.loss_weights['equivariance_value'] != 0:
                    value = torch.abs(kp_targets[f"kp_target_{idx}"]['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                    equivariance_value += self.loss_weights['equivariance_value'] * value            

        
        loss_values['perceptual'] = perceptual_loss/self.num_target_frames
        if rd_loss>0:
            loss_values['rd_loss'] = rd_loss/self.num_target_frames

        if ref_perp_loss>0:
            loss_values['ref_perceptual_loss'] = (ref_perp_loss)/self.num_target_frames
        
        if enh_perp_loss>0:
            generated['perp_distortion'] = (enh_perp_loss)/self.num_target_frames
        
        if rate>0:
            generated['rate'] = rate/self.num_target_frames

        if gen_gan >0:
            loss_values['gen_gan'] = gen_gan/self.num_target_frames

        if feature_matching >0:
            loss_values['feature_matching'] = feature_matching/self.num_target_frames

        if equivariance_value > 0:
            loss_values['equivariance_value'] = equivariance_value/self.num_target_frames

        if style_loss>0:
            loss_values['style_loss'] = style_loss/self.num_target_frames

        return loss_values, generated


