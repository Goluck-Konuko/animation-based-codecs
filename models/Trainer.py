import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .common.train_utils import Vgg19, ImagePyramide ,\
                            Transform
def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

class TrainerModel(nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    def __init__(self, kp_detector=None, generator=None, discriminator=None, config=None):
        super(TrainerModel, self).__init__()
        self.kp_detector = kp_detector
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_params = config['train_params']
        self.num_reference_frames = config['dataset_params']['num_references']
        self.scales = self.train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.mse = nn.MSELoss(reduction='none')
        self.loss_weights = self.train_params['loss_weights']
        self.vgg = Vgg19()

    def compute_perp_loss(self, real,generated):
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(generated)
        value_total = 0.0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
            for i, _ in enumerate(self.loss_weights['perceptual']):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach())
                value_total += (self.loss_weights['perceptual'][i] * value).mean()
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
    
    def ft_l1_loss(self,d_real_ft,d_gen_ft):
        value = torch.abs(d_real_ft - d_gen_ft).mean()
        return  value *self.loss_weights['feature_matching'][-1]**2
    
    def l1_loss(self, frame_1, frame_2):
        return torch.abs(frame_1 - frame_2).mean()
    

    def forward(self, x, **kwargs):
        anim_params = {**x}
        anim_params.update({'num_references':self.num_reference_frames})
        #get reference frame KP
        num_targets = len([i for i in list(x.keys()) if 'target' in i])
        kp_targets = {}
        for t in range(num_targets):
            kp_target = self.kp_detector(x[f'target_{t}'])
            anim_params.update({f"kp_target_{t}":kp_target})
            kp_targets.update({f"kp_target_{t}":kp_target})
            #print(torch.cuda.memory_allocated()/1e6)
        anim_params.update({'num_targets': num_targets})

        # if self.config['model_params']['generator_params']['residual_coding']:
        #     res_coding_params = self.config['model_params']['generator_params']['residual_coder_params']
        #     if res_coding_params['variable_bitrate']:
        #         rate_idx = np.random.choice(range(res_coding_params['levels']))
        #     else:
        #         rate_idx = self.config['train_params']['target_lambda']
        #     rd_lambda = self.config['train_params']['rd_lambda'][rate_idx]
        if 'con_iframe_params' in self.config['model_params']['generator_params']:
            con_iframe_params = self.config['model_params']['generator_params']['con_iframe_params']
            if con_iframe_params['variable_bitrate']:
                rate_idx = np.random.choice(range(con_iframe_params['levels']))
            else:
                rate_idx = self.config['train_params']['target_lambda']
            rd_lambda = self.config['train_params']['rd_lambda'][rate_idx]
        else:
            rd_lambda = self.config['train_params']['target_lambda']
            rate_idx = 0

        anim_params.update({'rate_idx': rate_idx})
        
        kp_reference_frames ={}
        pose_dist = []
        for idx in range(self.num_reference_frames):
            kp_ref = self.kp_detector(x[f'reference_{idx}'])
            kp_reference_frames.update({f'kp_reference_{idx}': kp_ref})

        anim_params.update(**kp_reference_frames)

        generated = self.generator(**anim_params)
        #print(torch.cuda.memory_allocated()/1e6)

        generated.update({**kp_reference_frames, **kp_targets})

        loss_values = {}
        perceptual_loss = 0.0
        bpp = 0.0
        gen_gan = 0.0
        ft_matching_loss =0.0
        equivariance_loss = 0.0
        perp_ref_loss = 0.0
        enh_perceptual_loss = 0.0
        mse_loss = 0.0

        if self.generator.ref_coder:
            #compute the perceptual loss for reference frame reconstruction

            for i in range(self.num_reference_frames):
                #compute RD Loss
                mse = self.mse(anim_params[f'reference_{idx}'],generated[f'reference_{idx}'])
                mse_loss += rd_lambda * 255**2 * mse.mean() 
            rd_loss = mse_loss + generated[f'rate']
            if rd_loss>0:
                loss_values['rd_loss'] = (rd_loss/self.num_reference_frames)*self.loss_weights['rd_loss']

        else:
            for i in range(num_targets):
                # generated.update({f"kp_target_{i}": anim_params[f"kp_target_{i}"]})

                target = x[f'target_{i}']
                B,_,H,W = target.shape

                animated_frame = generated[f'prediction_{i}']
                
                perceptual_loss +=  self.compute_perp_loss(target,animated_frame)          

                if f"enhanced_prediction_{i}" in generated:
                    #compute the rd loss
                    enh_pred = generated[f'enhanced_prediction_{i}']
                    enh_perceptual_loss +=  rd_lambda*self.compute_perp_loss(target,enh_pred)   
                    bpp += generated[f'rate_{i}']        

                #compute the gan losses
                gan_value, ft_matching_value = self.compute_ms_gan_loss(target, animated_frame, kp_target)       
                gen_gan += gan_value
                ft_matching_loss += ft_matching_value

                if  self.loss_weights['equivariance_value'] != 0:
                    transform = Transform(target.shape[0], **self.train_params['transform_params'])
                    transformed_frame = transform.transform_frame(target)
                    transformed_kp = self.kp_detector(transformed_frame)

                    ## Value loss part
                    value = torch.abs(kp_target['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                    equivariance_loss += self.loss_weights['equivariance_value'] * value

        # if self.generator.ref_coder:
        #     #compute an RD loss
        #     loss_values['ref_perceptual_loss'] = (rd_lambda*perp_ref_loss)/(self.num_reference_frames)
        #     loss_values['bpp'] =  generated[f'rate']/(self.num_reference_frames)
        
        
            #print(torch.cuda.memory_allocated()/1e6)
            loss_values['perceptual_loss'] = perceptual_loss/num_targets
            if enh_perceptual_loss>0:
                loss_values['enh_perceptual_loss'] = enh_perceptual_loss/num_targets
                loss_values['bpp'] = bpp/num_targets

            loss_values['gen_gan'] = gan_value/num_targets 
            loss_values['feature_matching'] = ft_matching_value/num_targets
            loss_values['equivariance_value'] =  equivariance_loss/num_targets
            if 'contrastive_loss' in generated:
                loss_values['contrastive_loss'] = (generated['contrastive_loss']*self.loss_weights['contrastive_loss'])/(self.num_reference_frames-1)
        return loss_values, generated




class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=[1e-2], vbr=False, metric = 'mse', quality_level=0):
        super().__init__()
        # if metric == 'mse':
        self.mse = nn.MSELoss(reduction='none')
        self.quality_level=quality_level

        # else:
        #     self.distortion = None # We can use ms-ssim
        self.vbr = vbr
        if isinstance(lmbda, list):
            self.lmbda_list = lmbda
            self.lmbda = self.lmbda_list[quality_level]
        else:
            self.lmbda = lmbda

    def get_rd_idx(self):
        if self.vbr:
            br_levels = len(self.lmbda_list)
            cur_lmbda_idx = np.random.choice(range(br_levels))
            self.lmbda = self.lmbda_list[cur_lmbda_idx]
            self.rate_idx = cur_lmbda_idx
            return cur_lmbda_idx
        else:
            return self.quality_level
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target,lambda_val, bpp_loss=0.0, mask=None, psnr=False):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        mse = self.mse(output, target)()
        rd_loss = lambda_val * 255**2 * mse.mean() + bpp_loss
        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)

        return out
