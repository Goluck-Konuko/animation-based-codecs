import os
import torch
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import utils
import models

class PrepareOptimizers:
    def __init__(self, generator, kp_detector=None, step_wise=False) -> None:
        self.step_wise= step_wise #If true, elements from the previous stage are frozen
        self.generator = generator
        self.kp_detector = kp_detector

    def get_animation_optimizer(self, config={'lr':2e-4}, betas=(0.5, 0.999)):
        #No previous stage so nothing to freeze
        parameters = list(self.generator.parameters()) + list(self.kp_detector.parameters())
        optimizer = torch.optim.Adam(parameters, lr=config['lr'], betas=betas)
        return optimizer
    
    def get_rdac_optimizer(self, config={'lr':1e-4, 'lr_aux':1e-3},temporal=False, betas=(0.5, 0.999)):
        if self.step_wise:
            for param in self.generator.parameters():
                param.requires_grad = False

            for param in self.kp_detector.parameters():
                param.requires_grad = False

            if temporal:
                net = self.generator.tdc.train()
            else:
                net = self.generator.sdc.train()

            parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
            aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles")) 

            for param in parameters:
                param.requires_grad = True

            for param in aux_parameters:
                param.requires_grad = True
                
        else:
            gen_params = set(p for n, p in self.generator.named_parameters() if not n.endswith(".quantiles")) 
            parameters = list(gen_params)+ list(self.kp_detector.parameters())

            if temporal:
                net = self.generator.tdc.train()
            else:
                net = self.generator.sdc.train()

            aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        
        optimizer = torch.optim.AdamW(parameters, lr=config['lr'])
        aux_optimizer = torch.optim.AdamW(aux_parameters, lr=config['lr_aux'])
        return optimizer, aux_optimizer

def load_pretrained_model(model, path ,name: str='generator', device:str='cpu'):
    cpk = torch.load(path, map_location=device)
    model.load_state_dict(cpk[name], strict=False)
    return model


def train(config,dataset,generator, kp_detector,discriminator,**kwargs ):
    train_params = config['train_params'] 
    debug = kwargs['debug'] 
    adversarial_training = train_params['adversarial_training']

    # create optimizers for generator, kp_detector and discriminator
    step = kwargs['step']
    step_wise = config['train_params']['step_wise']
    prepare = PrepareOptimizers(generator, kp_detector, step_wise=step_wise)

    if step == 0:
        optimizer = prepare.get_animation_optimizer()
        aux_optimizer = None
    elif step == 1:
        # Train the spatial difference coder
        optimizer, aux_optimizer = prepare.get_rdac_optimizer()
    elif step in [2,3]:
        # Train TDC but with/without motion compensation using deformation predicted
        optimizer, aux_optimizer = prepare.get_rdac_optimizer(temporal=True)
    else:
        raise NotImplementedError("Unknown training step")

    start_epoch = 0
    if adversarial_training:
        optimizer_discriminator = torch.optim.Adam(list(discriminator.parameters()), lr=train_params['lr_discriminator'],  betas=(0.5, 0.999))  
    else:
        optimizer_discriminator = None

    if kwargs['checkpoint'] is not None:
        start_epoch = utils.Logger.load_cpk(kwargs['checkpoint'],generator, discriminator, kp_detector,
                                    optimizer, optimizer_discriminator)
           
    scheduler_generator = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    generator_full = models.TrainerModel(kp_detector, generator, discriminator,config) 
    
    if adversarial_training:
        scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
        discriminator_full = models.DiscriminatorModel(discriminator, train_params) 

    if torch.cuda.is_available():
        generator_full = generator_full.cuda()
        if adversarial_training:
            discriminator_full = discriminator_full.cuda()

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = utils.DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with utils.Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if torch.cuda.is_available():
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}
                losses_generator, generated = generator_full(x, **params)
                losses_ = {} 
                if 'distortion' in generated:
                    losses_.update({'distortion': generated['distortion'].mean().detach().data.cpu().numpy().item(),
                                    'rate':generated['rate'].mean().detach().data.cpu().numpy().item()})
                if 'perp_distortion' in generated:
                    losses_.update({'perp_distortion':generated['perp_distortion'].mean().detach().data.cpu().numpy().item()})
                
                loss_values = [val.mean() for val in losses_generator.values()]
                
                loss = sum(loss_values) 
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

                if aux_optimizer is not None:
                    if step==1:
                        aux_loss = generator.sdc.aux_loss()
                    elif step in [2,3]:
                        aux_loss = generator.tdc.aux_loss()

                    aux_loss.backward()
                    aux_optimizer.step()
                    aux_optimizer.zero_grad()
                else:
                    aux_loss = 0

                if adversarial_training:
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    disc_loss = sum(loss_values)
                    disc_loss.backward()
                    
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                if aux_loss>0:
                    losses.update({"aux_loss":aux_loss.mean().detach().data.cpu().numpy().item()})
                logger.log_iter(losses=losses)
                
                if debug:
                    break

            scheduler_generator.step()
            if adversarial_training:
                scheduler_discriminator.step()

            state_dict = {
                          'generator': generator,
                          'kp_detector': kp_detector
                        }
            
            if adversarial_training:
                state_dict.update({'discriminator':discriminator})

            logger.log_epoch(epoch, state_dict, inp=x, out=generated)
            if debug:
                break
          
