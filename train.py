# -*- coding: utf-8 -*-
from tqdm import trange
import torch
import os
import models
from trainers import gen_trainers, disc_trainers
from utilities.utils.logger import Logger
from torch.utils.data import DataLoader
from utilities.utils.dataset import DatasetRepeater
from torch.optim.lr_scheduler import MultiStepLR

def load_pretrained_model(model, path ,name: str='generator', device:str='cpu'):
    cpk = torch.load(path, map_location=device)
    if name in cpk:
        if 'optimizer' in name:
            model.load_state_dict(cpk[name])
        else:
            model.load_state_dict(cpk[name],strict=False)
    return model
    

def train_dac(config,dataset,generator, kp_detector,discriminator,**kwargs):
    train_params = config['train_params'] 

    parameters = list(kp_detector.parameters()) + list(generator.parameters())
    
    gen_optimizer = torch.optim.AdamW(parameters, lr=train_params['lr'], betas=(0.5, 0.999))
    disc_optimizer = torch.optim.AdamW(list(discriminator.parameters()), lr=train_params['lr'],  betas=(0.5, 0.999))  
      
    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained_cpk_path = config['dataset_params']['cpk_path'] 
    if pretrained_cpk_path != '':
        #Retrain from a saved checkpoint specified in the config file
        generator = load_pretrained_model(generator, path=pretrained_cpk_path,name='generator', device=device)
        kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
        gen_optimizer = load_pretrained_model(gen_optimizer, path=pretrained_cpk_path,name='gen_optimzer', device=device)

        discriminator = load_pretrained_model(discriminator, path=pretrained_cpk_path,name='discriminator', device=device)
        disc_optimizer = load_pretrained_model(disc_optimizer, path=pretrained_cpk_path,name='disc_optimizer', device=device)
    
    if config['model_params']['generator_params']['ref_coder']:
        #load parameters of pretrained TIC model
        tic_weights = torch.load("checkpoints/tic.pth.tar", map_location=device, weights_only=True)
        generator.ref_coder.load_state_dict(tic_weights['tic'], strict=True)

    scheduler_generator = MultiStepLR(gen_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    scheduler_discriminator = MultiStepLR(disc_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
    
    
    generator_full = gen_trainers['dac'](kp_detector, generator, discriminator,config) 
    disc_type = config['model_params']['discriminator_params']['disc_type']
    discriminator_full = disc_trainers['dac'](discriminator, train_params, disc_type=disc_type) 


    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        generator_full = generator_full.cuda()
        discriminator_full = discriminator_full.cuda()
        if  torch.cuda.device_count()> 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in range(num_gpus)])
            generator_full = CustomDataParallel(generator_full)
            discriminator_full = CustomDataParallel(discriminator_full)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if device =='cuda':
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}
                losses_generator, generated = generator_full(x, **params)
                losses_ = {} 
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values) 
                
                loss.backward()
                
                gen_optimizer.step()
                gen_optimizer.zero_grad()

                #forward and backprop on the discriminator
                losses_discriminator = discriminator_full(x, generated)
                
                loss_values = [val.mean() for val in losses_discriminator.values()]
                disc_loss = sum(loss_values)
                disc_loss.backward()
                
                disc_optimizer.step()
                disc_optimizer.zero_grad()

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                logger.log_iter(losses=losses)
                if kwargs['debug']:
                    break

            scheduler_generator.step()
            scheduler_discriminator.step()

            state_dict = {'generator': generator,
                        'kp_detector': kp_detector, 
                        'gen_optimizer': gen_optimizer,
                        'discriminator':discriminator, 
                        'disc_optimizer':disc_optimizer}
            
            logger.log_epoch(epoch, state_dict,inp=x, out=generated)
            if kwargs['debug']:
                break


def train_mrdac(config,dataset,generator, kp_detector,discriminator,**kwargs):
    train_params = config['train_params'] 

    parameters = list(kp_detector.parameters()) + list(generator.parameters())
    
    gen_optimizer = torch.optim.AdamW(parameters, lr=train_params['lr'], betas=(0.5, 0.999))
    disc_optimizer = torch.optim.AdamW(list(discriminator.parameters()), lr=train_params['lr'],  betas=(0.5, 0.999))  
      
    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained_cpk_path = config['dataset_params']['cpk_path'] 
    if pretrained_cpk_path != '':
        #Retrain from a saved checkpoint specified in the config file
        generator = load_pretrained_model(generator, path=pretrained_cpk_path,name='generator', device=device)
        kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
        gen_optimizer = load_pretrained_model(gen_optimizer, path=pretrained_cpk_path,name='gen_optimzer', device=device)

        discriminator = load_pretrained_model(discriminator, path=pretrained_cpk_path,name='discriminator', device=device)
        disc_optimizer = load_pretrained_model(disc_optimizer, path=pretrained_cpk_path,name='disc_optimizer', device=device)
    
    if config['model_params']['generator_params']['ref_coder']:
        #load parameters of pretrained TIC model
        tic_weights = torch.load("checkpoints/tic.pth.tar", map_location=device, weights_only=True)
        generator.ref_coder.load_state_dict(tic_weights['tic'], strict=False)

    scheduler_generator = MultiStepLR(gen_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    scheduler_discriminator = MultiStepLR(disc_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
    
    
    generator_full = gen_trainers['mrdac'](kp_detector, generator, discriminator,config) 
    discriminator_full = disc_trainers['mrdac'](kp_detector, generator, discriminator, train_params) 


    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        generator_full = generator_full.cuda()
        discriminator_full = discriminator_full.cuda()
        if  torch.cuda.device_count()> 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in range(num_gpus)])
            generator_full = CustomDataParallel(generator_full)
            discriminator_full = CustomDataParallel(discriminator_full)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if device =='cuda':
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}
                losses_generator, generated = generator_full(x, **params)
                losses_ = {} 
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values) 
                
                loss.backward()
                
                gen_optimizer.step()
                gen_optimizer.zero_grad()

                #forward and backprop on the discriminator
                losses_discriminator = discriminator_full(x, generated)
                
                loss_values = [val.mean() for val in losses_discriminator.values()]
                disc_loss = sum(loss_values)
                disc_loss.backward()
                
                disc_optimizer.step()
                disc_optimizer.zero_grad()

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                logger.log_iter(losses=losses)
                if kwargs['debug']:
                    break

            scheduler_generator.step()
            scheduler_discriminator.step()

            state_dict = {'generator': generator,
                        'kp_detector': kp_detector, 
                        'gen_optimizer': gen_optimizer,
                        'discriminator':discriminator, 
                        'disc_optimizer':disc_optimizer}
            
            logger.log_epoch(epoch, state_dict,inp=x, out=generated)
            if kwargs['debug']:
                break

def train_rdac(config,dataset,generator, kp_detector,discriminator,**kwargs ):
    train_params = config['train_params'] 
    # create optimizers for generator, kp_detector and discriminator
    step = config['train_params']['step']
    aux_optimizer= None
    pretrained_cpk_path = config['dataset_params']['cpk_path'] #update the checkpoint path depending on the training step
    if step == 0:
        parameters = list(kp_detector.parameters())
        

        parameters = list(set(p for n, p in generator.named_parameters() if not n.endswith(".quantiles")))            
        aux_parameters = list(set(p for n, p in generator.named_parameters() if n.endswith(".quantiles"))) 

        aux_optimizer = torch.optim.Adam(aux_parameters, lr=train_params['lr_aux'])
        gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'],betas=train_params['betas'])

    elif step == 1:
        for param in generator.parameters():
            param.requires_grad = False

        for param in kp_detector.parameters():
            param.requires_grad = False

        parameters = []
        
        sdc_net = generator.sdc.train()
        parameters += list(set(p for n, p in sdc_net.named_parameters() if not n.endswith(".quantiles")))            
        aux_parameters = list(set(p for n, p in sdc_net.named_parameters() if n.endswith(".quantiles"))) 

        tdc_net = generator.tdc.train()
        parameters += list([p for n, p in tdc_net.named_parameters() if not n.endswith(".quantiles")])
        aux_parameters += list([p for n, p in tdc_net.named_parameters() if n.endswith(".quantiles")])
        for param in aux_parameters:
            param.requires_grad = True

        aux_optimizer = torch.optim.Adam(aux_parameters, lr=train_params['lr_aux'])

        assert len(parameters) != 0

        for param in parameters:
            param.requires_grad = True       

        gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'],betas=train_params['betas'])
    
    elif step == 2:
        # train the refinement network
        for param in kp_detector.parameters():
            param.requires_grad = False

        for param in generator.parameters():
            param.requires_grad = False

        parameters = list(generator.refinement_network.parameters())
        for param in parameters:
            param.requires_grad = True
            
        gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'],betas=train_params['betas'])
    elif step == 3:
        # OPTIONAL: Finetuning the Generator Networks with frozen residual coders
        for param in kp_detector.parameters():
            param.requires_grad = False

        sdc_net_params = list(generator.sdc.parameters())
        tdc_net_params = list(generator.tdc.parameters())
        for param in sdc_net_params+tdc_net_params:
            param.requires_grad = False
        gen_optimizer = torch.optim.Adam(list(generator.parameters()) , lr=train_params['lr'],betas=train_params['betas'])
    else:
        raise NotImplementedError("Unknown training step [step < 1 or step > 3]")

    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disc_optimizer = torch.optim.AdamW(list(discriminator.parameters()), lr=train_params['lr'],betas=(0.5, 0.999))

    if pretrained_cpk_path != '':
        generator = load_pretrained_model(generator, path=pretrained_cpk_path,name='generator', device=device)
        kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
        gen_optimizer = load_pretrained_model(gen_optimizer, path=pretrained_cpk_path,name='gen_optimzer', device=device)
        if aux_optimizer is not None:
            aux_optimizer = load_pretrained_model(aux_optimizer, path=pretrained_cpk_path,name='aux_optimzer', device=device)

        discriminator = load_pretrained_model(discriminator, path=pretrained_cpk_path,name='discriminator', device=device)
        disc_optimizer = load_pretrained_model(disc_optimizer, path=pretrained_cpk_path,name='disc_optimizer', device=device)
    
    if config['model_params']['generator_params']['ref_coder']:
        #load parameters of pretrained TIC model
        tic_weights = torch.load("checkpoints/tic.pth.tar", map_location=device, weights_only=True)
        generator.ref_coder.load_state_dict(tic_weights['tic'], strict=True)

    generator_full = gen_trainers['rdac'](kp_detector, generator, discriminator,config) 
    disc_type = config['model_params']['discriminator_params']['disc_type']
    discriminator_full = disc_trainers['rdac'](discriminator, train_params, disc_type=disc_type) 


    scheduler_generator = MultiStepLR(gen_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    scheduler_discriminator = MultiStepLR(disc_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
    if aux_optimizer is not None:
        scheduler_aux = MultiStepLR(aux_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        generator_full = generator_full.cuda()
        discriminator_full = discriminator_full.cuda()
        if  torch.cuda.device_count()> 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in range(num_gpus)])
            generator_full = CustomDataParallel(generator_full)
            discriminator_full = CustomDataParallel(discriminator_full)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if torch.cuda.is_available():
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}
                
                params.update({'variable_bitrate':config['model_params']['generator_params']['residual_coder_params']['variable_bitrate']})
                params.update({'bitrate_levels':config['model_params']['generator_params']['residual_coder_params']['levels']})
            
                losses_generator, generated = generator_full(x, **params)
                # print(losses_generator)
                losses_ = {} 

                if 'distortion' in generated:
                    losses_.update({'distortion': generated['distortion'].mean().detach().data.cpu().numpy().item(),
                                    'rate':generated['rate'].mean().detach().data.cpu().numpy().item()})

                if 'perp_distortion' in generated:
                    losses_.update({'perp_distortion':generated['perp_distortion'].mean().detach().data.cpu().numpy().item()})

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values) 
                
                loss.backward()
                
                gen_optimizer.step()
                gen_optimizer.zero_grad()


                aux_loss = 0
                if aux_optimizer is not None:
                    aux_loss = generator.sdc.aux_loss()
                    
                    if generator.tdc is not None:
                        aux_loss += generator.tdc.aux_loss()

                    aux_loss.backward()
                    
                    aux_optimizer.step()
                    aux_optimizer.zero_grad()
                    
                #Train the discriminator network
                losses_discriminator = discriminator_full(x, generated)
                loss_values = [val.mean() for val in losses_discriminator.values()]
                disc_loss = sum(loss_values)
                disc_loss.backward()
                
                disc_optimizer.step()
                disc_optimizer.zero_grad()


                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                if aux_loss > 0:
                    losses.update({"aux_loss":aux_loss.mean().detach().data.cpu().numpy().item()})
                logger.log_iter(losses=losses)
                if kwargs['debug']:
                    break

            scheduler_generator.step()
            scheduler_discriminator.step()
            if aux_optimizer is not None:
                scheduler_aux.step()

            state_dict = {'generator': generator,
                            'kp_detector': kp_detector,
                              'gen_optimizer': gen_optimizer,
                              'discriminator':discriminator, 
                              'disc_optimizer':disc_optimizer}
                        
            if aux_optimizer is not None:
                state_dict.update({'aux_optimizer': aux_optimizer})
            logger.log_epoch(epoch, state_dict, inp=x, out=generated)
            if kwargs['debug']:
                break


def train_hdac(config,dataset,generator, kp_detector,discriminator,**kwargs ):
    train_params = config['train_params'] 
    # create optimizers for generator, kp_detector and discriminator
    parameters = list(generator.parameters()) + list(kp_detector.parameters())
    gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'], betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=train_params['lr'],  betas=(0.5, 0.999))  
      

    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained_cpk_path = config['dataset_params']['cpk_path'] 
    if pretrained_cpk_path != '':
        #Retrain from a saved checkpoint specified in the config file
        generator = load_pretrained_model(generator, path=pretrained_cpk_path,name='generator', device=device)
        kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
        # gen_optimizer = load_pretrained_model(gen_optimizer, path=pretrained_cpk_path,name='gen_optimizer', device=device)

        discriminator = load_pretrained_model(discriminator, path=pretrained_cpk_path,name='discriminator', device=device)
        # disc_optimizer = load_pretrained_model(disc_optimizer, path=pretrained_cpk_path,name='disc_optimizer', device=device)
    
    if config['model_params']['generator_params']['ref_coder']:
        #load parameters of pretrained TIC model
        tic_weights = torch.load("checkpoints/tic.pth.tar", map_location=device, weights_only=True)
        generator.ref_coder.load_state_dict(tic_weights['tic'], strict=True)

    scheduler_generator = MultiStepLR(gen_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    scheduler_discriminator = MultiStepLR(disc_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
    
    generator_full = gen_trainers['hdac'](kp_detector, generator, discriminator,config) 
    disc_type = config['model_params']['discriminator_params']['disc_type']
    discriminator_full = disc_trainers['hdac'](discriminator, train_params, disc_type=disc_type) 

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        generator_full = generator_full.cuda()
        discriminator_full = discriminator_full.cuda()
        if  torch.cuda.device_count()> 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in range(num_gpus)])
            generator_full = CustomDataParallel(generator_full)
            discriminator_full = CustomDataParallel(discriminator_full)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if device =='cuda':
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}
                losses_generator, generated = generator_full(x, **params)
                losses_ = {} 
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values) 
                
                loss.backward(retain_graph=True)
                
                gen_optimizer.step()
                gen_optimizer.zero_grad()

                #forward and backprop on the discriminator
                losses_discriminator = discriminator_full(x, generated)
                loss_values = [val.mean() for val in losses_discriminator.values()]
                disc_loss = sum(loss_values)
                disc_loss.backward(retain_graph=True)
                
                disc_optimizer.step()
                disc_optimizer.zero_grad()

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                logger.log_iter(losses=losses)
                if kwargs['debug']:
                    break


            scheduler_generator.step()
            scheduler_discriminator.step()

            state_dict = {'generator': generator,
                        'kp_detector': kp_detector, 
                        'gen_optimizer': gen_optimizer,
                        'discriminator':discriminator, 
                        'disc_optimizer':disc_optimizer}
            
            logger.log_epoch(epoch, state_dict, inp=x, out=generated)
            if kwargs['debug']:
                break


train_functions = {'dac': train_dac,
                   'hdac': train_hdac,
                   'hdac_hf': train_hdac,
                   'rdac': train_rdac,
                   'mrdac': train_mrdac,
                   }

class CustomDataParallel(torch.nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)