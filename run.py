import torch
import os
import yaml
from shutil import copy
from argparse import ArgumentParser
from time import gmtime, strftime
import models
from train import train_functions
from test import test_functions
from utilities.utils.dataset import FramesDataset,HDACFramesDataset,MRFramesDataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", 
                        required=True, 
                        help="path to config")
    
    parser.add_argument("--mode", 
                        default="train", 
                        choices=["train", "compress","test"])
    
    parser.add_argument("--project_id", 
                        default='Animation-Based-Codecs', 
                        help="project name")
    
    parser.add_argument("--log_dir", 
                        default='log', 
                        help="path to log into")
    
    parser.add_argument("--checkpoint", 
                        default=None, 
                        help="Use pretrained generator and kp detector")
    
    parser.add_argument("--device_ids", 
                        default="0", 
                        type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    
    parser.add_argument("--verbose", 
                        dest="verbose", 
                        action="store_true", 
                        help="Print model architecture")

    
    parser.add_argument("--debug", 
                        dest="debug", 
                        action="store_true", 
                        help="Test on one batch to debug")
    
    parser.add_argument("--num_workers", 
                            dest="num_workers", 
                            default=4, type=int, 
                            help="num of cpu cores for dataloading")
    
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    num_sources = config['dataset_params']['num_sources']-1
    model_id = os.path.basename(opt.config).split('.')[0]
    if opt.mode == 'train':
        if opt.checkpoint is not None:
            log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
        else:
            log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
            log_dir += '_'+ strftime("%d_%m_%y_%H_%M_%S", gmtime())
    else:
        log_dir = os.path.join(opt.log_dir, model_id)
 
    generator_params = {
        **config['model_params']['common_params'],
        **config['model_params']['generator_params']}

    generator = models.generators[model_id](**generator_params)
    print(f"##..{generator.__class__.__name__} LOADED..##")
    
    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    
    
    kpd_params = {**config['model_params']['common_params'],
                    **config['model_params']['kp_detector_params']}
    kp_detector = models.kp_detectors[model_id](**kpd_params)
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    disc_params = {**config['model_params']['common_params'],
                    **config['model_params']['discriminator_params']}
    discriminator = models.MultiScaleDiscriminator(**disc_params)
    
    if torch.cuda.is_available():
        discriminator = discriminator.to(opt.device_ids[0])

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    if model_id in ['mvac','mrdac']:
        print("Loading Multiple reference Training Dataset..")
        dataset = MRFramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    elif model_id in ['hdac','hdac_hf']:
        print("Loading Hybrid animation training dataset..(Frame samples and base layer)")
        dataset = HDACFramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    else:
        print("Loading  Animation Training Dataset ...")
        dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.mkdir(log_dir+'/img_aug')
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)

        #pass config, generator, kp_detector and discriminator to the training module
        params = {  'project_id': opt.project_id,
                    'debug': opt.debug,
                    'model_id':model_id,
                    'checkpoint': opt.checkpoint, 
                    'log_dir': log_dir, 
                    'device_ids': opt.device_ids,
                    'num_workers': opt.num_workers}

        train_functions[model_id](config,dataset, generator,kp_detector,discriminator, **params)
    
    elif opt.mode == 'test':
        params = {  'model_id':model_id,
                    'checkpoint': opt.checkpoint, 
                    'log_dir': log_dir}
        ## TO-DO 
        ## Implement a unified test interface for the GFVC methods
        test_functions[model_id](config,dataset, generator,kp_detector, **params)


    
