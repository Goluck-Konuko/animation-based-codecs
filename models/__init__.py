from .Trainer import TrainerModel 
from .Discriminator import DiscriminatorModel

## DISC MODEL
from .common.discriminator import MultiScaleDiscriminator

## DAC MODULES
from .dac.generator import DAC_Generator
from .dac.kpd import DAC_KPD

##HDAC MODULES
from .hdac.generator import HDAC_Generator, HDAC_HF_Generator
from .hdac.kpd import HDAC_KPD

## RDAC(+) MODULES
from .rdac.generator import RDAC_Generator
from .rdac.kpd import RDAC_KPD

## Multi-Reference Animation Frameworks
from .mrdac.generator import MRDAC
from .mrdac.kpd import MRDAC_KPD



generators = {'dac': DAC_Generator,
              'hdac': HDAC_Generator,
              'hdac_hf': HDAC_HF_Generator,
              'rdac': RDAC_Generator,
              'rdac_plus': RDAC_Generator,
              'mrdac': MRDAC}

kp_detectors = {'dac': DAC_KPD, 
                'hdac': HDAC_KPD,
                'hdac_hf': HDAC_KPD,
                'rdac': RDAC_KPD,
                'rdac_plus': RDAC_KPD,
                'mrdac': MRDAC_KPD}