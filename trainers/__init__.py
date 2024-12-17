from .dac_trainer import GeneratorFullModel as DAC_Trainer
from .mr_trainer import GeneratorFullModel as MR_Trainer
from .rdac_trainer import GeneratorFullModel as RDAC_Trainer
from .hdac_trainer import GeneratorFullModel as HDAC_Trainer
from .disc_trainer import *


gen_trainers = {
    'hdac':HDAC_Trainer,
    'rdac': RDAC_Trainer,
    'dac': DAC_Trainer,
    'mrdac': MR_Trainer
    }

disc_trainers = {
    'hdac': DACDiscriminatorFullModel,
    'rdac': DACDiscriminatorFullModel,
    'dac': DACDiscriminatorFullModel,
    'mrdac': MRDiscriminatorFullModel
    }