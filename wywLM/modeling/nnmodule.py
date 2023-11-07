import pdb
import os
from requests import delete
import torch
import copy
from torch import nn, tensor
from .config import ModelConfig
from ..utils import xtqdm as tqdm
from .cache_utils import load_model_state
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, AutoModel

from ..utils import get_logger

logger = get_logger()

__all__ = ["NNModule"]


class NNModule(nn.Module):
    """ An abstract class to handle weights initialization and \
    a simple interface for dowloading and loading pretrained models.

  Args:
    
    config (:obj:`~DeBERTa.deberta.ModelConfig`): The model config to the module

  """

    def __init__(self, *inputs, **kwargs):
        super().__init__()

    def init_weights(self, module):
        """Apply Gaussian(mean=0, std=`config.initializer_range`) initialization to the module.
        Args:
          module (:obj:`torch.nn.Module`): The module to apply the initialization.
        Example::
          class MyModule(NNModule):
            def __init__(self, config):
              # Add construction instructions
              self.bert = DeBERTa(config)
              # Add other modules
              ...
              # Apply initialization
              self.apply(self.init_weights)
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # module.weight.data.copy_(self.initializer(module.weight.data.shape))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def load_model(
        cls, model_dir_or_name, model_class, config_file=None, *inputs, **kwargs
    ):
        """Instantiate a sub-class of NNModule from a pre-trained model file.
        Args:
          model_dir_or_name (:obj:`str`): Path or name of the pre-trained model which can be either,
            - The path of pre-trained model
            If `model_path` is `None` or `-`, then the method will create a new sub-class without initialing from pre-trained models.
          model_class (obj): model class to be instantiated. Should be derived from transformer.AutoModel
          config_file (:obj:`str`): The path of model config file. If it's `None`, then the method will try to find the the config in order:
            1. ['config'] in the model state dictionary.
            2. `model_config.json` aside the `model_path`.
            If it failed to find a config the method will fail.
        Return:
          :obj:`NNModule` : The sub-class object.
        """
        # Load config
        num_labels = kwargs.get("num_labels", 0)
        if os.path.exists(model_dir_or_name):
            if config_file is not None:
                model_config = AutoConfig.from_pretrained(
                    config_file, num_labels=num_labels
                )
                if model_config["model_type"].lower() == "roberta":
                    model_config["model_type"] = "bert"
            else:
                model_config = None
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir_or_name, config=model_config
            )
            try:
                model = model_class.from_pretrained(
                    model_dir_or_name, config=model_config
                )
            except:
                files = os.listdir(model_dir_or_name)
                pth = [f for f in files if ".bin" in f][0]
                pth = os.path.join(model_dir_or_name, pth)
                config_file = [f for f in files if ".json" in f][0]
                config_file = os.path.join(model_dir_or_name, config_file)
                if model_config is None:
                    model_config = AutoConfig.from_pretrained(
                        config_file, num_labels=num_labels
                    )
                    if model_config["model_type"].lower() == "roberta":
                        model_config["model_type"] = "bert"
                try:
                    model = model_class.from_pretrained(pth, config=model_config)
                except:
                    model_class = AutoModelForMaskedLM
                    return cls.load_model(
                        model_dir_or_name,
                        model_class,
                        config_file=config_file,
                        *inputs,
                        **kwargs
                    )
        else:
            try:
                model_config = AutoConfig.from_pretrained(
                    model_dir_or_name,
                    num_labels=num_labels,
                    finetuning_task="mc",
                    revision="main",
                )

                tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
                model = model_class.from_pretrained(
                    model_dir_or_name, config=model_config, revision="main"
                )

            except:
                model_class = AutoModelForMaskedLM
                return cls.load_model(
                    model_dir_or_name,
                    model_class,
                    config_file=config_file,
                    *inputs,
                    **kwargs
                )
        return model, tokenizer, model_config
