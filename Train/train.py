#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7
#  Modified by FCarrio to add multiscale support, 
import argparse
from pathlib import Path

from VSR.Backend import BACKEND
from VSR.DataLoader import CenterCrop, Loader, RandomCrop
from VSR.DataLoader import load_datasets
from VSR.Model import get_model, list_supported_models
from VSR.Util import Config, lr_decay, suppress_opt_by_args, compat_param
import numpy as np
import os #fcarrio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#fcarrio, that removes lots of not useful logs being printed

parser = argparse.ArgumentParser(description=f'VSR ({BACKEND}) Training Tool v1.0')
g0 = parser.add_argument_group("basic options")
g0.add_argument("model", choices=list_supported_models(), help="specify the model name")
g0.add_argument("-p", "--parameter", help="specify the model parameter file (*.yaml)")
g0.add_argument("--save_dir", default='../Results', help="working directory")
g0.add_argument("--data_config", default="../Data/datasets.yaml", help="specify dataset config file")
g1 = parser.add_argument_group("training options")
g1.add_argument("--dataset", default='none', help="specify a dataset alias for training")
g1.add_argument("--ensemble", action="store_true", help="specify if the dataset requires ensemble")
g1.add_argument("--epochs", type=int, default=1, help="specify total epochs to train")
g1.add_argument("--steps", type=int, default=200, help="specify steps of iteration in every training epoch")
g1.add_argument("--val_steps", type=int, default=50, help="steps of iteration of validations during training")
g2 = parser.add_argument_group("device options")
g2.add_argument("--cuda", action="store_true", help="using cuda gpu")
g2.add_argument("--threads", type=int, default=8, help="specify loading threads number")
g2.add_argument('--memory_limit', default=None, help="limit the CPU memory usage. i.e. '4GB', '1024MB'")
g3 = parser.add_argument_group("advanced options")
g3.add_argument("--traced_val", action="store_true")
g3.add_argument("--pretrain", help="specify the pre-trained model checkpoint or will search into `save_dir` if not specified")
g3.add_argument("--export", help="export ONNX (torch backend) or protobuf (tf backend) (needs support from model)")
g3.add_argument("-c", "--comment", default=None, help="extend a comment string after saving folder")

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  print("Name:", gpu.name, "  Type:", gpu.device_type)

def main():
  flags, args = parser.parse_known_args()
  opt = Config()  # An EasyDict object
  # overwrite flag values into opt object
  for pair in flags._get_kwargs():
    opt.setdefault(*pair)
  # fetch dataset descriptions
  data_config_file = Path(opt.data_config)
  if not data_config_file.exists():
    raise FileNotFoundError("dataset config file doesn't exist!")
  for _ext in ('json', 'yaml', 'yml'):  # for compat
    if opt.parameter:
      model_config_file = Path(opt.parameter)
    else:
      model_config_file = Path(f'par/{BACKEND}/{opt.model}.{_ext}')
    if model_config_file.exists():
      opt.update(compat_param(Config(str(model_config_file))))
  print("opt.multiscale", opt.multiscale)
  print ("parameter file is: ", model_config_file)
  # get model parameters from pre-defined YAML file
  model_params = opt.get(opt.model, {})
  print ("**model_params ", *model_params)
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  # construct model
  model = get_model(opt.model)(**model_params) 
  if opt.ensemble: 
    model.ensemble 
  if opt.cuda:
    model.cuda() 
  if opt.pretrain:
    model.load(opt.pretrain)
  root = f'{opt.save_dir}/{opt.model}'
  if opt.comment:
    root += '_' + opt.comment

  dataset = load_datasets(data_config_file, opt.dataset)
  print (data_config_file)

  print (opt.dataset)
  # construct data loader for training
  rs = np.random.RandomState(1234)
  lt = Loader(dataset.train.hr, dataset.train.lr, opt.scale, multiscale=opt.multiscale, threads=opt.threads,rs=rs, upscale_model=opt.upscale_model) # added multiscale
  lt.cropper(RandomCrop(1,rs)) #
  lt.image_augmentation()
  # construct data loader for validating
  lv = None
  if dataset.val is not None:
    lv = Loader(dataset.val.hr, dataset.val.lr, opt.scale, multiscale=opt.multiscale, threads=opt.threads,rs=rs,upscale_model=opt.upscale_model) # added multiscale
    lv.cropper(RandomCrop(1,rs))

  if opt.channel == 1:
    # convert data color space to grayscale
    lt.set_color_space('hr', 'L')
    lt.set_color_space('lr', 'L')
    if lv is not None:
      lv.set_color_space('hr', 'L')
      lv.set_color_space('lr', 'L')
  # enter model executor environment
  with model.get_executor(root) as t:
    t.set_seed(1234)
    config = t.query_config(opt)
    if opt.lr_decay:
      config.lr_schedule = lr_decay(lr=opt.lr, epoch=None, max_step=None, **opt.lr_decay)
    t.fit([lt, lv], config)
    if opt.export:
      t.export(opt.export)

if __name__ == '__main__':
  main()
