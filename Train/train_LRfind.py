#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7
#  Modified by FCarrio to add Cyclical Learning Rate support 

import argparse
from pathlib import Path

from VSR.Backend import BACKEND
from VSR.DataLoader import CenterCrop, Loader, RandomCrop
from VSR.DataLoader import load_datasets
from VSR.Model import get_model, list_supported_models
from VSR.Util import Config, lr_decay, suppress_opt_by_args, compat_param

import os #fcarrio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#fcarrio, that removes lots of not useful logs being printed
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description=f'VSR ({BACKEND}) Training Tool v1.0')
g0 = parser.add_argument_group("basic options")
g0.add_argument("model", choices=list_supported_models(), help="specify the model name")
g0.add_argument("-p", "--parameter", help="specify the model parameter file (*.yaml)")
g0.add_argument("--save_dir", default='../Results/LrFinder', help="working directory")
g0.add_argument("--data_config", default="../Data/datasets.yaml", help="specify dataset config file")
g1 = parser.add_argument_group("training options")
g1.add_argument("--dataset", default='none', help="specify a dataset alias for training")
g1.add_argument("--ensemble", action="store_true", help="specify if the dataset requires ensemble")
g1.add_argument("--epochs", type=int, default=1, help="specify total epochs to train")
g1.add_argument("--steps", type=int, default=200, help="specify steps of iteration in every training epoch")
g1.add_argument("--val_steps", type=int, default=10, help="steps of iteration of validations during training")
g2 = parser.add_argument_group("device options")
g2.add_argument("--cuda", action="store_true", help="using cuda gpu")
g2.add_argument("--threads", type=int, default=8, help="specify loading threads number")
g2.add_argument('--memory_limit', default=None, help="limit the CPU memory usage. i.e. '4GB', '1024MB'")
g3 = parser.add_argument_group("advanced options")
g3.add_argument("--traced_val", action="store_true")
g3.add_argument("--pretrain", help="specify the pre-trained model checkpoint or will search into `save_dir` if not specified")
g3.add_argument("--export", help="export ONNX (torch backend) or protobuf (tf backend) (needs support from model)")
g3.add_argument("-c", "--comment", default=None, help="extend a comment string after saving folder")


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

  # get model parameters from pre-defined YAML file
  model_params = opt.get(opt.model, {})
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  root = f'{opt.save_dir}/{opt.model}'
  if opt.comment:
    root += '_' + opt.comment

  dataset = load_datasets(data_config_file, opt.dataset)
  # construct data loader for training

  lt = Loader(dataset.train.hr, dataset.train.lr, opt.scale, threads=opt.threads)
  lt.image_augmentation()
  lt.cropper(RandomCrop(opt.scale))
  # construct data loader for validating
  lv = None
  if dataset.val is not None:
    lv = Loader(dataset.val.hr, dataset.val.lr, opt.scale, threads=opt.threads)
  if opt.traced_val and lv is not None:
    lv.cropper(CenterCrop(opt.scale))
  elif lv is not None:
    lv.cropper(RandomCrop(opt.scale))
  if opt.channel == 1:
    # convert data color space to grayscale, L is Luminance
    lt.set_color_space('hr', 'L')
    lt.set_color_space('lr', 'L')
    if lv is not None:
      lv.set_color_space('hr', 'L')
      lv.set_color_space('lr', 'L')
  # enter model executor environment
##########################################
  lr = np.logspace(np.log10(0.000001),np.log10(0.1),20)
  l2val = np.logspace(np.log10(0.0001),np.log10(0.1),100)
  loss = []
  for i in range(len(l2val)):
    # construct model
    model_params.update({'l2val':l2val[i]})
    model = get_model(opt.model)(**model_params)
    print (model_params)
    print ("model type is", type(model))
    opt.update({'epochs':6})
    opt.update({'val_steps':10})
    opt.update({'lr':1e-4})
    opt.update({'Finder':True})

    with model.get_executor(root) as t:
      config = t.query_config(opt)
      print ("t type is", type(t))
      if opt.lr_decay:
        config.lr_schedule = lr_decay(lr=opt.lr, epoch=None, **opt.lr_decay) #fcarrio added epoch
      t.fit([lt, lv], config) # training and validation
      loss.append(t.returnloss())
      if opt.export:
        t.export(opt.export)
  loss_np = np.array(loss)
  plot_loss(loss_np, l2val)

def plot_loss(loss_np, lr_np):
  """
  Plots the loss.
  Parameters:
      n_skip_beginning - number of batches to skip on the left.
      n_skip_end - number of batches to skip on the right.
  """
  np.savetxt("data_l2.csv", (loss_np,lr_np), delimiter=",")
  plt.ylabel("loss")
  plt.xlabel("learning rate (log scale)")
  plt.plot(lr_np, loss_np)
  plt.xscale('log')
  plt.yscale('log')
  plt.show()

if __name__ == '__main__':
  main()
