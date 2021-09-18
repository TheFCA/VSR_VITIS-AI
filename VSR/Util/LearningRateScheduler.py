#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7
#
#  Added Cyclical Learning Rate
#  Author: Fernando Carrió Argos
#  Email: fernando.carrio@uma.es
#  Update: 2020 - 6 - 25
#  
from functools import partial

import math #fcarrio

def _exponential_decay(start_lr, steps, decay_step, decay_rate, **kwargs):
  return_lr = (start_lr * decay_rate ** (steps / decay_step[0])) #fcarrio, otherwise returns a numpy array
  return return_lr

def _poly_decay(start_lr, end_lr, steps, decay_step, power, **kwargs):
  return (start_lr - end_lr) * (1 - steps / decay_step) ** power + end_lr

def _linear_decay(start_lr, decay_step, decay_rate, epoch, **kwargs):
  return start_lr * decay_rate ** (epoch // decay_step)

def _stair_decay(start_lr, steps, decay_step, decay_rate, **kwargs):
  return start_lr * decay_rate ** (steps // decay_step)

def _multistep_decay(start_lr, steps, decay_step, decay_rate, **kwargs):
  if not decay_step:
    return start_lr
  for n, s in enumerate(decay_step):
    if steps <= s:
      return start_lr * (decay_rate ** n)
  if steps > decay_step[-1]:
    return start_lr * (decay_rate ** len(decay_step))

def _cyclic_decay(start_lr, steps, epoch, max_step,**kwargs): #fcarrio 
  cstep = max_step*(epoch-1) + steps%max_step
#  print (cstep)
#  print (epoch)
#  print (steps%max_step)
  step_size = kwargs.get('step_size') or 1000
  mode = kwargs.get('mode') or 'triangular'
  max_lr = kwargs.get('max_lr') or 0.01
  lr = start_lr

  cycle = math.floor(1. + cstep / (2. * step_size))
  x = math.fabs(cstep / step_size - 2. * cycle + 1.)
  clr = (max_lr - lr) * max(0., 1. - x)
  if mode == 'triangular2':
    clr = clr/(math.pow(2, (cycle - 1)))
  if mode == 'exp_range':
    clr = clr * math.pow(.99994, cstep) #fcarrio, 0.99994 too bigger?
  return clr + lr

def lr_decay(method, lr, epoch, max_step, **kwargs):
  if method == 'exp':
    return partial(_exponential_decay, start_lr=lr, **kwargs)
  elif method == 'poly':
    return partial(_poly_decay, start_lr=lr, **kwargs)
  elif method == 'stair':
    return partial(_stair_decay, start_lr=lr, **kwargs)
  elif method == 'linear':
    return partial(_linear_decay, start_lr=lr, epoch=epoch, **kwargs)
  elif method == 'multistep':
    return partial(_multistep_decay, start_lr=lr, **kwargs)
  elif method == 'cyclic': #fcarrio
    return partial(_cyclic_decay, start_lr=lr, epoch=epoch, max_step=max_step, **kwargs)
  else:
    print('invalid decay method!')
    return None
