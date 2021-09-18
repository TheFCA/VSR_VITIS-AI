#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7
#  Fcarrio added random state for reproducibility
#  Fcarrio rejects samples with no info
import numpy as np
from ..Backend import DATA_FORMAT
import matplotlib.pyplot as plt #fcarrio

class Cropper(object):
  def __init__(self, scale, rs):
    self.scale = scale
    self.rs = rs # fcarrio random state
  def __call__(self, img_pair: tuple, shape: list):
    #print ("img_pair[0]",img_pair[0].shape)
    #print ("img_pair[1]",img_pair[1].shape)

    assert len(img_pair) >= 2, \
      f"Pair must contain more than 2 elements, which is {img_pair}"
    for img in img_pair:
      assert img.ndim == len(shape), \
        f"Shape mis-match: {img.ndim} != {len(shape)}"

    return self.call(img_pair, shape)

  def call(self, img: tuple, shape: (list, tuple)) -> tuple:
    raise NotImplementedError


class RandomCrop(Cropper):
  def call(self, img: tuple, shape: (list, tuple)) -> tuple:
    hr, lr = img
    if lr.shape[-2] < shape[-2]:
      raise ValueError("Batch shape is too large than data")
    valid=False
    while(not(valid)): 
      ind = [self.rs.randint(nd + 1) for nd in lr.shape - np.array(shape)]
      slc2 = slc1.copy()
      if DATA_FORMAT == 'channels_last':
        slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
        slc2[-3] = slice(ind[-3] * self.scale,
                       (ind[-3] + shape[-3]) * self.scale)
      else:
        slc2[-1] = slice(ind[-1] * self.scale,
                       (ind[-1] + shape[-1]) * self.scale)
        slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
      hr_sliced = hr[tuple(slc2)]
      lr_sliced = lr[tuple(slc2)]


      val =  (np.sum(hr_sliced)) #fcarrio to avoid using all black data
      if (val == 0)| (val > 255*shape[-2]*shape[-2]*0.99) | (val < 255*shape[-2]*shape[-2]*0.01):

        pass
      else:
        break
      
    return hr[tuple(slc2)], lr[tuple(slc1)]


class CenterCrop(Cropper):
  def call(self, img: tuple, shape: (list, tuple)) -> tuple:
    hr, lr = img
    ind = [nd // 2 for nd in hr.shape - np.array(shape)]
    slc1 = [slice(n, n + s) for n, s in zip(ind, shape)]
    slc2 = slc1.copy()
    if DATA_FORMAT == 'channels_last':
      slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
      slc2[-3] = slice(ind[-3] * self.scale,
                       (ind[-3] + shape[-3]) * self.scale)
    else:
      slc2[-1] = slice(ind[-1] * self.scale,
                       (ind[-1] + shape[-1]) * self.scale)
      slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
    return hr[tuple(slc2)], lr[tuple(slc1)]
