"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 17th 2018

Modified by: Fernando Carrió Argos
Corrected image concatenation position.


Architectures of common dense blocks used in SR researches
"""

import tensorflow as tf

from ..Framework.LayersHelper import Layers


def dense_block(layers: Layers, inputs, depth=8, rate=16, out_dims=128,
                scope=None, reuse=None):
  filters = out_dims - rate * depth
  feat = [inputs]
  with tf.variable_scope(scope, 'DenseBlock', reuse=reuse):
    for _ in range(depth):
      filters += rate
      print('FILTERS', filters)
      x = layers.relu_conv2d(feat[-1], filters, 3)
      feat.append(x)
      if len(feat)>2:  #fcarrio, wrong feature length
       feat[-1] = tf.concat(feat[-2:], axis=-1)       
      # feat[-1] = tf.concat(feat[1:], axis=-1) #old
    return x
