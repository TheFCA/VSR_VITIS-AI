"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 8th 2018
Updated Date: June 8th 2018

Image Super-Resolution via Deep Recursive Residual Network (CVPR 2017)
See http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf
"""

import logging

#import tensorflow as tf
import tensorflow.compat.v1 as tf
from ..Framework.SuperResolution import SuperResolution
from ..Util import bicubic_rescale

LOG = logging.getLogger('VSR.Model.DRRN')

class DRRN(SuperResolution):
  """Image Super-Resolution via Deep Recursive Residual Network

  Args:
      residual_unit: number of residual blocks in one recursion
      recursive_block: number of recursions
      grad_clip: gradient clip ratio according to the paper
      custom_upsample: use --add_custom_callbacks=upsample during fitting, or
        use `bicubic_rescale`. TODO: REMOVE IN FUTURE.
  """

  def __init__(self, residual_unit=3, recursive_block=3,
               custom_upsample=False,
               grad_clip=0.01, name='drrn', **kwargs):
    self.ru = residual_unit
    self.rb = recursive_block
    self.grad_clip = grad_clip
    self.do_up = not custom_upsample
    self.name = name
    super(DRRN, self).__init__(**kwargs)

  def display(self):
    super(DRRN, self).display()
    self.calc_param()
    LOG.info('Recursive Blocks: %d' % self.rb)
    LOG.info('Residual Units: %d' % self.ru)
    LOG.info('Total Number of Parameters is: %d' % self.npar) # fcarrio

  def calc_param(self): #fcarrio
    total_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value

      total_parameters += variable_parameters
    self.npar = total_parameters


  def _shared_resblock(self, inputs, name, **kwargs):
    """    kwargs.update({
      'padding': 'same',
      'data_format': 'channels_last',
      'activation': None,
      'use_bias': False,
      'use_batchnorm': False,
      'use_sn': False,
      'kernel_initializer': 'he_normal',
      'kernel_regularizer': 'l2'
    })"""
    x = tf.nn.relu(inputs, name = 'ReluBlock') #
    x = self.conv2d(x, 128, 3, name='ConvBlock',**kwargs) #
    ori = x

    for _ in range(self.ru):

      with tf.variable_scope(('ResUnit_'+name), reuse=tf.AUTO_REUSE):
        x = tf.nn.relu(x, name = 'Relu_Unit1')
        x = self.conv2d(x, filters=128, kernel_size=3, name = 'Conv_Unit1', **kwargs)
        x = tf.nn.relu(x, name = 'Relu_Unit2')
        x = self.conv2d(x, filters=128, kernel_size=3, strides=(1,1),name = 'Conv_Unit2', **kwargs)
        x +=ori
    return x

  def build_graph(self,**kwargs):
    super(DRRN, self).build_graph()
    l2val = self.unknown_args.get("l2val")
    if l2val == None:
      l2val = 0.01
    kwargs.update({
      'padding': 'same',
      'data_format': 'channels_last',
      'activation': None,
      'use_bias': False,
      'use_batchnorm': False,
      'use_sn': False,
      'kernel_initializer': 'he_normal',
      'kernel_regularizer': 'l2'})
    with tf.variable_scope(self.name):
      x = self.inputs[-1]
      bic = x

      for _ in range(self.rb):
        x = self._shared_resblock(x, name = str(_),**kwargs)
      x = tf.nn.relu(x, name = 'Relu_Last')
      x = self.conv2d(x, self.channel, 3, name='Conv_Last', **kwargs)

    self.outputs.append(tf.add(x,bic))
  def build_loss(self):

    with tf.name_scope('loss'):
      y_true = self.label[-1] # entire dataset
      y_pred = self.outputs[-1] # entire outputs
      print ("shape self.label[-1]", y_true.shape)
      print ("shape self.outptus[-1]", y_pred.shape)
      print ("shape self.inputs[-1]", self.inputs[-1].shape)

      mse = tf.losses.mean_squared_error(y_true, y_pred)
      loss = tf.add_n([mse] + tf.losses.get_regularization_losses())

      optimizer = 'Adam'
      if optimizer == 'Adam':
        LOG.info('Using Adam Optimizer')
        opt = tf.train.AdamOptimizer(self.learning_rate)
      elif optimizer == 'SGD':
        LOG.info('Using Gradient Descent Optimizer')
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #fcarrio
    
      with tf.control_dependencies(update_ops): # execute sequentially
        if self.grad_clip > 0:
          grads_and_vars = []
          for grad, var in opt.compute_gradients(loss):
            grads_and_vars.append((
              tf.clip_by_value(
                  grad,
                  -self.grad_clip / self.learning_rate,
                  self.grad_clip / self.learning_rate),
              var))
          op = opt.apply_gradients(grads_and_vars, self.global_steps)
        else:
          op = opt.minimize(loss, self.global_steps)

      self.loss.append(op)

      self.train_metric['loss'] = loss
      self.metrics['regloss'] =  tf.cast(sum(tf.losses.get_regularization_losses()),'float32') # fcarrio
      self.metrics['mse'] = mse
      self.metrics['psnr'] = tf.reduce_mean(
          tf.image.psnr(y_true, y_pred, 255))
      self.metrics['ssim'] = tf.reduce_mean(
          tf.image.ssim(y_true, y_pred, 255))

  def build_saver(self):
    self.savers[self.name] = tf.train.Saver()
