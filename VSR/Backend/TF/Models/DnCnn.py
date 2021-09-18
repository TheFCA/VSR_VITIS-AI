"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 23rd 2018
Updated Date: May 23rd 2018

Modified by Fernando Carrió
- Compatibility with VITIS-AI: Frozen and Xilinx graphs


Implementing Feed-forward Denoising Convolutional Neural Network
See http://ieeexplore.ieee.org/document/7839189/
"""
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from ..Framework.SuperResolution import SuperResolution
from ..Framework import Noise # fcarrio 25/072021

import logging
LOG = logging.getLogger('VSR.Model.DnCNN_FCA')
from tensorflow.python.platform import gfile #fcarrio


class DnCNN(SuperResolution):
  """Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for
    Image Denoising

  Args:
      layers: number of layers used
  """

  def __init__(self, layers=20, name='dncnn', **kwargs):
    self.name = name
    self.layers = layers
    if 'scale' in kwargs:
      kwargs.pop('scale')
    super(DnCNN, self).__init__(scale=1, **kwargs)

  def conv2d_fca (self, x, filters, kernel_size, name):
    seed = 789
    bi = tf.zeros_initializer()
    kr = tf.keras.regularizers.l2(self.weight_decay)
    ki = tf.keras.initializers.he_normal(seed=seed)
    nn = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1),
                    padding='same', data_format='channels_last',
                    dilation_rate=(1,1), use_bias=False,
                    kernel_initializer=ki, kernel_regularizer=kr,
                    bias_initializer=bi, name=name)
    nn.build(x.shape.as_list())
    x = nn(x)
    return x

  def conv2d (self, x, filters, kernel_size, name, bias=False):
      seed = 789
      weight_decay = 0.0001
      bi = tf.zeros_initializer()
      kr = tf.keras.regularizers.l2(weight_decay)
      ki = tf.keras.initializers.he_normal(seed=seed)

      nn = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1),
                      padding='same', data_format='channels_last',
                      dilation_rate=(1,1), use_bias=bias,
                      kernel_initializer=ki, kernel_regularizer=kr,
                      bias_initializer=bi, name=name)

      nn.build(x.shape.as_list())
      x = nn(x)
      return x
  def build_graph(self):
    super(DnCNN, self).build_graph()
    with tf.variable_scope(self.name):
      x = self.inputs[-1] # / 255.0
      x = self.conv2d(x,64,3,name="block1", bias = True)
      x = tf.nn.relu(x)           
      for i in range(2, self.layers):
        x = self.conv2d(x,64,3,name="block"+str(i), bias = True)
        x = tf.layers.batch_normalization(x, training=self.training_phase, trainable = True, reuse=False)                        
        x = tf.nn.relu(x)           
      x = self.conv2d(x,self.channel,3,name="blockOutput", bias = False)
      outputs = self.inputs[-1] + x
      self.outputs.append(outputs)


  def build_loss_(self):
    with tf.name_scope('loss'):
      mse, loss = super(DnCNN, self).build_loss()
      self.train_metric['loss'] = loss
      self.metrics['mse'] = mse
      self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(
        self.label[-1], self.outputs[-1], max_val=255))
      self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(
        self.label[-1], self.outputs[-1], max_val=255))

  def build_loss(self):
    with tf.name_scope('loss'):
      y_true = self.label[-1] # entire dataset
      y_pred = self.outputs[-1] # entire outputs

      mse = tf.losses.mean_squared_error(y_true, y_pred)
      loss = tf.add_n([mse] + tf.losses.get_regularization_losses())
      optimizer = 'Adam'
      if optimizer == 'Adam':
        LOG.info('Using Adam Optimizer')
        opt = tf.train.AdamOptimizer(self.learning_rate)
      elif optimizer == 'SGD':
        LOG.info('Using Gradient Descent Optimizer')
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # self.grad_clip = 0.01
        if self.grad_clip > 0.0:
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
      self.metrics['regloss'] =  tf.cast(sum(tf.losses.get_regularization_losses()),'float32')
      self.metrics['mse'] = mse
      self.metrics['psnr'] = tf.reduce_mean(
          tf.image.psnr(y_true, y_pred, 255))
      self.metrics['ssim'] = tf.reduce_mean(
          tf.image.ssim(y_true, y_pred, 255))
  
  def compile_frozen_graph(self):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      with gfile.FastGFile(self.pre_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.global_steps = tf.Variable(0, trainable=False, name='global_step')
        self.training_phase = tf.placeholder_with_default(False, shape=None, name='train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.label.append(
        tf.placeholder(tf.float32, shape=[ None, None, None, self.channel], name='label/hr')) #None
        self.inputs.append( #uint8
          tf.placeholder(tf.float32, shape=[ None,None, None, self.channel],name='input/lr'))    
        g_in = tf.import_graph_def(graph_def, input_map={'input/lr': tf.divide(self.inputs[-1], 255.0)})

      Tensor_in =tf.divide(self.inputs[-1], 255.0)
      Tensor_out = tf.get_default_graph().get_tensor_by_name('import/'+self.output_node+':0')
      Result = tf.add(Tensor_in,Tensor_out)
      self.outputs.append(tf.multiply(Result,255.0))
  
  def compile_xilinx_graph(self):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      with gfile.FastGFile(self.pre_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.global_steps = tf.Variable(0, trainable=False, name='global_step')
        self.training_phase = tf.placeholder_with_default(False, shape=None, name='train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.label.append(
        tf.placeholder(tf.float32, shape=[ None, None, self.channel], name='label/hr')) #None
        self.inputs.append( #uint8
          tf.placeholder(tf.float32, shape=[ None, None, self.channel],name='input/lr'))    
        g_in = tf.import_graph_def(graph_def, input_map={'input/lr': tf.divide(self.inputs[-1], 255.0)})

      Tensor_in =tf.divide(self.inputs[-1], 255.0)
      Tensor_out = tf.get_default_graph().get_tensor_by_name('import/'+self.output_node+':0')
      Result = tf.add(Tensor_in,Tensor_out)
      self.outputs.append(tf.multiply(Result,255.0))      