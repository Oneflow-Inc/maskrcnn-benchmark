import os
import numpy
# import torch

class TensorSaver(object):
    '''
    '''

    def __init__(self, base_dir, iteration):
        self.base_dir = base_dir
        self.iteration = iteration

    def step(self, iteration=None):
        if iteration:
            self.iteration = iteration
        else:
            self.iteration += 1

    def save(self, tensor, tensor_name, scope=None, save_grad=False, level=None, im_idx=None):
        save_dir = os.path.join(self.base_dir, 'iter_{}'.format(self.iteration))
        if scope:
            save_dir = os.path.join(save_dir, scope)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        suffix = ''
        if isinstance(im_idx, int):
            suffix = suffix + '.image{}'.format(im_idx)
        if isinstance(level, int):
            suffix = suffix + '.layer{}'.format(level)
        suffix = suffix + '.' + str(tuple(tensor.size()))

        save_path = os.path.join(save_dir, '{}{}'.format(tensor_name, suffix))
        numpy.save(save_path, tensor.cpu().detach().numpy())

        if save_grad:
            grad_save_path = os.path.join(save_dir, '{}_grad{}'.format(tensor_name, suffix))
            tensor.register_hook(lambda grad : numpy.save(grad_save_path, grad.cpu().detach().numpy()))


tensor_saver = None

def create_tensor_saver(base_dir, iteration=0):
    global tensor_saver 
    tensor_saver = TensorSaver(base_dir, iteration)

def get_tensor_saver():
    global tensor_saver
    if not tensor_saver:
        raise Exception("Tensor saver not created yet")

    return tensor_saver
