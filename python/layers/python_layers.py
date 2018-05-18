import caffe
import numpy as np

class Meshgrid(caffe.Layer):
   
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.width = params['width']
        self.height = params['height']
        self.batch = params['batch']

    def reshape(self, bottom, top):
        assert len(bottom) == 0, "No bottom accepted"
        assert len(top) == 1, "Only one top accepted"
        top[0].reshape(self.batch, 2, self.height, self.width)
        # top[1].reshape(self.batch, 1, self.height, self.width)

    def forward(self, bottom, top):
        gx, gy = np.meshgrid(range(self.width), range(self.height))
        gxy = np.concatenate((gx[None,:,:], gy[None,:,:]), axis=0)
        top[0].data[...] = gxy[None, :, :, :]
        # top[1].data[...] = gy[None,None,:,:]

    def backward(self, top, propagate_down, bottom):
       pass

class MeanVals(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.width = params['width']
        self.height = params['height']
        self.batch = params['batch']
    
    def reshape(self, bottom, top):
       assert len(bottom) == 0, "No bottom accepted"
       assert len(top) == 1, "Only one top accepted"
       top[0].reshape(self.batch, 3, self.height, self.width)

    def forward(self, bottom, top):
        m1 = 104 * np.ones((self.height, self.width))
        m2 = 117 * np.ones((self.height, self.width))
        m3 = 123 * np.ones((self.height, self.width))
        mall = np.concatenate((m1[None,:,:], m2[None,:,:], m3[None,:,:]), axis=0)
        top[0].data[...] = mall[None, :, :, :]
        # top[1].data[...] = gy[None,None,:,:]

    def backward(self, top, propagate_down, bottom):
       pass
