#!/usr/bin/env python

from __future__ import with_statement
import numpy as np
import Image
import code
import os

first_d = np.array([-.5, 0, .5])
second_d = np.array([1., -2., 1.])

images = Options(cone=Options(fname='data/cone.png', aspect=(1, 1)),
        point=Options(fname='data/point.png', aspect=(1, 1)),
        rings=Options(fname='data/rings.png', aspect=(1, 3)),
        feeth=Options(fname='data/feeth.png', aspect=(1, 1.39)),
        mich=Options(fname='data/mich.png', aspect=(1, 1)),
        teddy=Options(fname='data/teddy.png', aspect=(1, 3.61)),
        tooth=Options(fname='data/tooth.png', aspect=(1, 1)))

class Options(dict):
    '''a thin dict wrapper that aliases getitem to getattribute'''

    def __getattr__(self, name):
        return self.__getitem__(name)

def embed(data, n):
    '''embed(ndarray data, int n) -> ndarray result

    Add n columns of border to either side of the data, filling the border with
    the closest data element.

    This is why support must be small.
    '''

    dr, dc = data.shape
    universe = np.zeros((dr, dc+n))
    universe[:, n/2:-n/2] = data
    universe[:,:n/2] = data[:,:1]
    universe[:,-n/2:] = data[:,-1:]

    return universe

def convolve_f1d(data, kernel):
    '''convolve_f1d(ndarray data, ndarray kernel) -> ndarray convolution

    Returns a 1D convolution along the faster axis of a 2D array, assuming the
    kernel is of finite (and very small) support. Where the kernel requires
    data outside the sampled region, the closest is used. 

    To convolve along slower axis, transpose the data first.
    '''

    support = kernel.shape[0]
    universe = embed(data, support)
    return sum([ kernel[i] * universe[:,i:-support+i] for i in xrange(support) ])

# wrote it, then realized it was less convenient after all!
def convolve_png(filename, kernel, **kwargs):
    '''convolve_png(str filename, ndarray kernel, **kwargs) -> int32 ndarray

    A convenience function: reads data from the PNG, convolves, colors, and
    then returns the PIL Image object.

    Options: 

    prefilter: a function ndarray -> ndarray to apply before the convolution
    postfilter: for after the convolution

    colormap: a function ndarray -> (ndarray data, str mode) which does its own
    normalizing and also returns the necessary mode flag to construct an Image.

    saveas: a filename to write the image into, in addition to returning the
    ndarray

    max: an int, normalize to this maximum
    '''

    options = Options(colormap = lambda data: (igreyscale(data), 'I'),
            prefilter = lambda data: data, 
            postfilter = lambda data: data,
            saveas = None)
    options.update(kwargs)

    conv = options.postfilter(convolve_f1d(options.prefilter(png_to_ndarray(filename)), kernel))

    colored, mode =  options.colormap(conv.astype('int32'))
    if options.saveas is not None: 
        Image.fromarray(colored, mode=mode).save(options.saveas)
    return conv

def png_to_ndarray(filename): # pragma: no cover
    '''png_to_ndarray(str filename) -> ndarray 
    '''

    im = Image.open(filename)
    return np.array(im.getdata()).reshape(im.size[::-1])

def index_to_world(data, M):
    '''index_to_world(ndarray data, matrix M) -> ndarray
    '''

    M = M.I.T
    M = np.array([M[0,0], M[1,1]])

    return np.array([ [ M * col for col in row ] for row in data ])

def igreyscale(data, max=65535):
    '''igreyscale(ndarray data) -> ndarray

    Normalizes and then maps 0 to 0 and 1 to max, linearly. 

    The 'i' prefix means this function makes its changes in place!
    '''

    data -= data.min()
    data *= float(max)/data.max()
    return data

def lin_univariate(data, map):
    '''lin_univariate(ndarray data, ndarray map) -> ndarray float64 result

    map represents the color for the max data value; black is assumed for min

    if data.shape = (r, c) and map.shape = (3,), then result.shape = (r, c, 3)
    '''

    if data.dtype != np.float: data = data.astype('float64')
    data = igreyscale(data, max=1)
    return np.array([ m * data for m in map]).transpose(1, 2, 0)

def lin_bivariate(A, B, max=255):
    '''lin_bivariate(ndarray A, ndarray B) -> ndarray float64

    Normalizes and then applies the following bivariate map:
        green varies with A from 0=0 to 1=max, linearly
        red and blue vary with B from 0=0 to 1=max, linearly

    '''

    A = lin_univariate(A, np.array([0, 255, 0]))
    B = lin_univariate(B, np.array([255, 0, 255]))

    return A+B

def M(slow, fast):
    '''M(slow, fast) -> matrix

    Given an aspect ratio, returns the projection matrix to 1:1 ratio
    '''

    return np.matrix([[slow, 0], [0, fast]])

def x_partial(data):
    '''x_partial(ndarray data) -> ndarray'''

    return convolve_f1d(data, first_d)

def y_partial(data):
    '''y_partial(ndarray data) -> ndarray'''
    return convolve_f1d(data.T, first_d).T

def nabla(data):
    '''nabla(ndarray data) -> ndarray

    Returns the gradient vector at each data point.

    Nabla is an obscure name for the del operator. Del is a Python keyword.
    '''

    return np.dstack((x_partial(data), y_partial(data)))

def grad_mag(data, aspect):
    '''grad_mag(ndarray data, (slow, fast) ) -> ndarray

    the (slow, fast) tuple is an aspect ratio.  
    '''

    grad = np.vectorize(lambda x, y: np.sqrt(x**2+y**2), otypes=['float64'])
    world_dx, world_dy = index_to_world(nabla(data), M(*aspect)).transpose(2, 0, 1) 
    return grad(world_dx, world_dy)

#==============================================================================
# Project-Specific Functions

def part1a(filename='data/rings.png', saveas='output/part1a/', aspect=(1,3)):
    '''Writes out three PNGs giving the x-partial, y-partial, and gradient
    magnitude of the original.
    '''

    data = png_to_ndarray(filename)

    Image.fromarray(igreyscale(x_partial(data).astype('int32')), mode='I').save(saveas + 'dx.png')
    Image.fromarray(igreyscale(y_partial(data).astype('int32')), mode='I').save(saveas + 'dy.png')
    Image.fromarray(igreyscale(grad_mag(data, aspect).astype('int32')), mode='I').save(saveas + 'gm.png')

def part1b(files=None, saveas='output/part1b/'):
    '''writes out two PNGs per filename: 
        filename-biv.png for the bivariate colormap
        filename-gm.png for the univariate gradient 
    '''

    files = files or [ images.cone, images.point, images.feeth ]

    for f in files:
        data = png_to_ndarray(f.fname)
        out_prefix = os.path.join(saveas, os.path.basename(f.fname)[:-4])

        dx, dy = index_to_world(nabla(data), M(*f.aspect)).transpose(2, 0, 1)
        Image.fromarray(lin_bivariate(dx, dy).astype('uint8'), mode='RGB').save(out_prefix + '-biv.png')

        Image.fromarray(lin_univariate(grad_mag(data, f.aspect), np.array([255, 255, 255])).astype('uint8'), mode='RGB').save(out_prefix + '-gm.png')

