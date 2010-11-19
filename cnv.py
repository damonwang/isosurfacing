#!/usr/bin/env python

from __future__ import with_statement
import numpy as np
import Image
import code
import os
import collections
import itertools
from pysvg import builders
from multiprocessing import Pool
import sys

#==============================================================================
# General

class Options(dict):
    '''a thin dict wrapper that aliases getitem to getattribute'''

    def __getattr__(self, name):
        return self.__getitem__(name)

images = Options(cone=Options(fname='data/cone.png', aspect=(1, 1)),
        point=Options(fname='data/point.png', aspect=(1, 1)),
        rings=Options(fname='data/rings.png', aspect=(1, 3)),
        feeth=Options(fname='data/feeth.png', aspect=(1, 1.39)),
        mich=Options(fname='data/mich.png', aspect=(1, 1)),
        teddy=Options(fname='data/teddy.png', aspect=(1, 3.61)),
        noise=Options(fname='data/noise.png', aspect=(1,1)),
        tooth=Options(fname='data/tooth.png', aspect=(1, 1)))

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

#==============================================================================
# Part 1 functions

first_d = np.array([-.5, 0, .5])
second_d = np.array([1., -2., 1.])

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
# Run these functions to solve Part 1

def part1a(filename='data/rings.png', saveas='output/part1a/', aspect=(1,3)):
    '''Writes out three PNGs giving the x-partial, y-partial, and gradient
    magnitude of the original.
    '''

    data = png_to_ndarray(filename)
    if not os.path.exists(saveas): os.mkdir(saveas)

    Image.fromarray(igreyscale(x_partial(data).astype('int32')), mode='I').save(saveas + 'dx.png')
    Image.fromarray(igreyscale(y_partial(data).astype('int32')), mode='I').save(saveas + 'dy.png')
    Image.fromarray(igreyscale(grad_mag(data, aspect).astype('int32')), mode='I').save(saveas + 'gm.png')

def part1b(files=None, saveas='output/part1b/'):
    '''writes out two PNGs per filename: 
        filename-biv.png for the bivariate colormap
        filename-gm.png for the univariate gradient 
    '''

    if not os.path.exists(saveas): os.mkdir(saveas)
    files = files or [ images.cone, images.point, images.feeth ]

    for f in files:
        data = png_to_ndarray(f.fname)
        out_prefix = os.path.join(saveas, os.path.basename(f.fname)[:-4])

        dx, dy = index_to_world(nabla(data), M(*f.aspect)).transpose(2, 0, 1)
        Image.fromarray(lin_bivariate(dx, dy).astype('uint8'), mode='RGB').save(out_prefix + '-biv.png')

        Image.fromarray(lin_univariate(grad_mag(data, f.aspect), np.array([255, 255, 255])).astype('uint8'), mode='RGB').save(out_prefix + '-gm.png')

#==============================================================================
# Functions for Part 2

def grid(data):
    '''grid(ndarray data) -> generator

    This description discusses a 2D array. Higher dimensional arrays are
    handled by considering them as 2D arrays where each element is itself an
    ndarray object.

    iterates over the data one grid square at a time, where a grid square is
    a view with the four points at indices (i, j) (i, j+1) (i+1, j) (i+1, j+1)
    '''

    rows, cols = data.shape[:2]
    for r in xrange(rows - 1):
        for c in xrange(cols - 1):
            yield data[r:r+2,c:c+2]

# these lambdas were the fastest way to recover from discovering that Techstaff
# runs Python 2.5.2 which doesn't support named tuples
Line = lambda *args: tuple(args)
Side = lambda *args: tuple(args)
T = Side(np.array((0,0)), np.array((0,1)))
L = Side(np.array((0,0)), np.array((1,0)))
R = Side(np.array((0,1)), np.array((1,1)))
B = Side(np.array((1,0)), np.array((1,1)))

lookup = { (False, False, False, False) : [],
        (False, False, False, True) : [ Line(R, B) ],
        (False, False, True, False) : [ Line(L, B) ],
        (False, False, True, True) : [ Line(L, R) ],
        (False, True, False, False) : [ Line(T, R) ],
        (False, True, False, True) : [ Line(T, B) ],
        (False, True, True, False) : [ Line(T, L), Line(R, B) ],
        (False, True, True, True) : [ Line(T, L) ],
        (True, False, False, False) : [ Line(T, L) ],
        (True, False, False, True) : [ Line(T, R), Line(L, B) ],
        (True, False, True, False) : [ Line(T, B) ],
        (True, False, True, True) : [ Line(T, R) ],
        (True, True, False, False) : [ Line(L, R) ],
        (True, True, False, True) : [ Line(L, B) ],
        (True, True, True, False) : [ Line(R, B) ],
        (True, True, True, True) : [] }

def cartesian(*args, **kwds):
    '''re-implementation of itertools.product() as described in manual'''

    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool] 
    for prod in result:
        yield tuple(prod)

def draw_lines(data, val, origin=np.array([0,0])):
    '''draw_lines(ndarray data, val) -> [ Line ]

    Runs marching squares and returns a list of the Lines to be drawn
    '''

    lines = []
    r, c = data.shape
    for i, j in cartesian(xrange(r-1), xrange(c-1)):
        top_left = np.array((i, j)) + origin
        for l in lookup[tuple((data[i:i+2,j:j+2] > val).flatten())]:
            points = []
            for s in l:
                # calculate a weight, and then use it to average the given points
                weight = (val - data[i+s[0][0],j+s[0][1]]) / (data[i+s[1][0], j+s[1][1]] - data[i+s[0][0], j+s[0][1]])
                #code.interact(local=locals())
                points.append(top_left + s[0] + weight*(s[1] - s[0]))
                #print Options(top_left=top_left, s=s, weight=weight)
            lines.append(Line(*points))
    return lines

def mp_wrap_draw_lines(args):
    '''unpacks args for draw_lines'''

    (data, origin), values = args
    return itertools.chain(*[draw_lines(data, val, origin=origin) for val in values])

def data_chunks(data, chunksize=100):
    '''breaks ndarray data into many overlapping horizontal stripes.'''

    rows = data.shape[0]
    for i in xrange(0, rows, chunksize):
        yield (data[i:i+chunksize+1], np.array([i, 0]))

MP_PROCMAX = None
MP_THRESH = 1000000

def mp_draw_isocontours(data, isovalues):
    if data.nbytes > MP_THRESH:
        workers = Pool(MP_PROCMAX)
        return itertools.chain(*workers.map(mp_wrap_draw_lines, [ (chunk, isovalues) for chunk in data_chunks(data)]))
    else: return itertools.chain(*[draw_lines(data, val) for val in isovalues ])

def test_lines():
    for i in cartesian([False, True], repeat=4):
        g = np.array(i, dtype='int32').reshape((2,2))
        yield (g, draw_lines(g, .5))

def overlay_isocontours(dataset, isovalues, name, scale=1, drawfn=mp_draw_isocontours):
    aspect = scale * np.array(dataset.aspect)
    data = png_to_ndarray(dataset.fname)
    shb = builders.ShapeBuilder()
    out = builders.svg(name)
    img = builders.image(x=0, y=0, width=data.shape[1], height=data.shape[0])
    img.set_xlink_href(os.path.abspath(dataset.fname))
    grp = builders.g()
    grp.set_transform("scale(%f, %f)" % tuple(aspect))
    grp.addElement(img)
    out.addElement(grp)
    translation = np.array([.5, .5])
    #for l in itertools.chain(overlay_isocontours.workers.map(LineDrawer(data), isovalues)):
    for l in drawfn(data, isovalues):
        l = [ aspect[::-1] * (p + translation) for p in l ]
        out.addElement(shb.createLine(l[1][1], l[1][0], l[0][1], l[0][0], strokewidth=2, stroke="rgb(0,255,0)"))
    return out

def swap_fors((data, values, origin)):
    lines = []
    r,c = data.shape
    for i, j in cartesian(xrange(r-1), xrange(c-1)):
        top_left = np.array((i,j)) + origin
        for val in values:
            for l in lookup[tuple((data[i:i+2,j:j+2] > val).flatten())]:
                points = []
                for s in l:
                    weight = (val - data[i+s[0][0],j+s[0][1]]) / (data[i+s[1][0], j+s[1][1]] - data[i+s[0][0], j+s[0][1]])
                    points.append(top_left + s[0] + weight*(s[1] - s[0]))
                lines.append(Line(*points))
    return lines

def mp_swap_fors(data, isovalues):
    if data.nbytes > MP_THRESH:
        workers = Pool(MP_PROCMAX)
        return itertools.chain(*workers.map(swap_fors, [ (chunk, isovalues, origin) for chunk, origin in data_chunks(data)]))
    else: return itertools.chain(*[draw_lines(data, val) for val in isovalues ])

#==============================================================================
# Run these functions to solve Part 2

def part2a(dataset=images.noise, value=40000., saveas="output/part2a/"):
    if not os.path.exists(saveas): os.mkdir(saveas)
    overlay_isocontours(dataset, [value], 'noise-lines', scale=37.5).save(saveas + 'noise-lines.svg')

def part2b(dataset=images.rings, value=32000, saveas='output/part2b/'):
    if not os.path.exists(saveas): os.mkdir(saveas)
    overlay_isocontours(dataset, [value], 'rings-lines', scale=1.).save(saveas + 'rings-lines.svg')

def part2c(dataset=images.mich, values=np.linspace(0,65535., num=100), saveas="output/part2c/"):
    if not os.path.exists(saveas): os.mkdir(saveas)
    overlay_isocontours(dataset, values, 'mich-lines', scale=1.).save(saveas + 'mich-lines.svg')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        MP_PROCMAX = int(sys.argv[1])

    part2c()
