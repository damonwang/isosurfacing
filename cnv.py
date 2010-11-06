#!/usr/bin/env python

from __future__ import with_statement
import numpy as np
import Image
import code
import os
import collections
from itertools import repeat
from pysvg import builders
import ImageColor
import cm

#==============================================================================
# General

class Options(dict):
    '''a thin dict wrapper that aliases getitem to getattribute'''

    def __getattr__(self, name):
        return self.__getitem__(name)

images = Options(cone=Options(fname='data/cone.png', aspect=(1, 1)),
        point=Options(fname='data/point.png', aspect=(1, 1)),
        rings=Options(fname='data/rings.png', aspect=(1, 3)),
        noise=Options(fname='data/noise.png', aspect=(1,1)),
        feeth=Options(fname='data/feeth.png', aspect=(1, 1.39)),
        feeth_sml=Options(fname='data/feeth-sml.png', aspect=(1, 1.39)),
        mich=Options(fname='data/mich.png', aspect=(1, 1)),
        mich_sml=Options(fname='data/mich-sml.png', aspect=(1, 1)),
        teddy=Options(fname='data/teddy.png', aspect=(1, 3.61)),
        teddy_sml=Options(fname='data/teddy-sml.png', aspect=(1, 3.61)),
        tooth=Options(fname='data/tooth.png', aspect=(1, 1)),
        tooth_sml=Options(fname='data/tooth-sml.png', aspect=(1, 1)))

def png_to_ndarray(filename): # pragma: no cover
    '''png_to_ndarray(str filename) -> ndarray 
    '''

    im = Image.open(filename)
    try: 
        return np.array(im.getdata()).reshape(im.size[::-1])
    except ValueError: # rgb data?
        return np.array(im.getdata()).reshape(im.size[::-1] + (3,))

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
    write_svg(Options(fname=saveas + 'dx.png', aspect=images.rings.aspect), 'rings-dx').save(saveas + 'dx.svg')
    Image.fromarray(igreyscale(y_partial(data).astype('int32')), mode='I').save(saveas + 'dy.png')
    write_svg(Options(fname=saveas + 'dy.png', aspect=images.rings.aspect), 'rings-dy').save(saveas + 'dy.svg')
    Image.fromarray(igreyscale(grad_mag(data, aspect).astype('int32')), mode='I').save(saveas + 'gm.png')
    write_svg(Options(fname=saveas + 'gm.png', aspect=images.rings.aspect), 'rings-gm').save(saveas + 'gm.svg')

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
        write_svg(Options(fname=out_prefix + '-biv.png',
            aspect=f.aspect), os.path.basename(out_prefix) +
            '-biv').save(out_prefix + '-biv.svg')

        Image.fromarray(lin_univariate(grad_mag(data, f.aspect), np.array([255, 255, 255])).astype('uint8'), mode='RGB').save(out_prefix + '-gm.png')
        write_svg(Options(fname=out_prefix + '-gm.png',
            aspect=f.aspect), os.path.basename(out_prefix) +
            '-gm').save(out_prefix + '-gm.svg')

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


def draw_lines(data, val):
    '''draw_lines(ndarray data, val) -> [ Line ]

    Runs marching squares and returns a list of the Lines to be drawn
    '''

    lines = []
    r, c = data.shape
    for i, j in cartesian(xrange(r-1), xrange(c-1)):
        top_left = np.array((i, j))
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

def test_lines():
    for i in cartesian([False, True], repeat=4):
        g = np.array(i, dtype='int32').reshape((2,2))
        yield (g, draw_lines(g, .5))

def write_svg(dataset, name, scale=1, img=True):
    '''write_svg(Options dataset, str name, num scale) -> SVG'''

    if 'size' in dataset:
        shape = dataset.size
    else: shape = png_to_ndarray(dataset.fname).shape

    shb = builders.ShapeBuilder()
    out = builders.svg(name)
    img = builders.image(x=0, y=0, width=shape[1], height=shape[0])
    img.set_xlink_href(os.path.abspath(dataset.fname))
    grp = builders.g()
    grp.set_transform("scale(%f, %f)" % tuple(scale * np.array(dataset.aspect)))
    grp.addElement(img)
    out.addElement(grp)
    return out

def overlay_isocontours(dataset, isovalues, svg, **kwargs):
    '''overlay_isocontours(Options dataset, list isovalues, SVG svg, **kwargs) -> SVG

    keywords:
    scale (default 1)
    colors (default green): a string of the form "rgb(red, green, blue)"
    widths (default 1)

    isovalues, colors, and widths get zipped together, so to specify color of
    ith isocontour, put that color in the ith element of the colors list.
    '''
    options = Options(scale=1, colors=repeat("rgb(0,255,0)"), 
            widths=repeat(1))
    options.update(kwargs)

    aspect = options.scale * np.array(dataset.aspect)
    shift = np.array([0.5, 0.5])
    data = png_to_ndarray(dataset.fname)
    shb = builders.ShapeBuilder()
    for val, color, width in zip(isovalues, options.colors, options.widths):
        for l in draw_lines(data, val):
            l = [ aspect[::-1] * ( p + shift) for p in l ]
            svg.addElement(shb.createLine(l[1][1], l[1][0], l[0][1], l[0][0], strokewidth=width, stroke=color))
    return svg

shb = builders.ShapeBuilder()
def make_line(l, width=1, color="rgb(0,255,0)"): 
    return shb.createLine(l[1][1], l[1][0], l[0][1], l[0][0], strokewidth=width, stroke=color)

def param_isocontours(dataset, isovalues, f, scale=1., **kwargs):
    '''param_isocontours(Options dataset, list isovalues, function (Line line, isovalue, (i,j) -> A) f) yields A 

    Computes the line segments for each isocontour and calls f with these arguments:
    Line line: the line segment
    isovalue
    tuple (i,j): the indices of the top left corner of the grid square containing the line segment
    '''

    data = png_to_ndarray(dataset.fname)
    lines = []
    r, c = data.shape
    aspect = scale * np.array(dataset.aspect)
    shift = np.array([.5,.5])
    for i, j in cartesian(xrange(r-1), xrange(c-1)):
        top_left = np.array((i, j)) + shift
        for val in isovalues:
            for l in lookup[tuple((data[i:i+2,j:j+2] > val).flatten())]:
                points = []
                for s in l:
                    # calculate a weight, and then use it to average the given points
                    weight = (val - data[i+s[0][0],j+s[0][1]]) / (data[i+s[1][0], j+s[1][1]] - data[i+s[0][0], j+s[0][1]])
                    #code.interact(local=locals())
                    points.append(aspect[::-1] * (top_left + s[0] + weight*(s[1] - s[0])))
                    #print Options(top_left=top_left, s=s, weight=weight)
                yield f(Line(*points), val, (i,j))

def hsl2rgb(h, s, l):
    return "rgb(%d,%d,%d)" % ImageColor.getrgb("hsl(%d,%d%%,%d%%)" % (h,s,l))

def mask(min, max):
    return np.vectorize(lambda x: x if x > min and x < max else 0)

#==============================================================================
# Run these functions to solve Part 2

def part2a(dataset=images.noise, value=40000., saveas="output/part2a/"):
    if not os.path.exists(saveas): os.mkdir(saveas)
    svg = write_svg(dataset, 'noise-lines', scale=37.5)
    shb = builders.ShapeBuilder()
    for line in param_isocontours(dataset, [value], lambda line, val, index: make_line(line), scale=37.5):
        svg.addElement(line)
    svg.save(saveas + 'noise-lines.svg')
    #overlay_isocontours(dataset, [value], svg, scale=37.5).save(saveas + 'noise-lines.svg')

def part2b(dataset=images.rings, value=32000, saveas='output/part2b/'):
    if not os.path.exists(saveas): os.mkdir(saveas)
    svg = write_svg(dataset, 'rings-lines', scale=1.)
    shb = builders.ShapeBuilder()
    for line in param_isocontours(dataset, [value], lambda line, val, index: make_line(line)):
        svg.addElement(line)
    svg.save(saveas + 'rings-lines.svg')
    #overlay_isocontours(dataset, [value], svg, scale=1.).save(saveas + 'rings-lines.svg')

def part2c(dataset=images.mich, values=None, saveas="output/part2c/"):
    def f(line, val, index):
        if val == 31700:
            return make_line(line, color="rgb(0,0,0)", width=2)
        else: return make_line(line, color="rgb(128,128,128)")

    if not os.path.exists(saveas): os.mkdir(saveas)
    cm.rgb2image(cm.ColorMap.from_hsl_file('mich2b.txt')(cm.read_image(dataset.fname, color='grey'))).save('output/mich2b.png')
    svg = write_svg(Options(fname='output/mich2b.png', aspect=(1,1)), 'mich-lines', scale=1.)
    for line in param_isocontours(dataset, values or list(np.linspace(15000, 31700, num=5)), f):
        svg.addElement(line)
    svg.save(saveas + 'mich-lines.svg')
    return

def part2d(dataset=images.feeth, values=None, saveas="output/part2d/"):
    MAXWIDTH = 3
    #values = [20000., 24800., 30000.]
    values = map(float, [20000, 32000, 35000, 40000])
    colors = { 20000. : hsl2rgb(60,50,55), 
            28000. : hsl2rgb(120,50,50),
            29000. : hsl2rgb(180,50,50)}
    gm = grad_mag(png_to_ndarray(dataset.fname), dataset.aspect)
    gm_scale = 1. / gm.max()
    def f(line, val, (i,j)):
        width = gm[i:i+2,j:j+2].sum() * gm_scale / 4 * MAXWIDTH
        color = hsl2rgb(60 if val < 30000 else 240, width * 100/MAXWIDTH, 50)
        return make_line(line, color=color, width=width)
        #return make_line(line, color=colors[val], width=0 + gm[i:i+2,j:j+2].sum()**2 * gm_scale)

    if not os.path.exists(saveas): os.mkdir(saveas)
    svg = builders.svg('feeth-lines')
    for line in param_isocontours(dataset, values, f):
        svg.addElement(line)
    svg.save(saveas + 'feeth-lines.svg')

def part2e(dataset=images.tooth, values=None, saveas="output/part2e/"):
    #values = list(np.hstack((np.linspace(0,5000,num=3), np.array([15000]), np.linspace(38000,42000, num=1), np.linspace(60000,63000,num=2))))
    values = map(float, [15000, 38000, 42000, 60000]) # 42000, 
    WHITE = hsl2rgb(0, 0, 100)
    YELLOW = hsl2rgb(60, 75, 70)
    RED = hsl2rgb(0,50,50)
    def f(line, val, (i,j)):
        return make_line(line, color=WHITE if val > 45000 else YELLOW if val > 20000 else RED, width=2 if val != 42000 else 1)

    if not os.path.exists(saveas): os.mkdir(saveas)
    Image.fromarray(np.zeros((161,94), dtype='int32'), mode='I').save(saveas + 'solid-bg.png')
    svg = write_svg(Options(fname=saveas + 'solid-bg.png', aspect=(1,1)), 'tooth-lines', scale=1.)
    for line in param_isocontours(dataset, values, f):
        svg.addElement(line)
    svg.save(saveas + 'tooth-lines.svg')

def part2f(dataset=images.teddy, values=None, saveas="output/part2f/"):
    MAXWIDTH = 3.
    # < 3000 background
    # 9000 ~ 11000 fur
    # 12000 ~ 16000 seams
    # 63000 ~ 65000 eyes, nose
    values  = list(np.hstack((np.linspace(0,65535,num=30),)))
    gm = grad_mag(png_to_ndarray(dataset.fname), dataset.aspect)
    gm_scale = MAXWIDTH / gm.max()
    def f(line, val, (i,j)):
        width = gm[i:i+2,j:j+2].sum() * gm_scale * .25
        return make_line(line, width=width, color="rgb(0,0,0)")


    if not os.path.exists(saveas): os.mkdir(saveas)
    Image.fromarray(np.zeros((161,94), dtype='int32'), mode='I').save(saveas + 'solid-bg.png')
    #svg = write_svg(Options(fname=saveas + 'solid-bg.png', aspect=(1,1)), 'teddy-lines', scale=1.)
    svg = builders.svg('teddy-lines')
    for line in param_isocontours(dataset, values, f):
        svg.addElement(line)
    svg.save(saveas + 'teddy-lines.svg')


#==============================================================================
# Run this function to generate all output

def do_all():

    for f in [ v for k, v in globals().items() if 'part' == k[:4] ]:
        f()

if __name__ == '__main__':
    part2c(dataset=images.mich)
