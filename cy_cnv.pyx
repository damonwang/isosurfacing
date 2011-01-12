#!/usr/bin/env python
# cython: profile=True

from __future__ import with_statement
import numpy as np
import Image
import code
import os
from os import path
import collections
from itertools import repeat, product
from pysvg import builders
import ImageColor
import cm
import sys

from matplotlib.pyplot import imread

cimport numpy as np
from stdlib cimport *
import cython

# profiles inner functions
__deep_profile__ = False

#==============================================================================
# General

class Options(dict):
    '''a thin dict wrapper that aliases getitem to getattribute'''

    def __getattr__(self, name):
        return self.__getitem__(name)

# aspect ratios are specified (fast-axis, slow-axis), so (1, 3) means the
# samples are three times as widely spaced along the slow axis as on the fast.
# This usually (e.g., in a web browser) means the picture looks vertically
# squished when displayed improperly.
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

    Returns int16's or int8's in dtype-int32 ndarrays to satisfy odd quirk in PIL.
    '''

    rv = imread(filename)
    if rv.ndim == 2:
        rv *= 65535 #* maximum int16 value
        return rv.astype('int32')
    else:
        rv *= 256 #* maximum int8 value
        return rv.astype('int32')

#==============================================================================
# Functions for Part 2

cdef struct IsoSeg_t:
    double start[2], end[2]
    int isovalue

@cython.profile(__deep_profile__)
cdef inline void IsoSeg(double r1, double c1, double r2, double c2,
        int isovalue, IsoSeg_t * rv):
    rv[0].start[0] = r1
    rv[0].start[1] = c1
    rv[0].end[0] = r2
    rv[0].end[1] = c2
    rv[0].isovalue = isovalue

# four bits: r0, c0, r1, c1
ctypedef char Side_t 

cdef Side_t T=1    # TOP    0,0 -> 0,1
cdef Side_t L=2    # LEFT   0,0 -> 1,0
cdef Side_t R=7    # RIGHT  0,1 -> 1,1
cdef Side_t B=11   # BOTTOM 1,0 -> 1,1

ctypedef char Line_t # eight bits, two Sides

cdef enum: # numerical constants
    SIZEOF_SIDE = 4
    SIZEOF_LINE = sizeof(Line_t) * 8

cdef inline Line_t Line(Side_t start, Side_t end):
    return (start << SIZEOF_SIDE) | end

ctypedef int Intersect_t # four bytes, two used for two Lines

cdef inline Intersect_t Intersect(Line_t a, Line_t b):
    return (a << SIZEOF_LINE) | b

# 2D marching cubes table, with ambiguous cases drawn to connect the
# higher-value corners. Keyed by this bit-string:
#       MSB...top-left top-right bottom-left bottom-right...LSB
# a 1 bit means that corner is above the isovalue
    
cdef Intersect_t lookup[16]
lookup[0] = Intersect(0, 0)
lookup[1] = Intersect(Line(R, B), 0)
lookup[2] = Intersect(Line(L, B), 0)
lookup[3] = Intersect(Line(L, R), 0)
lookup[4] = Intersect(Line(T,R), 0)
lookup[5] = Intersect(Line(T,B), 0)
lookup[6] = Intersect(Line(T, L), Line(R, B))
lookup[7] = Intersect(Line(T, L), 0)
lookup[8] = Intersect(Line(T, L), 0)
lookup[9] = Intersect(Line(T, R), Line(L, B))
lookup[10] = Intersect(Line(T, B), 0)
lookup[11] = Intersect(Line(T, R), 0)
lookup[12] = Intersect(Line(L, R), 0)
lookup[13] = Intersect(Line(L, B), 0)
lookup[14] = Intersect(Line(R, B), 0)
lookup[15] = Intersect(0, 0)

@cython.profile(__deep_profile__)
def write_svg(dataset, name, scale=1, img=True):
    '''write_svg(Options dataset, str name, num scale) -> SVG'''

    if 'size' in dataset:
        shape = dataset.size
    else: shape = png_to_ndarray(dataset.fname).shape

    shb = builders.ShapeBuilder()
    out = builders.svg(name)
    img = builders.image(x=0, y=0, width=shape[1], height=shape[0])
    img.set_xlink_href(path.abspath(dataset.fname))
    grp = builders.g()
    grp.set_transform("scale(%f, %f)" % tuple(scale * np.array(dataset.aspect)))
    grp.addElement(img)
    out.addElement(grp)
    return out

shb = builders.ShapeBuilder()
def make_line(l, width=1, color="rgb(0,255,0)"): 
    return shb.createLine(l[1][1], l[1][0], l[0][1], l[0][0], strokewidth=width, stroke=color)

@cython.profile(__deep_profile__)
cdef inline int min4(int a, int b, int c, int d): 
    cdef int rv
    rv = a
    if b < rv: rv = b
    if c < rv: rv = c
    if d < rv: rv = d
    return rv

@cython.profile(__deep_profile__)
cdef inline int max4(int a, int b, int c, int d): 
    cdef int rv
    rv = a
    if b > rv: rv = b
    if c > rv: rv = c
    if d > rv: rv = d
    return rv

def make_minmax(np.ndarray[np.int32_t, ndim=2] data, Py_ssize_t r, Py_ssize_t c):
    '''
    make_minmax(np.ndarray[np.int32_t, ndim=2] data, Py_ssize_t rows,
        Py_ssize_t cols) -> ndarray

    rows, cols: the shape of the data ndarray

    Returns an ndarray where [i, j, 0] is the minimum and [i, j, 1] the maximum
    of the four values at data[i:i+2,j:j+2].
    '''

    cdef Py_ssize_t i, j
    cdef np.ndarray[np.int32_t, ndim=3] rv

    rv = np.zeros((r-1,c-1,2), dtype='int32')
    for i in xrange(r-1):
        for j in xrange(c-1):
            rv[i,j,0] = min4(data[i,j], data[i,j+1], data[i+1, j], data[i+1, j+1])
            rv[i,j,1] = max4(data[i,j], data[i,j+1], data[i+1, j], data[i+1, j+1])
    return rv

def unpack_params(dataset, isovalues, scale=1., **kwargs):

    '''unpack_params(Options dataset, list isovalues, float scale) -> [ line ]

    A Python-side wrapper which calls param_isocontours to calculate the
    isocontours at the given isovalues, returning a list specifying line
    segments as tuples (x0, y0, x1, y1)

    This function builds two intermediate data structures. For efficiency when
    finding isocontours repeatedly on the same data, you can cache these
    structures externally and pass them in using these keywords: 
        data: an ndarray such as that returned by png_to_ndarray(dataset.fname)
        minmax: an ndarray as that returned by make_minmax(data, *data.shape)
    You should probably pass in neither or both, to avoid inconsistency.

    Do NOT interpolate or resample `data` to force 1:1 aspect ratio. 
    '''

    cdef int nvals = len(isovalues)

    if 'data' not in kwargs:
        data = png_to_ndarray(dataset.fname)
    else: data = kwargs['data']

    if 'minmax' not in kwargs:
        minmax = make_minmax(data, *data.shape)
    else: minmax = kwargs['minmax']

    aspect = scale * np.array(dataset.aspect[::-1])

    rv = param_isocontours(data, aspect, minmax, nvals,
            np.asarray(isovalues, dtype='int32'))

    return rv

@cython.profile(__deep_profile__)
cdef inline int get_bits(int c, int i, int width):
    return (c >> (i * width)) & ~( ~0 << width)

@cython.profile(__deep_profile__)
cdef void lin_interp(int val, double * r, double * c, Side_t s,
        Py_ssize_t i, Py_ssize_t j, int* data_arr):
    '''
    lin_interp(int val, double * r, double * c, Side_t s,
        Py_ssize_t i, Py_ssize_t j, int* data_arr) -> None

    val: isovalue
    r, c: pointers to places to write the type double results
    s: the side along which to interpolate
    i, j: index-space indices of the top-left value
    data_arr: an array of length four containing top-left, top-right,
        bottom-left, bottom-right values of a square, in that order

    Linear interpolation for the index-space coordinates of the point along
    side `s` of of the square with top-left index `i, j` where the data should
    have value `val`.

    Writes row, col to places in memory because C can't return two values. 
    '''

    cdef int r0 = get_bits(s, 3, 1)
    cdef int c0 = get_bits(s, 2, 1)
    cdef int r1 = get_bits(s, 1, 1)
    cdef int c1 = get_bits(s, 0, 1)
    cdef int y0 = data_arr[2 * r0 + c0]
    cdef int y1 = data_arr[2 * r1 + c1]
    cdef double weight = ((val - y0) / <double>(y1 - y0))

    r[0]  = i + .5 + r0 + weight * ( r1 - r0 )
    c[0] = j + .5 + c0 + weight * (c1 - c0)

cdef IsoSeg_t segments[2]

@cython.profile(__deep_profile__)
def find_isoseg(int val, np.ndarray[np.int32_t, ndim=2] data, 
        object aspect_obj, Py_ssize_t i, Py_ssize_t j):
    '''
    find_isoseg(int val, np.ndarray[np.int32_t, ndim=2] data, 
        object aspect_obj, Py_ssize_t i, Py_ssize_t j) -> int

    val: isovalue at which to draw the isocontour
    data: data
    aspect_obj: the float64 ndarray containing data's aspect ratio
    i, j: the indices of the top-left of the square's four corners 

    Returns the number of line segments in this square. The line segments
    themselves are written into the global IsoSeg_t array `segments`
    (therefore, this function is NOT re-entrant, in case that matters)

    Note that the line segments are returned in world-space.
    '''

    # see the definition of `lookup` to understand this bit-twiddling
    cdef char case = (data[i, j] > val) << 3
    case |= (data[i, j+1] > val) << 2
    case |= (data[i+1, j] > val) << 1
    case |= (data[i+1, j+1] > val)

    cdef Intersect_t isect = lookup[case]

    cdef int data_arr[4]
    cdef int m, n
    for m in xrange(2):
        for n in xrange(2):
            data_arr[2*m + n] = data[i+m, j+n]
    cdef double r1, c1, r2, c2
    cdef np.ndarray[np.float64_t, ndim=1] aspect = aspect_obj

    lin_interp(val, &r1, &c1, get_bits(isect, 3, SIZEOF_SIDE), i, j, data_arr)
    lin_interp(val, &r2, &c2, get_bits(isect, 2, SIZEOF_SIDE), i, j, data_arr)

    IsoSeg(aspect[0] * r1, aspect[1] * c1, aspect[0] * r2,
       aspect[1] * c2, val, &segments[0])

    if get_bits(isect, 0, SIZEOF_LINE) != 0:
       lin_interp(val, &r1, &c1, get_bits(isect, 1, SIZEOF_SIDE), i, j, data_arr)
       lin_interp(val, &r2, &c2, get_bits(isect, 0, SIZEOF_SIDE), i, j, data_arr)

       IsoSeg(aspect[0] * r1, aspect[0] * c1, aspect[1] * r2, aspect[1]
              * c2, val, &segments[1])
       return 2
    else: return 1

@cython.profile(__deep_profile__)
cdef isoseg_to_tuple(IsoSeg_t seg):
    return ((seg.start[0], seg.start[1]), (seg.end[0], seg.end[1]),
            seg.isovalue)

def param_isocontours(np.ndarray[np.int32_t, ndim=2] data,  
        object aspect, object minmax, 
        int nvals, np.ndarray[np.int32_t] isovals):
    '''
    param_isocontours(np.ndarray[np.int32_t, ndim=2] data,  
        object aspect, object minmax, 
        int nvals, np.ndarray[np.int32_t] isovals):

    data: data
    aspect: ndarray with aspect ratio of data
    minmax: minmax structure from make_minmax
    nvals: length of isovals array
    isovals: isovalues at which to draw isocontours

    Returns a Python list of tuples. See docstring for unpack_params.

    Please call unpack_params instead. It is much friendlier.
    '''

    #cdef np.ndarray[np.int32_t, ndim=3] minmax_buf = minmax
    cdef Py_ssize_t n, i, j
    cdef int k, val, nsegs

    lines = []
    shift = np.array([.5,.5])
    for n in xrange(nvals):
        val = isovals[n]
        for i, j in np.array(np.where(np.logical_and(minmax[...,0] < val, minmax[...,1] > val))).transpose():
            nsegs = find_isoseg(val, data, aspect, i, j)
            for k in xrange(nsegs):
                lines.append(isoseg_to_tuple(segments[k]))
    return lines

def part2a(dataset=images.noise, value=32000., saveas="output/part2a/"):

    if not os.path.exists(saveas): os.mkdir(saveas)
    svg = write_svg(dataset, 'noise-lines', scale=37.5)
    shb = builders.ShapeBuilder()
    for line in unpack_params(dataset, [value], scale=37.5):
        svg.addElement(make_line(line[:2]))
    svg.save(saveas + 'noise-lines.svg')

def f(start, end, isovalue):
    if isovalue == 31700:
        return make_line((start, end), color="rgb(0,0,0)", width=2)
    else: return make_line((start, end), color="rgb(128,128,128)")

def part2c(dataset=images.mich_sml, values=None, saveas="output/part2c/"):

    if not path.exists(saveas): os.mkdir(saveas)
    colored_map = path.join(saveas, path.basename(dataset.fname))
    topo_map = path.join(saveas, 'mich-lines.svg')
    if path.exists(topo_map): os.unlink(topo_map)
    if not path.exists(colored_map):
        cm.rgb2image(cm.ColorMap.from_hsl_file('mich2b.txt')(cm.read_image(dataset.fname,
            color='grey'))).save(colored_map)
    svg = write_svg(Options(fname=colored_map, aspect=dataset.aspect), 'mich-lines', scale=1.)
    for line in unpack_params(dataset, values or list(np.linspace(15000,
        31700, num=5))):
        svg.addElement(f(*line))
    svg.save(topo_map)
    return
