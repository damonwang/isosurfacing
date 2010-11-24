#!/usr/bin/env python

from __future__ import with_statement
import numpy as np
import Image
import code
import os
from os import path
import collections
from itertools import repeat

from pysvg import builders
import ImageColor
import cm
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
IsoSeg = Side = Line = lambda *args: tuple(args)
#Line = collections.namedtuple('Line', 'start end')
#IsoSeg = collections.namedtuple('IsoSeg', 'start end isovalue')
#Side = collections.namedtuple('Side', 'left right')
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

def unpack_params(fname, aspect, isovalues, scale=1., **kwargs):

    if 'data' not in kwargs:
        data = png_to_ndarray(fname)
    else: data = kwargs['data']

    if 'minmax' not in kwargs:
        r, c = data.shape
        minmax = np.array([ (a.min(), a.max()) 
            for a in grid(data) ]).reshape((r-1, c-1, 2))
    else: minmax = kwargs['minmax']

    return param_isocontours(data, map(lambda x: scale * x, aspect), minmax, len(isovalues), isovalues)

def param_isocontours(data, aspect, minmax, nvals, isovalues):
    '''param_isocontours(Options dataset, list isovalues, function (Line line, isovalue, (i,j) -> A) f) yields A 

    Computes the line segments for each isocontour and calls f with these arguments:
    Line line: the line segment
    isovalue
    tuple (i,j): the indices of the top left corner of the grid square containing the line segment
    '''
    lines = []
    aspect = np.array(aspect)
    shift = np.array([.5,.5])
    for val in isovalues:
        for i, j in np.array(np.where(np.logical_and(minmax[...,0] < val, minmax[...,1] > val))).transpose():
            top_left = np.array((i, j)) + shift
            for l in lookup[tuple((data[i:i+2,j:j+2] > val).flatten())]:
                points = []
                for s in l:
                    # calculate a weight, and then use it to average the given points
                    weight = (val - data[i+s[0][0],j+s[0][1]]) / (data[i+s[1][0], j+s[1][1]] - data[i+s[0][0], j+s[0][1]])
                    #code.interact(local=locals())
                    points.append(aspect[::-1] * (top_left + s[0] + weight*(s[1] - s[0])))
                    #print Options(top_left=top_left, s=s, weight=weight)
                yield IsoSeg(points[0], points[1], val)

def make_minmax(data):
    r, c = data.shape
    return np.array([ (a.min(), a.max()) for a in grid(data) ]).reshape((r-1, c-1, 2))


def part2a(dataset=images.noise, value=40000., saveas="output/part2a/"):
    if not os.path.exists(saveas): os.mkdir(saveas)
    svg = write_svg(dataset, 'noise-lines', scale=37.5)
    shb = builders.ShapeBuilder()
    for line in param_isocontours(dataset, [value], scale=37.5):
        svg.addElement(make_line(line[:2]))
    svg.save(saveas + 'noise-lines.svg')

def part2c(dataset=images.mich_sml, values=None, saveas="output/part2c/"):
    def f(start, end, isovalue):
        if isovalue == 31700:
            return make_line((start, end), color="rgb(0,0,0)", width=2)
        else: return make_line((start, end), color="rgb(128,128,128)")

    if not path.exists(saveas): os.mkdir(saveas)
    colored_map = path.join(saveas, path.basename(dataset.fname))
    if not path.exists(colored_map):
        cm.rgb2image(cm.ColorMap.from_hsl_file('mich2b.txt')(cm.read_image(dataset.fname,
            color='grey'))).save(colored_map)
    svg = write_svg(Options(fname=colored_map, aspect=dataset.aspect), 'mich-lines', scale=1.)
    data = png_to_ndarray(dataset.fname)
    r, c = data.shape
    minmax = make_minmax(data)
    for line in unpack_params(dataset.fname, dataset.aspect, values or list(np.linspace(15000,
        31700, num=5)), data=data, minmax=minmax):
        svg.addElement(f(*line))
    svg.save(path.join(saveas, 'mich-lines.svg'))
    return
