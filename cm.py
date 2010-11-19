#!/usr/bin/env python

from __future__ import with_statement

import PIL
import numpy as np
from matplotlib import pyplot as pp
from optparse import OptionParser

import os

import Image
import ImageColor

import code

def add_options(parser):
    '''add_options(parser) -> parser

    initializes the given OptionParser and returns it.'''

    parser.add_option("-f", "--file", type="string", dest="file", 
            help="imread FILE as the dataset")

    return parser

def interp(*args, **kwargs):
    '''currently a wrapper for numpy's interp'''

    return np.interp(*args, **kwargs)

def apply_aspect_ratio(s, f, img):
    '''apply_aspect_ratio(slow, fast, image)

    Assuming the image begins as s:f aspect ratio, upsample one axis with
    nearest-neighbor interpolation to give 1:1 as the final aspect ratio.

    If neither s nor f is 1, resample from 1:f/s, altering only the fast axis.
    '''

    if s < 1 or f < 1:
        m = min(s,f)
        return apply_aspect_ratio(s/m, f/m, img)
    if s != 1:
        if f == 1:
            return apply_aspect_ratio(f, s, img.transpose()).transpose()
        else:
            return apply_aspect_ratio(1, f/s, img)

    mask = interp(np.arange(0, img.shape[1]/f), [0, img.shape[1]/f], [0, img.shape[1]]).astype(int)
    def scale_one_row(row):
        return row[mask]

    return np.asarray(map(scale_one_row, img))

def histogram(*args, **kwargs):
    '''a wrapper around numpy's histogram that plots the histogram.'''

    log = False
    if 'log' in kwargs:
        log = kwargs['log']
        del kwargs['log']

    vals, bins = np.histogram(*args, **kwargs)

    if log: vals = np.log(vals)
    pp.plot(bins[:-1], vals)

def rgb2image(A, mode="RGB"):
    '''rgb2image(ndarray float) -> Image.Image instance'''

    return Image.fromarray(A.astype('uint8'), mode=mode)

def scale_to(maxval, A):
    return maxval / A.max() * A

def read_image(filename, *args, **kwargs):
    '''read_image(filename) -> ndarray

    numpy's imread with a few project-specific post-processing steps tacked on,
    depending on the options given:

        color=grey: assuming image is 2D rgba,
            grey: slice like [:,:,0]
        scale_to=max: rescale so highest value in image is the given max
        dtype=type: convert to given numpy type

    
    post-processing done in the order listed, not the order in which the
    keywords are specified.
    '''

    def color(arg, img):
        if arg == "grey":
            return img[:,:,0] if len(img.shape) > 2 else img
        else: raise ValueError("color must be 'grey'")

    def dtype(arg, img):
        return img.astype(arg)

    filters = [ color, scale_to, dtype ]

    rv = pp.imread(filename)
    for f in filters:
        if f.__name__ in kwargs: rv = f(kwargs[f.__name__], rv)

    return rv

def hsl2rgb(hsl):
    '''given a length-3 ndarray containing hue, saturation, and value, returns
    an ndarray of length three with the corresponding rgb representation.
    '''

    return np.asarray(ImageColor.getrgb("hsl(%d,%d%%,%d%%)" % tuple(scale_to(255,hsl))))

class BivariateColorMap(object):
    '''Represents a color map. Callable object which takes two 2D ndarrays,
    each representing an image, and returns returns the colored version in rgb
    format'''

    @staticmethod
    def from_univariates(S, F, merge=lambda x,y: x + y, colortype='rgb'):
        '''combines the given slow and fast axes using the given operation.

        e.g., S varies from dark blue to bright blue,
              F varies from dark green to bright green,
              merge is addition
        gives a greyscale ramp along the diagonal and the pure colors in the
        other two corners.
        '''

        framp = np.linspace(F.points[:,0].min(), F.points[:,0].max(), num=256)
        sramp = np.linspace(S.points[:,0].min(), S.points[:,0].max(), num=256)
        points = np.asarray([ [ merge(s, f) for f in F(framp)] for s in S(sramp) ])
        if colortype == 'rgb':
            return BivariateColorMap(scale_to(255, points).astype('uint8'), colortype=colortype)
        else: return BivariateColorMap(points, colortype=colortype)

    @staticmethod
    def from_uni_hsl_files(sfile, ffile, merge=lambda x,y: x + y, colortype='rgb'):
        '''constructs the univariate maps from the given files, then calls from_univariates.'''

        S = ColorMap.from_hsl_file(sfile)
        F = ColorMap.from_hsl_file(ffile)

        return BivariateColorMap.from_univariates(S, F, merge, colortype=colortype)

    def __init__(self, points, colortype='rgb'):
        '''points must be an ndarray of shape (256, 256, 3) representing a 2D
        image in rgb format.'''

        self.points = points
        self.colortype = colortype

    def __call__(self, A, B):

        A = scale_to(255, A).astype('uint8')
        B = scale_to(255, B).astype('uint8')
        
        def gen():
            for row_a, row_b in zip(A, B):
                for a, b in zip(row_a, row_b):
                    yield self.points[b][a]

        if self.colortype == 'rgb':
            C = list(gen())
        elif self.colortype == 'hsl':
            C = map(hsl2rgb, gen())
        else: raise ValueError("colortype not recognized")

        return np.asarray(C).reshape(A.shape + (3,))

    def colorbar(self):
        '''colorbar() -> ndarray

        shape will be (256, 256, 3) in rgb format
        dtype will be uint8'''

        return self.points.astype('uint8')

    def show(self):

        rgb2image(self.colorbar()).show()
    

class ColorMap(object):
    '''Represents a color map. Callable object which takes a 2D ndarray and
    returns the corresponding (red, green, blue) tuple.'''

    @staticmethod
    def double_ended(low, zero, high, *args, **kwargs):
        '''double_ended(low, zero, high) -> ColorMap

        low, zero, high should be scalar. 

        Returns a ColorMap that is grey at zero and increases in saturation
        toward low and high.'''

        lowcolor = ImageColor.getrgb("hsl(180,100%,50%)")
        highcolor = ImageColor.getrgb("hsl(0,100%,50%)")
        zerocolor = ImageColor.getrgb("hsl(0,100%,0%)")

        points = np.asarray([ (low,) + lowcolor,
            (zero,) + zerocolor,
            (high,) + highcolor ])

        return ColorMap(points)

    @staticmethod
    def linear_lum_hue(low, high, *args, **kwargs):
        '''linear_lum_hue(low, high) -> ColorMap

        returns a ColorMap that starts with 25% luminosity and blue at low and
        increases linearly in luminosity and hue to 75% and red at high.

        A better colormap would walk through the hues in some way that
        compensates for the fact that hues have different widths on a linear
        ramp---e.g., a large swath looks green.
        '''

        n = 10
        points = [ (val,) + col for val, col in zip(*[ np.linspace(low, high, num=n), 
            [ ImageColor.getrgb("hsl(%d,100%%,%d%%)" % (c % 360, l)) 
                for c,l in zip(np.linspace(350,600,num=n), np.linspace(60,25,num=n)) ][::-1] ]) ]

        return ColorMap(np.asarray(points))

    @staticmethod
    def from_file(filename, *args, **kwargs):
        '''from_file(filename) -> ColorMap

        reads the file as a list of control points specified
            value   red     green   blue
        one per line and returns the appropriate ColorMap.

        TODO
        currently casts everything to float when ints should do, but that needs
        me to scale all my input images to 0..2^16 instead of 0..1 as provided
        by imread.
        '''

        with open(filename) as inf:
            points = [ map(float, line.split()) for line in inf ]

        return ColorMap(np.asarray(points))

    @staticmethod
    def from_hsl_file(filename, *args, **kwargs):
        '''from_hsl_file(filename) -> ColorMap

        reads the file as a list of control points specified as suitable for
        ImageColor.getrgb's hsl format, namely,
            value   hue     saturation   luminosity
        one per line and returns the appropriate ColorMap.

        TODO
        see from_file()
        '''

        points = []
        with open(filename) as inf:
            for line in inf:
                fields = line.split()
                points.append((float(fields[0]),) + ImageColor.getrgb("hsl(%d,%d%%,%d%%)" % tuple(map(int, fields[1:]))))

        return ColorMap(np.asarray(points))

    def __init__(self, points):
        '''points is an ndarray [ [ value, red, green, blue ] ]

        interpolates linearly between the given points.'''

        self.points = points
        self.ufunc = np.vectorize(self.map_one)

    def __call__(self, A):

        if not isinstance(A, np.ndarray):
            raise TypeError("input must be ndarray")
        return np.asarray(self.ufunc(A.transpose())).transpose()
    
    def map_one(self, x):

        '''map_one(x) -> (red, green, blue)

        returns the color in rgb format that corresponds to the given input.

        TODO:
        returns float32 types; bad, but not worth optimizing right now.
        '''
        
        return tuple([interp(x, self.points[:,0], self.points[:,i])
                for i in range(1, 4)])

    def colorbar(self, height, width, vertical=True):
        '''colorbar(height, width) -> ndarray

        returns a colorbar of given height and width (in pixels), running from
        the lowest control point to the highest.

        Caller should pass result into rgb2image to get an Image.Image.
        '''

        # linspace from highest control point at top to lowest at bottom
        if vertical:
            A = np.linspace(self.points[-1][0], self.points[0][0], num=height)
            A = np.asarray([A] * width).transpose()
        else:
            A = np.linspace(self.points[0][0], self.points[-1][0], num=width)
            A = np.asarray([A] * height)
        return self(A)

    def show(self):

        rgb2image(self.colorbar(200,20)).show()

    def to_file(self, filename):
        '''to_file(filename) -> None

        writes out this ColorMap in a way suitable for from_file to read again.

        In particular, this means rgb format even if the ColorMap was
        originally created from an hsl format file.
        '''

        text = '\n'.join([ "%f\t%d\t%d\t%d" % tuple(p) for p in self.points ])

        with open(filename, "w") as outf:
            print >> outf, text


def foreach(A, fn, min_dim = 1):
    '''foreach(A, fn) -> ndarray

    assumes fn(x) -> x'

    returns a copy of A with every element replaced by fn(elt)'''

    if len(A.shape) > min_dim:
        return np.asarray([ foreach(a, fn) for a in A ])
    else: return np.asarray([ fn(a) for a in A ])

def main():
    '''generates all the necessary output for the write-up.'''

    data_dir = 'data'
    output_dir = 'output'
    ct_feet_path = os.path.join(data_dir, 'ct-feet.png')
    ct_feet_1to1_path = os.path.join(output_dir, 'ct-feet-1to1.png')
    mich_path = os.path.join(data_dir, 'elev-mich.png')
    myst_path = os.path.join(data_dir, 'elev-myst.png')
    myst_1to1_path = os.path.join(output_dir, 'elev-myst-1to1.png')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for s, f, img, out in [(.722, 1, ct_feet_path, ct_feet_1to1_path),
            (1, 1.6, myst_path, myst_1to1_path)]:
        print "rescaling %s from %f:%f to 1:1 ..." % (ct_feet_path, s, f),
        with open(out, "w") as outf:
            rgb2image(apply_aspect_ratio(s, f, read_image(img, scale_to=255)), mode="RGBA").save(outf)
        print "saved to %s" % out

    for color, data in [('myst2a.txt', myst_1to1_path), ('mich2a.txt', mich_path),
            ('mich2b.txt', mich_path), ('myst2b.txt', myst_1to1_path),
            ('ct-feet3a.txt', ct_feet_1to1_path), ('ct-feet3b.txt', ct_feet_1to1_path)]:
        print "applying %s map to %s ..." % (color, data),
        data_name, data_ext = os.path.splitext(os.path.basename(data))
        out = os.path.join(output_dir, "%s_%s.jpg" % (data_name, os.path.splitext(color)[0]))
        with open(out, "w") as outf:
            rgb2image(ColorMap.from_hsl_file(color)(read_image(data, color='grey'))).save(outf)
        print "saved image to %s" % out,
        out = os.path.join(output_dir, "%s_colorbar.jpg" % (os.path.splitext(color)[0]))
        with open(out, "w") as outf:
            rgb2image(ColorMap.from_hsl_file(color).colorbar(240, 20)).save(outf)
        print "saved colorbar to %s" % out
        out = os.path.join(output_dir, color)
        ColorMap.from_hsl_file(color).to_file(out)
        print "saved rgb map to %s" % out

if __name__ == '__main__':

    main()
