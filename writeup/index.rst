.. Interactive Isocontours documentation master file, created by
   sphinx-quickstart on Sat Nov 20 14:23:16 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. highlight:: cython
   :linenothreshold: 5

=======================
Interactive Isocontours
=======================

This is an extension of Project 3 from Prof. Kindlmann's Scientific
Visualization course (CMSC 23710), which required a two-dimensional
implementation of the `Marching Cubes`_ algorithm. I was disappointed by the
speed, so Prof. Kindlmann suggested two improvements:

1. use the Span Space algorithm with Row/Column representation [Shen96_]

2. use `Cython`_

I implemented these suggestions in four phases:

1. mimic the suggested data structure using Numpy's logical indexing
2. port to Cython
3. port to C using Cython's seamless FFI and implement the full data
   structure (forthcoming)
4. demonstrate the improved speed by writing an interactive topographical map
   in WxPython (forthcoming)

The code for this project is available on GitHub_

.. code-block:: bash

    git clone git@github.com:damonwang/isosurfacing.git

.. _Marching Cubes: http://en.wikipedia.org/wiki/Marching_cubes
.. _Shen96: http://ieeexplore.ieee.org/iel3/4271/12277/00568121.pdf?arnumber=568121
.. _Cython: http://docs.cython.org/src/quickstart/overview.html
.. _GitHub: https://github.com/damonwang/isosurfacing/tree/interactive

A few conventions:

* A square is a 2-by-2 slice of the rectangular data array; its corners
  are those four data points.
* We say an isocontour intersects the square if at least one but not all
  of the corner have value below the contour's isovalue.
* The indices of a square are the indices of its top left corner.
* Indices are given row-column format, so the top left element has index
  ``0,0`` and the element below that has index ``1,0``. 

---------
Profiling
---------

Performance was measured with the :class:`cProfile` module using the 188-by-200
pixel ``mich-sml.png`` case from the project. I kept all the original code which
handles, for example, reading in a PNG and writing out SVG, as per the original
project description, so that I could tell when my optimizations produced
incorrect output. Hopefully, this overhead will be constant across all
implementations; even if it isn't, I can still look at the ``totime`` and
``cumtime`` numbers for individual functions.

Here's the Python test script:

.. literalinclude:: ../cnv_py_profile.py

And here's the Cython test script:

.. literalinclude:: ../cnv_profile.py


---------------------------
Naive Python Implementation
---------------------------

Excluding all the overhead for input and output, the important part of the
"Marching Squares" implementation is this for-loop from
``param_isocontours``: ::

    for i, j in product(xrange(r-1), xrange(c-1)):
        top_left = np.array((i, j)) + shift
        for val in isovalues:
            for l in lookup[tuple((data[i:i+2,j:j+2] > val).flatten())]:
                points = []
                for s in l:
                    # calculate a weight, and then use it to average the given points
                    weight = (val - data[i+s[0][0],j+s[0][1]]) / (data[i+s[1][0], j+s[1][1]] - data[i+s[0][0], j+s[0][1]])
                    points.append(aspect[::-1] * (top_left + s[0] + weight*(s[1] - s[0])))
                yield f(Line(*points), val, (i,j))

where ``lookup`` is a dict mapping from 4-tuples of booleans to an encoding of
how to draw the isocontour through that region: ::

    Line = collections.namedtuple('Line', 'start end')
    IsoSeg = collections.namedtuple('IsoSeg', 'start end isovalue')
    Side = collections.namedtuple('Side', 'left right')
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

The performance is pretty bad: 16 CPU seconds in ``param_isocontours``
and 26 overall. 

.. code-block:: none

    Sat Nov 20 15:05:44 2010    Profile.prof

         1178802 function calls (1173066 primitive calls) in 25.845 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         5618   16.486    0.003   18.820    0.003 cnv.py:371(param_isocontours)
        37214    0.200    0.000    0.200    0.000 cnv.py:288(product)
         5617    0.058    0.000    0.432    0.000 cnv.py:368(make_line)
         5617    0.051    0.000    0.483    0.000 cnv.py:426(f)
            1    0.022    0.022   25.833   25.833 cnv.py:425(part2c)


-------------------------------------------------
Span Space via Logical Indexing: 14 seconds saved
-------------------------------------------------

The naive algorithm considers every possible square, even though most of
them are irrelevant. The only squares which will contain a segment of
the isocontour are those where at least one corner is below the isovalue
and at least one corner is above the isovalue. The heart of the Span
Space algorithm is to provide a cheap way to discard most of the
irrelevant samples. A quick and dirty first step toward this
was to using Numpy's `logical indexing`_ to find the relevant squares. 

.. _logical indexing: http://www.scipy.org/Tentative_NumPy_Tutorial#head-d55e594d46b4f347c20efe1b4c65c92779f06268

Instead of ::

    for i, j in product(xrange(r-1), xrange(c-1)):

I wrote ::

    minmax = np.array([ (a.min(), a.max()) for a in grid(data) ]).reshape((r-1, c-1, 2))
    for i, j in np.array(np.where(np.logical_and(minmax[...,0] < val, minmax[...,1] > val))).transpose():

The first line creates an array whose ``i,j`` element is the minimum and
maximum value across the corners of the ``i,j`` square. The second line
iterates only over the indices of the squares which intersect the
isocontour. This still runs in linear time rather than ``O(sqrt(N) +
log(N))`` as promised by the Span Space algorithm, because Numpy still
checks every element of the ``data`` array, but now we're doing linear
work at C speed instead of Python speed.

This one algorithmic improvement saves us 14 CPU seconds in
``param_isocontours`` and 22 overall. 

.. code-block:: none

    Sat Nov 20 13:29:36 2010    Profile.prof

             404286 function calls (398671 primitive calls) in 5.508 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         5614    1.592    0.000    1.792    0.000 cnv.py:175(param_isocontours)
        37214    0.255    0.000    0.255    0.000 cnv.py:60(grid)
            1    0.242    0.242    5.496    5.496 cnv.py:224(part2c)
         5613    0.044    0.000    0.380    0.000 cnv.py:225(f)
         5613    0.044    0.000    0.336    0.000 cnv.py:172(make_line)

-------------------------
Cython: 3.8 seconds saved
-------------------------

Cython is very Python-like language which drops a few higher-level
features like generators but provides an (optional) C-like type system.
This has two benefits:

1. Providing type hints allows Cython to compile all the way down to C,
   so that a for loop is actually a for loop.
2. Cython can call C/C++ functions without the usual FFI business. 

Naive Cython: slightly slower
-------------------------------

The first step was to rewrite my generators to accumulate their results
internally and return a list all at once, since Cython doesn't support
generators. They weren't necessary anyway. So ::

    def grid(data):
        rows, cols = data.shape[:2]
        for r in xrange(rows - 1):
            for c in xrange(cols - 1):
                yield data[r:r+2,c:c+2]

    minmax = np.array([ (a.min(), a.max()) for a in grid(data) ]).reshape((r-1, c-1, 2))

became ::

    def make_minmax(data):
        r, c = data.shape
        rv = np.zeros((r-1,c-1,2))
        for i in xrange(r-1):
            for j in xrange(c-1):
                rv[i,j,0] = data[i:i+2,j:j+2].min()
                rv[i,j,0] = data[i:i+2,j:j+2].max()
        return rv

    minmax = make_minmax(data)

Interestingly enough, moving to Cython actually makes
``param_isocontours`` slightly slower, although the overall time drops a
few hundredths of a second. 


.. code-block:: none

    Sat Nov 20 17:15:06 2010    Profile.prof

             264672 function calls (259057 primitive calls) in 5.440 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    1.897    1.897    3.637    3.637 cy_cnv.pyx:121(param_isocontours)
            1    1.687    1.687    1.687    1.687 cy_cnv.pyx:112(make_minmax)
            2    0.474    0.237    0.493    0.247 cy_cnv.pyx:39(png_to_ndarray)
            1    0.052    0.052    5.440    5.440 cy_cnv.pyx:168(part2c)
         5613    0.052    0.000    0.396    0.000 cy_cnv.pyx:95(make_line)
         5613    0.049    0.000    0.445    0.000 cy_cnv.pyx:163(f)

This is probably due to making expensive dynamically-typed Python calls,
so let's rewrite the two slowest functions in a less Pythonic style.

make_minmax: 1.687s to .061s
------------------------------------------------

I start by providing type hints on ``make_minmax``, which unfortunately
requires replacing Numpy's ``ndarray.min`` and ``ndarray.max`` methods
with some ugly, non-general functions. On the bright side, these are
pure C functions. ::

    cdef int min4(int a, int b, int c, int d): 
        cdef int rv
        rv = a
        if b < rv: rv = b
        if c < rv: rv = c
        if d < rv: rv = d
        return rv

    cdef int max4(int a, int b, int c, int d): 
        cdef int rv
        rv = a
        if b > rv: rv = b
        if c > rv: rv = c
        if d > rv: rv = d
        return rv

    def make_minmax(np.ndarray[np.int32_t, ndim=2] data, Py_ssize_t r, Py_ssize_t c):
        cdef Py_ssize_t i, j
        cdef np.ndarray[np.int32_t, ndim=3] rv

        rv = np.zeros((r-1,c-1,2), dtype='int32')
        for i in xrange(r-1):
            for j in xrange(c-1):
                rv[i,j,0] = min4(data[i,j], data[i,j+1], data[i+1, j], data[i+1, j+1])
                rv[i,j,1] = max4(data[i,j], data[i,j+1], data[i+1, j], data[i+1, j+1])
        return rv

So with five minutes of effort, I have saved 1.6 seconds of execution.

.. code-block:: none

    Sat Nov 20 17:51:25 2010    Profile.prof

             339100 function calls (333485 primitive calls) in 3.649 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    1.895    1.895    2.047    2.047 cy_cnv.pyx:127(param_isocontours)
            2    0.484    0.242    0.503    0.251 cy_cnv.pyx:41(png_to_ndarray)
            1    0.061    0.061    0.099    0.099 cy_cnv.pyx:116(make_minmax)
            1    0.043    0.043    3.649    3.649 cy_cnv.pyx:174(part2c)
         5613    0.042    0.000    0.320    0.000 cy_cnv.pyx:97(make_line)
         5613    0.041    0.000    0.360    0.000 cy_cnv.pyx:169(f)
        37213    0.019    0.000    0.019    0.000 cy_cnv.pyx:108(max4)
        37213    0.019    0.000    0.019    0.000 cy_cnv.pyx:100(min4)

param_isocontours: 1.895s to .197s
-----------------------------------------------

Rewriting ``param_isocontours`` is a bit more involved, because here the
heavy Python calls actually get a lot of work done.

Previously, I had used :meth:`collections.namedtuple` to pass three
kinds of data: 

1. sides of a square, defined by the indices of two corners
2. lines to draw, defined by two sides of the square 
3. intersections of a square and an isocontour, defined by two lines and
   an isovalue

But that required a heavy Python object to carry, in the best case, a
few bytes and, in the worst case, four bits. So I devised a new
representation using C types:

1. since squares are 2-by-2, the corner indices are at most 1 and a side
   can be represented by four bits---two for the indices of each end
2. two sides, therefore, fit into a ``char``
3. intersections required doubles for the corner coordinates, so I
   defined a C struct

Then it was just a matter of laboriously rewriting ``param_isocontours``
as a loop around C-style ``find_isoseg`` function which computed the
intersection of one isocontour with one square. In order to avoid the
overhead of several thousand calls to :meth:`numpy.interp`, a custom
``lin_interp`` function was written in Cython. 

Note that not only did I just hack out over two hundred lines of Cython to
replace three lines of (Numpy-assisted) Python, but those two hundred
lines are about as readable and expressive as straight C. ::

    cdef inline int get_bits(int c, int i, int width):
        return (c >> (i * width)) & ~( ~0 << width)

    cdef void lin_interp(int val, double * r, double * c, Side_t s,
            Py_ssize_t i, Py_ssize_t j, int* data_arr):

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

    def find_isoseg(int val, np.ndarray[np.int32_t, ndim=2] data, 
            object aspect_obj, Py_ssize_t i, Py_ssize_t j):
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

    cdef isoseg_to_tuple(IsoSeg_t seg):
        return ((seg.start[0], seg.start[1]), (seg.end[0], seg.end[1]),
                seg.isovalue)

    def param_isocontours(np.ndarray[np.int32_t, ndim=2] data,  
            object aspect, object minmax, 
            int nvals, np.ndarray[np.int32_t] isovals):

        '''param_isocontours(Options dataset, list isovalues, function (Line line, isovalue, (i,j) -> A) f) yields A 

        Computes the line segments for each isocontour and calls f with these arguments:
        Line line: the line segment
        isovalue
        tuple (i,j): the indices of the top left corner of the grid square containing the line segment
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

So with several hours of work, I have saved a futher 1.6 seconds of
execution. On the bright side, the only function of mine which appears
in the top five is now an I/O call! 

.. code-block:: none

    Sun Nov 21 01:45:08 2010    Profile.prof

             428763 function calls (423148 primitive calls) in 1.709 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            2    0.418    0.209    0.435    0.217 cy_cnv.pyx:42(png_to_ndarray)
       5616/1    0.311    0.000    0.608    0.608 core.py:49(getXML)
        28075    0.166    0.000    0.286    0.000 core.py:111(quote_attrib)
         5613    0.078    0.000    0.142    0.000 shape.py:252(__init__)
        84225    0.078    0.000    0.078    0.000 {method 'replace' of 'str' objects}
         5613    0.059    0.000    0.275    0.000 builders.py:137(createLine)
            1    0.058    0.058    0.197    0.197 cy_cnv.pyx:243(param_isocontours)
            1    0.055    0.055    0.091    0.091 cy_cnv.pyx:145(make_minmax)
         5613    0.053    0.000    0.059    0.000 builders.py:291(getStyle)
         5575    0.049    0.000    0.127    0.000 cy_cnv.pyx:205(find_isoseg)
        28088    0.042    0.000    0.042    0.000 {isinstance}
        11226    0.038    0.000    0.060    0.000 cy_cnv.pyx:178(lin_interp)
            1    0.033    0.033    1.708    1.708 cy_cnv.pyx:282(part2c)
         5613    0.031    0.000    0.306    0.000 cy_cnv.pyx:126(make_line)
        61705    0.031    0.000    0.031    0.000 cy_cnv.pyx:175(get_bits)
        37213    0.018    0.000    0.018    0.000 cy_cnv.pyx:137(max4)
        37213    0.018    0.000    0.018    0.000 cy_cnv.pyx:129(min4)
         5613    0.017    0.000    0.322    0.000 cy_cnv.pyx:277(f)

-----------------------------------
Minimal profiling: .2 seconds saved
-----------------------------------

And, just for vanity's sake, one run with profiling disabled on all the
functions that get called repeatedly:

.. code-block:: none

    Sun Nov 21 02:50:58 2010    Profile.prof

             264604 function calls (258989 primitive calls) in 1.515 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            2    0.426    0.213    0.447    0.224 cy_cnv.pyx:43(png_to_ndarray)
       5616/1    0.303    0.000    0.610    0.610 core.py:49(getXML)
        28075    0.169    0.000    0.296    0.000 core.py:111(quote_attrib)
        84225    0.087    0.000    0.087    0.000 {method 'replace' of 'str' objects}
         5613    0.079    0.000    0.138    0.000 shape.py:252(__init__)
            1    0.070    0.070    0.077    0.077 cy_cnv.pyx:239(param_isocontours)
         5613    0.061    0.000    0.275    0.000 builders.py:137(createLine)
         5613    0.054    0.000    0.060    0.000 builders.py:291(getStyle)
        28088    0.040    0.000    0.040    0.000 {isinstance}
            1    0.034    0.034    1.515    1.515 cy_cnv.pyx:279(part2c)
         5613    0.031    0.000    0.307    0.000 cy_cnv.pyx:129(make_line)
         5613    0.016    0.000    0.323    0.000 cy_cnv.pyx:274(f)

So really, ``param_isocontour`` is down to .077s, not .197s as
previously reported.

-----------------------------------------------
imread instead of Image.open: .35 seconds saved
-----------------------------------------------

I replaced 

.. code-block:: cython

    def png_to_ndarray(filename): # pragma: no cover
        '''png_to_ndarray(str filename) -> ndarray
        '''

        im = Image.open(filename)
        try:
            return np.array(im.getdata()).reshape(im.size[::-1])
        except ValueError: # rgb data?
            return np.array(im.getdata

with the matplotlib equivalent:

.. code-block:: cython

    def png_to_ndarray(filename): # pragma: no cover
        '''png_to_ndarray(str filename) -> ndarray 
        '''

        rv = imread(filename)
        if rv.ndim == 2:
            rv *= 65535
            return rv.astype('int32')
        else:
            rv *= 256
            return rv.astype('int32')

And now I/O is no longer my single most expensive operation.

.. code-block:: none

    Sun Nov 21 03:47:17 2010    Profile.prof

             263906 function calls (258291 primitive calls) in 1.184 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       5616/1    0.314    0.000    0.612    0.612 core.py:49(getXML)
        28075    0.167    0.000    0.285    0.000 core.py:111(quote_attrib)
         5613    0.099    0.000    0.174    0.000 shape.py:252(__init__)
            1    0.089    0.089    0.099    0.099 cy_cnv.pyx:243(param_isocontours)
        84225    0.078    0.000    0.078    0.000 {method 'replace' of 'str' objects}
         5613    0.070    0.000    0.335    0.000 builders.py:137(createLine)
         5613    0.065    0.000    0.072    0.000 builders.py:291(getStyle)
        28077    0.040    0.000    0.040    0.000 {isinstance}
            1    0.039    0.039    1.184    1.184 cy_cnv.pyx:283(part2c)
         5613    0.037    0.000    0.372    0.000 cy_cnv.pyx:133(make_line)
         5613    0.020    0.000    0.392    0.000 cy_cnv.pyx:278(f)
         5618    0.019    0.000    0.019    0.000 {method 'keys' of 'dict' objects}
         5616    0.017    0.000    0.036    0.000 core.py:81(setKWARGS)
         5615    0.016    0.000    0.020    0.000 core.py:40(addElement)
         5616    0.015    0.000    0.015    0.000 core.py:26(__init__)
        11229    0.015    0.000    0.015    0.000 {method 'items' of 'dict' objects}
            2    0.011    0.005    0.011    0.005 {built-in method read_png}
        11154    0.010    0.000    0.010    0.000 numpy.pxd:187(__getbuffer__)
         5613    0.010    0.000    0.010    0.000 builders.py:186(__init__)
         5613    0.008    0.000    0.008    0.000 attributes.py:62(set_style)
         5613    0.007    0.000    0.007    0.000 shape.py:271(set_x1)
         5613    0.006    0.000    0.006    0.000 shape.py:286(set_y2)
         5613    0.006    0.000    0.006    0.000 shape.py:276(set_y1)
         5613    0.005    0.000    0.005    0.000 shape.py:281(set_x2)
         5616    0.005    0.000    0.005    0.000 {len}
         5622    0.004    0.000    0.004    0.000 {method 'append' of 'list' objects}
            2    0.004    0.002    0.015    0.007 cy_cnv.pyx:45(png_to_ndarray)
            1    0.003    0.003    0.003    0.003 {method 'write' of 'file' objects}
            1    0.002    0.002    0.002    0.002 cy_cnv.pyx:154(make_minmax)
            1    0.001    0.001    0.001    0.001 core.py:95(wrap_xml)
            1    0.000    0.000    0.617    0.617 core.py:102(save)
            1    0.000    0.000    0.106    0.106 cy_cnv.pyx:165(unpack_params)


----------
Conclusion
----------

I started with a naive algorithm in Python, saved 14 seconds via a
(slightly) smarter algorithm, and then saved another 4 seconds by
rewriting several functions in Cython in a C-like style. Probably the
correct conclusion is that this sort of nitpicky hand-optimization
should come only after all possible algorithmic improvements have been
considered.
