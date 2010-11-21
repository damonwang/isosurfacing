.. Interactive Isocontours documentation master file, created by
   sphinx-quickstart on Sat Nov 20 14:23:16 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. highlight:: python
   :linenothreshold: 5

=======================
Interactive Isocontours
=======================

Contents:

.. toctree::
   :maxdepth: 2


-------------
Introduction
-------------

This is an extension of Project 3 from Prof. Kindlmann's Scientific
Visualization course (CMSC 23710), which required a two-dimensional
implementation of the `Marching Cubes`_ algorithm. I was disappointed by the
speed, so Prof. Kindlmann suggested two improvements:

1. use the Span Space algorithm with Row/Column representation [Shen96_]

2. use `Cython`_

I implemented these suggestions in four phases:

1. implemented a hacky version of the suggested data structure using Numpy's
   logical indexing
2. moved to Cython
3. moved to C using Cython's seamless FFI and implemented the full data
   structure (forthcoming)
4. demonstrate the improved speed by writing an interactive topographical map
   in WxPytho (forthcoming)

.. _Marching Cubes: http://en.wikipedia.org/wiki/Marching_cubes
.. _Shen96: http://ieeexplore.ieee.org/iel3/4271/12277/00568121.pdf?arnumber=568121
.. _Cython: http://docs.cython.org/src/quickstart/overview.html

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
and 26 overall. ::

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
``param_isocontours`` and 22 overall. **This is more than all the
subsequence Cython tweaking combined.** ::

    Sat Nov 20 13:29:36 2010    Profile.prof

             404286 function calls (398671 primitive calls) in 5.508 CPU seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         5614    1.592    0.000    1.792    0.000 cnv.py:175(param_isocontours)
        37214    0.255    0.000    0.255    0.000 cnv.py:60(grid)
            1    0.242    0.242    5.496    5.496 cnv.py:224(part2c)
         5613    0.044    0.000    0.380    0.000 cnv.py:225(f)
         5613    0.044    0.000    0.336    0.000 cnv.py:172(make_line)

------
Cython
------

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
few hundredths of a second. ::


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

So with five minutes of effort I have saved 1.6 seconds of execution. ::

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

Rewriting

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
