#!/usr/bin/env python

import pstats, cProfile
import sys

if len(sys.argv) > 1 and sys.argv[1] == '--pyx':
    import pyximport
    pyximport.install()
    import cy_cnv as cnv
else: import cnv

cProfile.runctx("cnv.part2c()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
