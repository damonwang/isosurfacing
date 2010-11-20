#!/usr/bin/env python

import pstats, cProfile

import pyximport
pyximport.install()

import cy_cnv

cProfile.runctx("cy_cnv.part2c()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
