#!/usr/bin/env python

import pstats, cProfile

import cnv

cProfile.runctx("cnv.part2c(dataset=cnv.images.mich_sml)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

