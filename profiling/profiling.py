import pstats
from pstats import SortKey

p = pstats.Stats('simple_hist_timing.dat')
p.sort_stats(SortKey.TIME).print_stats(10)