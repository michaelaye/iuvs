import sys
import time

from IPython.display import display
from ipywidgets import FloatProgress, IntProgress
from numpy import linspace


def progress_display(results, objectlist, sleep=10):
    while not results.ready():
        print("{:.1f} % done.".format(100*results.progress/len(objectlist)))
        sys.stdout.flush()
        time.sleep(sleep)


def nb_progress_display(results, objectlist, sleep=1):
    prog = IntProgress(min=0, max=len(objectlist))
    display(prog)
    while not results.ready():
        prog.value = results.progress
        time.sleep(sleep)


def int_progress(min_, max_):
    prog = IntProgress(min=min_, max=max_)
    display(prog)
    for i in linspace(min_, max_, 25):
        time.sleep(0.1)
        prog.value = i


def float_progress(min_, max_):
    prog = FloatProgress(min=min_, max=max_)
    display(prog)
    for i in linspace(min_, max_, 100):
        time.sleep(0.1)
        prog.value = i
