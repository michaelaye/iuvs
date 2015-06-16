import time
import sys
from IPython.html.widgets import FloatProgress, IntProgress
from IPython.display import display
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


def int_progress(min, max):
    prog = IntProgress(min=min, max=max)
    display(prog)
    for i in linspace(min, max, 25):
        time.sleep(0.1)
        prog.value = i


def float_progress(min, max):
    prog = FloatProgress(min=min, max=max)
    display(prog)
    for i in linspace(min, max, 100):
        time.sleep(0.1)
        prog.value = i
