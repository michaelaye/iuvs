import time
import sys


def progress_display(results, objectlist, sleep=10):
    while not results.ready():
        print("{:.1f} % done.".format(100*results.progress/len(objectlist)))
        sys.stdout.flush()
        time.sleep(sleep)
