# from bokeh.plot_object import PlotObject
from bokeh.server.utils.plugins import object_page
from bokeh.server.app import bokeh_app
# from bokeh.plotting import curdoc, cursession
from bokeh.crossfilter.models import CrossFilter
# from bokeh.sampledata.autompg import autompg
import pandas as pd

mydata = pd.read_hdf('/Users/klay6683/data/iuvs/dark_stuff/to_study.h5', 'df')


@bokeh_app.route("/bokeh/crossfilter/")
@object_page("crossfilter")
def make_crossfilter():
    # autompg['cyl'] = autompg['cyl'].astype(str)
    # autompg['origin'] = autompg['origin'].astype(str)
    mydata['INT_TIME'] = mydata['INT_TIME'].astype(str)
    mydata['BINNING_SET'] = mydata['BINNING_SET'].astype(str)
    app = CrossFilter.create(df=mydata)
    return app
