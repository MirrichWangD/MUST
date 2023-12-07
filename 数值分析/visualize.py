from pyecharts.charts import Geo
from pyecharts import options
from pyecharts.globals import GeoType


addr = "琴海湾"

addr = ["p1", "p2", "p3", "p4"]
latitude = [22.13821354, 22.13823075, 22.1380986, 22.13812574]
longitude = [113.5385331, 113.538336, 113.5386487, 113.5386084]

g = Geo().add_schema(maptype="珠海")
for i, (a, lat, lon) in enumerate(zip(addr, latitude, longitude)):
    g.add_coordinate(a, lon, lat)
    data_pair = [(a, i)]
    g.add("", data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=8)
g.set_series_opts(label_opts=options.LabelOpts(is_show=False))
g.set_global_opts(title_opts=options.TitleOpts(title="pyecharts地图标点测试"))

g.render_notebook()
