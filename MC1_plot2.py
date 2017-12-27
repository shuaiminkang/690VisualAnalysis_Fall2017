
##clustering movement dataset; code for move3.png
##Run: bokeh serve --show MC1_plot2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Slider
from bokeh.models import Select
from bokeh.layouts import row, widgetbox,column,layout
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from bokeh.plotting import curdoc

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show, output_file



Sun_sumry1 = pd.read_csv("Sun_sumry.csv")
Sun_sumry = Sun_sumry1.drop(["Unnamed: 0","id"],axis=1)

Sun_sumry_data = StandardScaler().fit_transform(Sun_sumry)
#Sun_sumry_data = Sun_sumry_data1[:,0:89]


cluster = KMeans(n_clusters=6, random_state=10)
pca = PCA(n_components = 20)
pipe = Pipeline(steps=[('pca', pca), ('cluster', cluster)])
cluster_labels = pipe.fit_predict(Sun_sumry_data[:,0:89])

X = PCA(n_components = 2).fit_transform(Sun_sumry_data[:,0:89])

cluster_color = []
for j,i in enumerate(cluster_labels):
    if i==0:
        cluster_color.append('red')
    if i==1:
        cluster_color.append('blue')
    if i==2:
        cluster_color.append('yellow')
    if i==3:
        cluster_color.append('pink')
    if i==4:
        cluster_color.append('green')
    if i==5:
        cluster_color.append('grey')

color = ['red','blue','yellow','pink','green','grey']

Sun_sumry2 = Sun_sumry.copy()
Sun_sumry2["cluster"] = cluster_labels
Sun_group = Sun_sumry2.groupby(["cluster"]).mean()


b = pd.DataFrame({"ind":range(0,6)})
c = pd.DataFrame({"ind":range(0,6)})
ncount = Sun_sumry2.groupby("cluster").count().iloc[:,1]
for i in range(5,47):
    data = Sun_sumry2.ix[(Sun_sumry2.iloc[:,i]>0),[i,42+i,7658]]
    data1 = Sun_sumry2.ix[(Sun_sumry2.iloc[:,i]>0),[i,7658]]
    
    a = data.groupby(["cluster"]).mean()
    b = pd.concat([a,b],axis=1)

    a1 = data1.groupby(["cluster"]).count().divide(ncount,axis=0)
    c = pd.concat([a1,c],axis=1)

c1 = c.fillna(0)
b1 = b.fillna(0)

xy1 = pd.read_csv("xy.csv")
xy = xy1.sort_values(by=["X"])

def create_figure1():
    k = int(Cluster.value)
    
    p = figure(y_range=(8,18),plot_width=300,plot_height = 400)
    p.circle(xy["X"], b1.iloc[k,range(1,84,2)]/60,size=b1.iloc[k,range(0,84,2)]*5,color="blue")
    p.line(xy["X"], b1.iloc[k,range(1,84,2)]/60)
    p.xaxis.axis_label = 'X-coordinate'
    p.yaxis.axis_label = 'avg_checkin_time'


    p11 = figure(plot_width=300,plot_height = 400)
    p11.circle(xy["X"], c1.iloc[k,:],size=c1.iloc[k,:]*5)
    p11.line(xy["X"], c1.iloc[k,:])
    p11.xaxis.axis_label = 'X-coordinate'
    p11.yaxis.axis_label = 'avg_checkin_percentage'


    

    p0 = figure(plot_width=350,plot_height = 400,title='2D cluster visualization')
    p0.circle(X[:,0], X[:,1],color=cluster_color)


    p1 = figure(plot_width=350,plot_height = 400,title="Duration in the park")
    p1.vbar(x=range(0,6), top=Sun_group["time_dur"]/60,width=0.5,color=color)

    p2 = figure(plot_width=350,plot_height = 400,title="Number of check-in")
    p2.vbar(x=range(0,6), top=Sun_group["count_checkin"],width=0.5,color=color)

    p3 = figure(plot_width=350,plot_height = 400,title="Number of movement records")
    p3.vbar(x=range(0,6), top=Sun_group["count_move"],width=0.5,color=color)



    return (column(row(p0,p,p11),row(p1,p2,p3)))

def create_figure2():
    k = int(Cluster.value)
    
    p = figure(plot_width=300,plot_height = 400)
    p.circle(xy["X"], c1.iloc[k,:],size=c1.iloc[k,:]*5)
    p.line(xy["X"], c1.iloc[k,:])
    p.xaxis.axis_label = 'X-coordinate'
    p.yaxis.axis_label = 'avg_checkin_percentage'
    
    return (p)

def update(attr, old, new):
    layout.children[1] = create_figure1()



clusters = ["0","1","2","3","4","5"]
Cluster = Select(title='Cluster', value='0', options=clusters)
Cluster.on_change('value', update)

controls = widgetbox([Cluster], width=100)

#row1 = row(p0)
row2 = row(controls, create_figure1())
#row2 = row(p1,p2,p3)

layout = row(controls, create_figure1()) #row2#column(row2)

curdoc().add_root(layout)
curdoc().title = 'Compare groups by check-in Statistics of 42 locations'



