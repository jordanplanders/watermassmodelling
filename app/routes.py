from flask import render_template, flash, redirect, url_for, request
from app import app, db#, cache
from app.forms import LoginForm, PlotForm, RegistrationForm
from flask_login import logout_user, login_user, login_required, current_user
from app.models import User
# from flask_googlemaps import GoogleMaps, Map
from werkzeug.urls import url_parse

from chem_ocean import Build_Map as bm
from chem_ocean import Plot_Raw as pr
from chem_ocean import plan_modelcomp as md
import io
import base64



@app.route('/')
@app.route('/index')
@login_required
def index():
	# return 'Hello, World!'
	user = {'username': current_user.username}
	posts = [
        {
            'tracer': {'tracername': 'Nitrate'},
            'body': 'figure1'
        },
        {
            'tracer': {'tracername': 'Phosphate'},
            'body': 'figure2'
        }
    ]
	# used to return html string with string formatting for username, replaced to render_template with dynamic values
	return render_template('index.html', title = 'Home', user = user, posts = posts)

@app.route('/plots', methods=['GET', 'POST'])
def plots():
    # print('hello from form world'+ form.tracer.data)
    Tracers = {
        'salinity' : {
            'tracer': {'tracername': 'Salinity'},
            'body': 'figure1',
            'img': 'https://github.com/jpl86/ODVx/blob/master/NSsection-15_raw_S.png?raw=true'
        },

        'phosphate' :{
             'tracer': {'tracername': 'Phosphate'},
             'body': 'figure2',
             'img': 'https://github.com/jpl86/ODVx/blob/master/NSsection-15_raw_P.png?raw=true'
        },

        'oxygen' :{
            'tracer': {'tracername': 'Oxygen'},
            'body': 'figure3',
            'img': 'https://github.com/jpl86/ODVx/blob/master/NSsection_15_raw_O.png?raw=true'
        }
        }

    if request.method == 'POST':
        form = request.form
        # tracers = request.form.getlist('tracer')
        # tracers = form.getlist('tracer')
        # flash('Plot requested for tracer {}, plot_type={}'.format(tracers[1], form['plot_type']))
    
    posts = []
    print(form.getlist('tracer'))
    tracers = form.getlist('tracer')
    for tracer in form.getlist('tracer'):
        flash('Plot requested for tracer {}'.format(tracer))
        posts.append(tracer)
    print('got stuck')
    # show map
    # img = io.BytesIO()
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(13, 7), facecolor='w')
    # _map, map_fig, ax = bm.build_map('y', 'merc', float(form['lat_s']), float(form['lat_n']), float(form['lon_w']), float(form['lon_e']), 'c', fig, ax1, 111, 'lrbt')
    # map_fig.savefig(img, format='png')
    # img.seek(0)
    # # print(img.seek(0))
    # plot_url = base64.b64encode(img.getvalue()).decode('ascii')
    # print(plot_url)
    #show raw tracer data
    print('made it over the hump')
    img2 = io.BytesIO()
    print(form['plot_type'])
    if form['plot_type'] == "plan":
        _depth= form['depth']
        lonLine =-15
    else:
        _depth=1500
        lonLine =-15
        # _latLimits = (float(form['lat_s']), float(form['lat_n']))
    print(form['tracer'])
    _x, _y, _feat_data, _basemap, _xLab, _yLab, _latLon_params = pr.getRaw(float(form['lat_s']), float(form['lat_n']), float(form['lon_w']), float(form['lon_e']), tracers, form['plot_type'], depth =  _depth)#, lonTraj = (float(form['lon_w']), float(form['lon_e'])), latLimits = (float(form['lat_s']), float(form['lat_n'])))

    if form['raw_model'] == 'raw':
        raw_data_fig = pr.plotRaw(float(form['lat_s']), float(form['lat_n']), float(form['lon_w']), float(form['lon_e']), [form['tracer']], form['plot_type'], depth =  _depth)#, lonTraj = (float(form['lon_w']), float(form['lon_e'])), latLimits = (float(form['lat_s']), float(form['lat_n'])))
        raw_data_fig.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode('ascii')
    elif form['raw_model'] == 'model':
        print(form['model_opts'])
        # model_data_fig = test_clustering2(x, y, feat_data, xLab, yLab, N_CLUSTERS, latLon_params, **kwargs):
        model_data = md.test_clustering3(_x, _y, _feat_data, _xLab, _yLab, 6, _latLon_params, _basemap, models = form.getlist('model_opts'))
        print(model_data.keys())
        pred_dict, model_fig = md.plot_model_output(_x, _y, _xLab, _yLab, float(form['lat_s']), float(form['lat_n']), float(form['lon_w']), float(form['lon_e']), _latLon_params, model_data, form['plot_type'], sil = form['sil_coef'])
        print('pred_dict and model_fig exist')
        model_fig.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode('ascii')

    return render_template('plots.html', title = 'Plots', posts = posts, form= form, plot_url2=plot_url2)#, plot_url2=plot_url2)

@app.route('/gallery')
def gallery():
	user = {'username': 'Jordan'}
	posts = [
        {
            'tracer': {'tracername': 'Salinity'},
            'body': 'figure1',
            'img': 'https://github.com/jpl86/ODVx/blob/master/NSsection-15_raw_S.png?raw=true'
        },
        {
            'tracer': {'tracername': 'Phosphate'},
            'body': 'figure2',
            'img': 'https://github.com/jpl86/ODVx/blob/master/NSsection-15_raw_P.png?raw=true'
        }
    ]
	return render_template('gallery.html', title = 'Gallery', user = user, posts = posts)



@app.route('/login', methods=['GET', 'POST'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('index'))
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(username=form.username.data).first()
		if user is None or not user.check_password(form.password.data):
			flash('Invalid username or password')
			return redirect(url_for('login'))
		login_user(user, remember=form.remember_me.data)
		next_page = request.args.get('next')
		if not next_page or url_parse(next_page).netloc != '':
			next_page = url_for('index')
		return redirect(next_page)
        # flash('Login requested for user {}, remember_me={}'.format(
        #     form.username.data, form.remember_me.data))
		return redirect('/index')
	return render_template('login.html', title='Sign In', form=form)



@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)



@app.route('/plotform', methods=['GET', 'POST'])
def plotform():
    form = PlotForm()
    if form.validate_on_submit():
        # flash('Plot requested for tracer {}, plot_type={}'.format(
        #     form.tracer.data, form.plot_type.data))
        # return render_template('plots.html', title = "Plots", form=form)
        # _map, map_fig = build_map('y', 'merc', float(form.lat_s.data), float(form.lat_n.data), float(form.lon_w.data), float(form.lon_e.data), 'c')
        # basemp = Plot(basemap=map_fig, user_id=current_user.id)
        # db.session.add(basemp)
        # db.session.commit()
        return redirect('/plots')
        # instead, let's try returning render_template with 
    return render_template('plotform.html', title='Plot Form', form=form)



@app.route("/mapview")
def mapview():
    # creating a map in the view
    mymap = Map(
        identifier="view-side",
        lat=37.4419,
        lng=-122.1419,
        markers=[(37.4419, -122.1419)]
    )
    sndmap = Map(
        identifier="sndmap",
        lat=37.4419,
        lng=-122.1419,
        markers=[
          {
             'icon': 'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
             'lat': 37.4419,
             'lng': -122.1419,
             'infobox': "<b>Hello World</b>"
          },
          {
             'icon': 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
             'lat': 37.4300,
             'lng': -122.1400,
             'infobox': "<b>Hello World from other place</b>"
          }
        ]
    )
    return render_template('mapview.html', mymap=mymap, sndmap=sndmap)


import pandas as pd 
from itertools import islice
import re

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import sys, pickle, time

from chem_ocean.oc_db_fxns import build_oc_station_db, make_arrays, min_max_print, oc_data_2df
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import cluster, datasets, metrics
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from chem_ocean.oc_clustering_fxns import build_cluster_data, test_clustering

from numpy.random import uniform, seed
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import shiftgrid
from numpy import linspace
from numpy import meshgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt


title_sz = 27
axis_sz = 22
tick_sz = 21

import shapefile
from matplotlib import cm, rcParams
from mpl_toolkits.basemap import Basemap
rcParams.update({'font.size': 11}) # Increase font-size
import time

# @cache.cached(timeout=50, key_prefix='custom_map')
# def build_map(show, proj, minLat, maxLat, minLon, maxLon, res, fig, ax, pos_num, labels_in):
#     t0 = time.time()
#     if proj in ['cyl', 'merc', 'mill', 'cea', 'gall', 'lcc']:
#         _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
#             resolution = res, area_thresh = 1,
#             llcrnrlon=(minLon)*1, llcrnrlat=(minLat)*1,
#             urcrnrlon=(maxLon)*1, urcrnrlat=(maxLat)*1)
#     if proj in ['stere']:
#         _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
#             resolution = res, area_thresh = 1,
#             llcrnrlon=(minLon)-30, llcrnrlat=(minLat)*1,
#             urcrnrlon=(maxLon)*1, urcrnrlat=(maxLat)*1)
#     if proj in ['ortho', 'geos', 'nsper']:
#         _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
#             resolution = res, area_thresh = 1,
#             llcrnry=minLat*1,urcrnry=maxLat*1)#, llcrnrx=minLon*1, urcrnrx=maxLon*1, )
#     else: 
#         _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
#             resolution = res, area_thresh = 1, llcrnrlon=(minLon)*1, llcrnrlat=(minLat)*1,
#             urcrnrlon=(maxLon)*1, urcrnrlat=(maxLat)*1, rsphere=(6378137.00,6356752.3142))

#     t1 = time.time()
# #     print(1, t1-t0)
#     if show == 'y':
#         t0 = time.time()
#         _map.ax = ax
#         t1 = time.time()
#         _map.drawcoastlines(color='k')
#         _map.drawcountries()
#         _map.fillcontinents(lake_color='b',color = 'gray')
#         _map.drawmapboundary(linewidth=2)
        
#         # labels = [left,right,top,bottom]
#         lbls = []
#         for label in ['l','r','t','b']:
#             if label in labels_in:
#                 lbls.append(1)
#             else:
#                 lbls.append(0)
            
#         _map.drawmeridians(np.arange(0, 360, 30), labels=lbls)
#         _map.drawparallels(np.arange(-90, 90, 30), labels=lbls)    
#         t2 = time.time()
#         print(t2-t1)
# #     plt.savefig('/static/temp_map.png', dpi=200)
#     return _map, fig, ax


#     