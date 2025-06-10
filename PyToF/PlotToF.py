########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import numpy as np
import math
import scipy

import os

import matplotlib.pyplot as plt
from matplotlib import colormaps

from PyToF.color import c

def default_opts():

    """
    This function implements the standard parameters used for plotting,
    except for the kwargs given by the user.
    """

    opts = {}

    opts['new_figure'] = True

    opts['xmargin'] = 0.01
    opts['ymargin'] = 0.01

    opts['lw']      = 1
    opts['ls']      = '-'
    opts['color']   = 'C0'
    opts['label']   = ''

    opts['plot_literature'] = False
    opts['literature']      = None

    opts['extra_plot']  = False
    opts['extra_coord'] = [0.7, 0.7, 0.25, 0.25] # [left, bottom, width, height]
    opts['e_x']         = 2
    opts['e_y']         = 3
    opts['extra_lim']   = None  

    opts['legend']              = False
    opts['legend_loc']          = 'best'
    opts['legend_ncol']         = 1
    opts['legend_fontsize']     = 14
    opts['legend_frame_alpha']  = 0.8

    opts['dpi']         = 100
    opts['fontsize']    = 16
    
    opts['save']        = False
    opts['path_name']   = os.getcwd()
    opts['fig_name']    = 'figure'
    opts['format']      = 'png'
    opts['transparent'] = False

    opts['contourf_levels']     = 50
    opts['contourf_cmap']       = 'viridis'

    opts['contour_levels']      = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    opts['contour_colors']      = 'white'
    opts['contour_fontsize']    = 12

    opts['colorbar_ticks']      = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    opts['shape_circle']        = False
    opts['shape_circle_lw']     = 2
    opts['shape_circle_ls']     = '-'
    opts['shape_circle_color']  = 'grey'
    opts['shape_circle_label']  = r'average surface radius'

    opts['rho_maxs']        = None
    opts['sm']              = None 
    opts['state_cmap']      = 'viridis'
    opts['wrong_color']     = (0.0, 0.0, 0.0, 0.1)
    opts['len_color_bar']   = 0.04
    opts['uni_color']       = False
    opts['bins']            = None
    opts['Js_data']         = False

    return opts

def plot_xy(class_obj, x, y, **kwargs):

    opts = {**default_opts(), **kwargs}

    shell_m     = 4*np.pi*(class_obj.li**3-np.append(class_obj.li[1:],0)**3)/3 * class_obj.rhoi
    summed_m    = np.array([sum(shell_m[i:]) for i in range(len(shell_m))])

    xy = [  class_obj.li/class_obj.li[0]/1,
            summed_m/summed_m[0]/1,
            class_obj.rhoi/1000,
            class_obj.Pi/1e5
            ]

    if opts['plot_literature']:

            literature_xys = []

            for i in range(len(opts['literature'].labels)):

                li      = opts['literature'].lis[i]
                p       = np.argsort(li)[::-1]

                shell_m     = 4*np.pi*(li[p]**3-np.append(li[p][1:],0)**3)/3*opts['literature'].rhois[i][p]
                summed_m    = np.array([sum(shell_m[i:]) for i in range(len(shell_m))])

                literature_xys.append([ li[p]/li[p][0],
                                        summed_m/summed_m[0],
                                        opts['literature'].rhois[i][p]/1000,
                                        opts['literature'].Pis[i][p]/1e5])

    label_xy = [r'average $r/R$',
                r'enclosed $m/M$',
                r'$\rho$ [g/cm$^3$]',
                r'$P$ [bar]'
                ]

    scale_xy = ['linear',
                'linear',
                'linear',
                'log']

    if opts['new_figure']:

        plt.rcParams['axes.xmargin'] = opts['xmargin']
        plt.rcParams['axes.ymargin'] = opts['ymargin']
        
        class_obj.fig  = plt.figure(layout='constrained', dpi=opts['dpi'])
        class_obj.ax   = class_obj.fig.gca()

        class_obj.ax.set_xlabel(label_xy[x], fontsize=opts['fontsize'])
        class_obj.ax.set_ylabel(label_xy[y], fontsize=opts['fontsize'])
        class_obj.ax.set_xscale(scale_xy[x])
        class_obj.ax.set_yscale(scale_xy[y])
        class_obj.ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

        if opts['extra_plot']:

            class_obj.e_ax = class_obj.fig.add_axes(opts['extra_coord'])  

            class_obj.e_ax.set_xlabel(label_xy[opts['e_x']], fontsize=opts['fontsize'])
            class_obj.e_ax.set_ylabel(label_xy[opts['e_y']], fontsize=opts['fontsize'])
            class_obj.e_ax.set_xscale(scale_xy[opts['e_x']])
            class_obj.e_ax.set_yscale(scale_xy[opts['e_y']])
            class_obj.e_ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

            if opts['extra_lim'] is not None:

                class_obj.e_ax.set_xlim(opts['extra_lim'][0])
                class_obj.e_ax.set_ylim(opts['extra_lim'][1])

    class_obj.ax.plot(xy[x], xy[y], lw=opts['lw'], ls=opts['ls'], color=opts['color'], label=opts['label'])

    if opts['extra_plot']:

        class_obj.e_ax.plot(xy[opts['e_x']], xy[opts['e_y']], lw=opts['lw'], ls=opts['ls'], color=opts['color'])

    if opts['plot_literature']:

        for i in range(len(literature_xys)):

            class_obj.ax.plot(literature_xys[i][x], literature_xys[i][y], lw=opts['literature'].lws[i], ls=opts['literature'].lss[i], color=opts['literature'].colors[i], label=opts['literature'].labels[i])
            
            if opts['extra_plot']:

                class_obj.e_ax.plot(literature_xys[i][opts['e_x']], literature_xys[i][opts['e_y']], lw=opts['literature'].lws[i], ls=opts['literature'].lss[i], color=opts['literature'].colors[i])
    
    if opts['legend']:

        class_obj.ax.legend(ncol=opts['legend_ncol'], fontsize=opts['legend_fontsize'], loc=opts['legend_loc'], framealpha=opts['legend_frame_alpha'])

    if opts['sm'] is not None:

        axc         = class_obj.fig.add_axes([1, class_obj.ax.get_position().y0, opts['len_color_bar'], 1-class_obj.ax.get_position().y0])
        colorbar    = class_obj.fig.colorbar(opts['sm'], orientation='vertical', cax=axc)
        colorbar.set_label(r'Core density $\rho_{c}$ [g/cm$^{3}$]', fontsize=opts['fontsize'])
        colorbar.ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

    if opts['save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], dpi=opts['dpi'], format=opts['format'], transparent=opts['transparent'], bbox_inches='tight')

def plot_shape(class_obj, **kwargs):

    opts = {**default_opts(), **kwargs}

    plt.rcParams['axes.xmargin'] = opts['xmargin']
    plt.rcParams['axes.ymargin'] = opts['ymargin']

    ###
    
    theta   = np.linspace(0, 2*np.pi, 1000)
    r_l_mu  = class_obj.get_r_l_mu(np.cos(theta))
    N       = np.shape(r_l_mu)[0]

    X       = np.zeros((N+1,1000))
    X[:N,:] = r_l_mu/np.max(class_obj.li) #radius 0 in the core

    Y       = np.ones((N+1,1000))*theta/(2*np.pi)*360
    
    Z       = np.ones((N+1,1000))
    Z[:N,:] = np.outer(class_obj.rhoi/np.max(class_obj.rhoi),np.ones_like(theta))

    ###

    class_obj.fig  = plt.figure(layout='constrained', dpi=opts['dpi'])
    class_obj.ax   = class_obj.fig.add_subplot()

    con         = class_obj.ax.contourf(X, Y, Z, levels=opts['contourf_levels'], cmap=opts['contourf_cmap'])

    con_line    = class_obj.ax.contour( X, Y, Z, levels=opts['contour_levels'], colors=opts['contour_colors'])
    class_obj.ax.clabel(con_line, inline=True, fontsize=opts['contour_fontsize'])

    con_bar = class_obj.fig.colorbar(con, ticks=opts['colorbar_ticks'])
    con_bar.set_label(r'$\rho$ [$\cdot$' + str(np.round(np.max(class_obj.rhoi/1000),2)) + ' g/cm$^3$]', fontsize=opts['fontsize'])

    con_bar.ax.tick_params(  axis='both', which='major', labelsize=opts['fontsize'])
    class_obj.ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

    class_obj.ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360], labels=['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE', 'N'])
    class_obj.ax.set_xlabel(r'spheroid shell radius / average surface radius', fontsize=opts['fontsize'])

    if opts['save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], dpi=opts['dpi'], format=opts['format'], transparent=opts['transparent'], bbox_inches='tight')

    ###

    class_obj.fig  = plt.figure(layout='constrained', dpi=opts['dpi'])
    class_obj.ax   = class_obj.fig.add_subplot(projection='polar')

    con         = class_obj.ax.contourf(Y*(2*np.pi)/360, X, Z, levels=opts['contourf_levels'], cmap=opts['contourf_cmap'])

    con_bar     = class_obj.fig.colorbar(con, ticks=opts['colorbar_ticks'])
    con_bar.set_label(r'$\rho$ [$\cdot$' + str(np.round(np.max(class_obj.rhoi/1000),2)) + ' g/cm$^3$]', fontsize=opts['fontsize'])

    con_bar.ax.tick_params(  axis='both', which='major', labelsize=opts['fontsize'])
    class_obj.ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

    class_obj.ax.set_theta_zero_location("N")
    class_obj.ax.set_thetagrids([], labels=[])
    class_obj.ax.set_rgrids([], labels=[])
    class_obj.ax.spines['polar'].set_visible(False)
    
    if opts['shape_circle']:

        class_obj.ax.plot(theta, np.ones_like(theta), lw=opts['shape_circle_lw'], ls=opts['shape_circle_ls'], color=opts['shape_circle_color'], label=opts['shape_circle_label'])
        
    if opts['legend']:

        class_obj.ax.legend(ncol=opts['legend_ncol'], fontsize=opts['legend_fontsize'], loc=opts['legend_loc'], framealpha=opts['legend_frame_alpha'])

    if opts['save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], dpi=opts['dpi'], format=opts['format'], transparent=opts['transparent'], bbox_inches='tight')

def plot_ss(class_obj, **kwargs):

    opts = {**default_opts(), **kwargs}

    plt.rcParams['axes.xmargin'] = opts['xmargin']
    plt.rcParams['axes.ymargin'] = opts['ymargin']

    class_obj.fig  = plt.figure(layout='constrained', dpi=opts['dpi'])
    class_obj.ax   = class_obj.fig.add_subplot()

    ss = class_obj.ss

    for i in range(len(ss)):

        if i != 0:   

            x = np.append(class_obj.li/class_obj.li[0], 0)
            y = np.append(np.flip(ss[i]/np.max(abs(ss[i]))), ss[i][0]/np.max(abs(ss[i]))) 

            class_obj.ax.plot(x, y, lw=opts['lw'], ls=opts['ls'], label='{:.2e}'.format(np.max(abs(ss[i])))+r'$\cdot s_{'+str(2*i)+r'}$')
    
    class_obj.ax.set_xlabel(r'average $r/R$', fontsize=opts['fontsize'])
    class_obj.ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

    class_obj.ax.legend(ncol=opts['legend_ncol'], fontsize=opts['legend_fontsize'], loc=opts['legend_loc'], framealpha=opts['legend_frame_alpha'], bbox_to_anchor=(1, 1), frameon=False)

    if opts['save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], dpi=opts['dpi'], format=opts['format'], transparent=opts['transparent'], bbox_inches='tight')

def plot_state_xy(class_obj, x, y, state, what_model, **kwargs):

    opts = {**default_opts(), **kwargs}

    if opts['rho_maxs'] is not None:

        idxs    = np.argsort(opts['rho_maxs'])
        cmap    = colormaps.get_cmap(opts['state_cmap'])
        colors  = cmap((opts['rho_maxs']-min(opts['rho_maxs'])) / (max(opts['rho_maxs']) - min(opts['rho_maxs'])))[idxs]

    else:

        idxs   = range(len(state[:,0]))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*(len(idxs)//10+1)

    j = 0

    save = opts['save']; plot_literature = opts['plot_literature']; legend = opts['legend']

    for i in idxs:

        if what_model == 'baro':

            costJs = class_obj.baro_cost_function(state[i,:], return_sum=False)

        elif what_model == 'dens':

            costJs = class_obj.dens_cost_function(state[i,:], return_sum=False)
            
        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_model! Use \'baro\' or \'dens\'.' + c.ENDC)

        if (np.abs(costJs)<1e0).all():

            color = colors[j]
            j += 1
            
        else:

            color = opts['wrong_color']

            if opts['rho_maxs'] is not None:

                j += 1

        if i == idxs[0] and len(idxs) == 1:

            opts['color'] = color; opts['save'] = save; opts['new_figure'] = True; opts['plot_literature'] = plot_literature; opts['legend'] = legend

        elif i == idxs[0]:

            opts['color'] = color; opts['save'] = False; opts['new_figure'] = True; opts['plot_literature'] = False; opts['legend'] = False

        elif i != idxs[-1]:

            opts['color'] = color; opts['save'] = False; opts['new_figure'] = False; opts['plot_literature'] = False; opts['legend'] = False

        else:

            if opts['rho_maxs'] is not None:

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(opts['rho_maxs'])/1000, vmax=max(opts['rho_maxs'])/1000))
                sm.set_array(opts['rho_maxs'])
                opts['sm'] = sm

            opts['color'] = color; opts['save'] = save; opts['new_figure'] = False; opts['plot_literature'] = plot_literature; opts['legend'] = legend
        
        plot_xy(class_obj, x, y, **opts)

def plot_state_corr_xy(class_obj, x, y, state, what_model, **kwargs):

    opts = {**default_opts(), **kwargs}

    Js_list = []
    color_list = []

    if opts['rho_maxs'] is not None:

        idxs    = np.argsort(opts['rho_maxs'])
        cmap    = colormaps.get_cmap(opts['state_cmap'])
        colors  = cmap((opts['rho_maxs']-min(opts['rho_maxs'])) / (max(opts['rho_maxs']) - min(opts['rho_maxs'])))[idxs]

    else:

        idxs   = range(len(state[:,0]))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*(len(idxs)//10+1)

        if opts['uni_color']:

            colors = ['#1f77b4']*len(idxs)
    
    j = 0

    for i in idxs:

        if what_model == 'baro':

            costJs = class_obj.baro_cost_function(state[i,:], return_sum=False)

        elif what_model == 'dens':

            costJs = class_obj.dens_cost_function(state[i,:], return_sum=False)

        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_model! Use \'baro\' or \'dens\'.' + c.ENDC)

        if (np.abs(costJs)<1e0).all():

            color = colors[j]
            j += 1
            
        else:

            color = opts['wrong_color']

            if opts['rho_maxs'] is not None:

                j += 1

        Js_list.append(class_obj.Js)
        color_list.append(color)
        
    Js_list = np.array(Js_list)
    
    plt.rcParams['axes.xmargin'] = opts['xmargin']
    plt.rcParams['axes.ymargin'] = opts['ymargin']
    
    fig = plt.figure(figsize=(6, 6), layout='constrained', dpi=opts['dpi'])
    gs = fig.add_gridspec(2, 2,   width_ratios=(4, 1.25), height_ratios=(1.25, 4),
                                  wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if x >= len(Js_list[0,:]):

        x_label = r'$b_{' + str(x-len(Js_list[0,:])) + r'}$'
        x_array = state[:,x-len(Js_list[0,:])]
        
    else:

        x_label = r'$J_{' + str(2*x) + r'}$'
        x_array = Js_list[:,x]

    if y >= len(Js_list[0,:]):

        y_label = r'$b_{' + str(y-len(Js_list[0,:])) + r'}$'
        y_array = state[:,y-len(Js_list[0,:])]
        
    else:

        y_label = r'$J_{' + str(2*y) + r'}$'
        y_array = Js_list[:,y]

    x_scale = math.floor(math.log(np.max(abs(x_array)), 10))+1
    y_scale = math.floor(math.log(np.max(abs(y_array)), 10))+1

    if opts['Js_data'] and (x < len(Js_list[0,:]) and y < len(Js_list[0,:])):

        ax.errorbar(class_obj.opts['Target_Js'][x-1]/10**x_scale, class_obj.opts['Target_Js'][y-1]/10**y_scale, xerr = class_obj.opts['Sigma_Js'][x-1]/10**x_scale, yerr = class_obj.opts['Sigma_Js'][y-1]/10**y_scale, color='k', capsize=3)
    
    x_array /= 10**x_scale
    y_array /= 10**y_scale

    x_label += r' [$\cdot 10^{'+str(-1*x_scale)+'}$]'
    y_label += r' [$\cdot 10^{'+str(-1*y_scale)+'}$]'
        
    ax.scatter(x_array, y_array, color=color_list)

    ax_histx.hist(x_array, bins=opts['bins'], color='black')
    ax_histy.hist(y_array, bins=opts['bins'], color='black', orientation='horizontal')

    ax.set_xlabel(x_label, fontsize=opts['fontsize'])
    ax.set_ylabel(y_label, fontsize=opts['fontsize'])

    ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])
    ax_histx.tick_params(axis='both', which='major', labelsize=opts['fontsize'])
    ax_histy.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

    if opts['rho_maxs'] is not None:

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(opts['rho_maxs'])/1000, vmax=max(opts['rho_maxs'])/1000))
        sm.set_array(opts['rho_maxs'])
        axc      = ax.inset_axes([0.1, 0.2, 0.8, opts['len_color_bar']])
        colorbar = fig.colorbar(sm, orientation='horizontal', cax=axc)
        colorbar.set_label(r'Core density $\rho_{c}$ [g/cm$^{3}$]', fontsize=opts['fontsize'])
        colorbar.ax.tick_params(axis='both', which='major', labelsize=opts['fontsize'])

    if opts['legend']:

        p1, = ax.plot(np.nan, np.nan)
        p2, = ax.plot(np.nan, np.nan)
        ax.legend([p1,p2], [r"$\rho=${:.2f}".format(scipy.stats.spearmanr(x_array, y_array, nan_policy='omit')[0]), r"$p=${:.2f}".format(scipy.stats.spearmanr(x_array, y_array, nan_policy='omit')[1])], 
                  handletextpad=0, handlelength=0, ncol=opts['legend_ncol'], fontsize=opts['legend_fontsize'], loc=opts['legend_loc'], framealpha=opts['legend_frame_alpha'])

        ax_histx.legend([p1,p2], [r"$\mu=${:.2f}".format(np.average(x_array)), r"$\sigma=${:.2f}".format(np.std(x_array))], 
                  handletextpad=0, handlelength=0, ncol=opts['legend_ncol'], fontsize=opts['legend_fontsize'], loc=opts['legend_loc'], framealpha=opts['legend_frame_alpha'])

        ax_histy.legend([p1,p2], [r"$\mu=${:.2f}".format(np.average(y_array)), r"$\sigma=${:.2f}".format(np.std(y_array))], 
                  handletextpad=0, handlelength=0, ncol=opts['legend_ncol'], fontsize=opts['legend_fontsize'], loc=opts['legend_loc'], framealpha=opts['legend_frame_alpha'])

    if opts['save']:

        fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], dpi=opts['dpi'], format=opts['format'], transparent=opts['transparent'], bbox_inches='tight')

def plot_autocorr(autocorr, **kwargs):

    opts = {**default_opts(), **kwargs}

    N = len(autocorr)
    max_length = len(autocorr[0]) + 1

    fig  = plt.figure(layout='constrained')
    ax   = fig.add_subplot()

    for i in range(N):

        n = 100 * np.arange(1, len(autocorr[i]) + 1)
        y = autocorr[i]
        ax.plot(n, y, 'o-')

        if len(autocorr[i]) + 1 > max_length:
            max_length = len(autocorr[i]) + 1

    n = 100 * np.arange(1, max_length)
    ax.plot(n, n / 100.0, "--k")

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel("number of steps", fontsize=16)
    ax.set_ylabel(r"mean $\hat{\tau}$", fontsize=16)
    
    fig.savefig(opts['path_name'] + 'autocorr' + opts['fig_name'] + '.' + opts['format'], dpi=opts['dpi'], format=opts['format'], transparent=opts['transparent'], bbox_inches='tight')
