########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import numpy as np
import math
import scipy

import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from PyToF.color import c

def _default_mpl_opts():

    """
    This function customises standard rcParams options for plotting.
    """

    opts = {}

    #Lines and errobar: 
    opts['lines.linewidth']             = 3.0
    opts['lines.markersize']            = 9
    opts['errorbar.capsize']            = 0.0

    #Axes and margins:
    opts['axes.labelsize']              = 18
    opts['axes.xmargin']                = 0.0
    opts['axes.ymargin']                = 0.0
    opts['axes.formatter.useoffset']    = False

    #Ticks:
    opts['xtick.labelsize']             = 16
    opts['ytick.labelsize']             = 16
    opts['xtick.direction']             = 'in'
    opts['ytick.direction']             = 'in'
    opts['xtick.top']                   = True
    opts['ytick.right']                 = True
    opts['xtick.minor.visible']         = True
    opts['ytick.minor.visible']         = True
    opts['xtick.major.size']            = 7.0
    opts['ytick.major.size']            = 7.0
    opts['xtick.minor.size']            = 4.0
    opts['ytick.minor.size']            = 4.0
    opts['xtick.major.width']           = 1.6
    opts['ytick.major.width']           = 1.6
    opts['xtick.major.pad']             = 7.0
    opts['ytick.major.pad']             = 7.0

    #Legend:
    opts['legend.title_fontsize']       = 16
    opts['legend.fontsize']             = 16
    
    #Figure properties:
    opts['figure.figsize']              = [6.4, 4.8]
    opts['figure.dpi']                  = 100
    opts['font.family']                 = 'Ubuntu'

    return opts

def _apply_mpl_opts(opts):

    """
    This function applies _default_mpl_opts().
    """

    for kwd, value in opts.items():

        try:

            plt.rcParams[kwd] = value

        except:

            continue

def _default_plot_xy_opts():

    """
    This function implements standard paramters used for the function plot_xy().
    """

    opts = {}

    opts['color']       = 'C0'
    opts['linestyle']   = '-'
    opts['label']       = ''
    opts['zorder']      = 0

    opts['do_legend']               = False
    opts['loc_legend']              = 'best'
    opts['ncol_legend']             = 1
    opts['frameon_legend']          = True
    opts['bbox_to_anchor_legend']   = (0.0, 0.0, 1.0, 1.0)

    opts['do_save']     = False
    opts['path_name']   = os.getcwd()
    opts['fig_name']    = 'figure'
    opts['format']      = 'png'
    opts['transparent'] = False

    return opts

def _default_plot_shape_opts():

    """
    This function implements standard paramters used for the function plot_shape().
    """

    opts = {}

    opts['contourf_levels']  = 50
    opts['contourf_cmap']    = 'viridis'
    opts['contour_levels']   = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    opts['contour_colors']   = 'white'
    opts['contour_fontsize'] = 15
    opts['colorbar_ticks']   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    opts['do_save']     = False
    opts['path_name']   = os.getcwd()
    opts['fig_name']    = 'figure'
    opts['format']      = 'png'
    opts['transparent'] = False

    return opts

def _default_plot_ss():

    """
    This function implements standard paramters used for the function plot_ss().
    """

    opts = {}

    opts['do_legend']               = True
    opts['loc_legend']              = 'best'
    opts['ncol_legend']             = 1
    opts['frameon_legend']          = False
    opts['bbox_to_anchor_legend']   = (1.0, 1.0)

    opts['do_save']     = False
    opts['path_name']   = os.getcwd()
    opts['fig_name']    = 'figure'
    opts['format']      = 'png'
    opts['transparent'] = False

    return opts

def _default_plot_autocorr():

    """
    This function implements standard paramters used for the function plot_autocorr().
    """

    opts = {}

    opts['do_legend']               = False
    opts['loc_legend']              = 'best'
    opts['ncol_legend']             = 1
    opts['frameon_legend']          = True
    opts['bbox_to_anchor_legend']   = (0.0, 0.0, 1.0, 1.0)

    opts['do_save']     = False
    opts['path_name']   = os.getcwd()
    opts['fig_name']    = 'figure'
    opts['format']      = 'png'
    opts['transparent'] = False

    return opts

def plot_xy(class_obj, x, y, do_new_figure=True, literature=None, inset_plot_xy=None, colorbar=None, **kwargs):

    #Apply default and user supplied options:
    mpl_opts    = {**_default_mpl_opts()    , **kwargs}; _apply_mpl_opts(mpl_opts)
    opts        = {**_default_plot_xy_opts(), **kwargs}

    #Calculate mass of each layer and enclosed mass:
    shell_m     = 4/3*np.pi * (class_obj.li**3 - np.append(class_obj.li[1:],0)**3) * class_obj.rhoi
    summed_m    = np.array([sum(shell_m[i:]) for i in range(len(shell_m))])

    #Data for plotting:
    xy          = [class_obj.li/class_obj.li[0], summed_m/summed_m[0], class_obj.rhoi/1000 , class_obj.Pi/1e5]
    label_xy    = [r'average $r/R$'            , r'enclosed $m/M$'   , r'$\rho$ [g/cm$^3$]', r'$P$ [bar]'    ]
    scale_xy    = ['linear'                    , 'linear'            , 'linear'            , 'log'           ]

    #Append radius zero entry for completeness:
    xy[0] = np.append(xy[0], 0); xy[1] = np.append(xy[1], 0); xy[2] = np.append(xy[2], xy[2][-1]); xy[3] = np.append(xy[3], xy[3][-1])

    #Additional data for plotting supplied by the user:
    if literature is not None:

        literature_xys = []

        for i in range(len(literature.lis)):

            #Get radii and sort them:
            li = literature.lis[i]; p = np.argsort(li)[::-1]

            #Calculate mass of each layer and enclosed mass:
            shell_m     = 4/3*np.pi * (li[p]**3 - np.append(li[p][1:],0)**3) * literature.rhois[i][p]
            summed_m    = np.array([sum(shell_m[i:]) for i in range(len(shell_m))])

            #Data for plotting:
            literature_xys.append([li[p]/li[p][0], summed_m/summed_m[0], literature.rhois[i][p]/1000, literature.Pis[i][p]/1e5])

    #Create new figure if needed:
    if do_new_figure:

        class_obj.fig, [class_obj.ax, class_obj.cax]  = plt.subplots(1, 2, layout='constrained', gridspec_kw={'width_ratios': [1, 0.01]})
        class_obj.cax.axis('off')
        class_obj.ax.set_xlabel(label_xy[x]); class_obj.ax.set_ylabel(label_xy[y])
        class_obj.ax.set_xscale(scale_xy[x]); class_obj.ax.set_yscale(scale_xy[y])
        if x == 2 or x == 3:
            class_obj.ax.invert_xaxis()

        #Add inset plot within figure if needed:
        if inset_plot_xy is not None:

            class_obj.ax_ins = inset_axes(class_obj.ax, width=inset_plot_xy.width, height=inset_plot_xy.height, loc=inset_plot_xy.loc, borderpad=inset_plot_xy.borderpad, bbox_to_anchor=inset_plot_xy.bbox_to_anchor, bbox_transform = class_obj.ax.transAxes, axes_kwargs={'xlim':inset_plot_xy.xlim, 'ylim':inset_plot_xy.ylim, 'xscale': scale_xy[inset_plot_xy.x], 'yscale': scale_xy[inset_plot_xy.y]})
            class_obj.ax_ins.set_xlabel(label_xy[inset_plot_xy.x]); class_obj.ax_ins.set_ylabel(label_xy[inset_plot_xy.y])
            if inset_plot_xy.x == 2 or inset_plot_xy.x == 3:
                class_obj.ax_ins.invert_xaxis()
            
    #Do the plotting:     
    class_obj.ax.plot(xy[x], xy[y], color=opts['color'], linestyle=opts['linestyle'], label=opts['label'], zorder=opts['zorder'])

    #Do the plotting (inset plot):
    if inset_plot_xy is not None:

        class_obj.ax_ins.plot(xy[inset_plot_xy.x], xy[inset_plot_xy.y], color=opts['color'], linestyle=opts['linestyle'], label=opts['label'], zorder=opts['zorder'])

    #Do the plotting (user provided data):
    if literature is not None:

        for i in range(len(literature_xys)):
            
            #Do the plotting: 
            class_obj.ax.plot(literature_xys[i][x], literature_xys[i][y], color=literature.colors[i], linestyle=literature.linestyles[i], label=literature.labels[i], zorder=literature.zorders[i])
            
            #Do the plotting (inset plot):
            if inset_plot_xy is not None:

                class_obj.ax_ins.plot(literature_xys[i][inset_plot_xy.x], literature_xys[i][inset_plot_xy.y], linestyle=literature.linestyles[i], label=literature.labels[i], zorder=literature.zorders[i])
    
    #Add a colorbar if needed:
    if colorbar is not None:

        gs = class_obj.ax.get_subplotspec().get_gridspec(); gs.set_width_ratios([1, colorbar.width])
        class_obj.cax.axis('on'); class_obj.fig.set_size_inches(mpl_opts['figure.figsize'][0]*colorbar.stretch, mpl_opts['figure.figsize'][1], forward=True)

        cbar = class_obj.fig.colorbar(colorbar.mappable, cax=class_obj.cax, orientation='vertical')
        cbar.set_label(colorbar.label)
        cbar.ax.tick_params(left=True, right=True)

        if hasattr(colorbar, "tick_lab"):
            cbar.set_ticks(colorbar.tick_pos); cbar.set_ticklabels(colorbar.tick_lab)

        if hasattr(colorbar, "tick_show"):
            if not colorbar.tick_show:
                cbar.ax.tick_params(left=False, right=False)

        cbar.ax.tick_params(which='minor', left=False, right=False)

    #Do the legend if needed:
    if opts['do_legend']:

        class_obj.ax.legend(loc=opts['loc_legend'], ncol=opts['ncol_legend'], frameon=opts['frameon_legend'], bbox_to_anchor=opts['bbox_to_anchor_legend'])

    #Save the plot if needed:
    if opts['do_save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], format=opts['format'], transparent=opts['transparent'])

def plot_shape(class_obj, **kwargs):

    #Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts()       , **kwargs}; _apply_mpl_opts(mpl_opts)
    opts     = {**_default_plot_shape_opts(), **kwargs}

    #Define coordinates:       
    theta   = np.linspace(0, 2*np.pi, 1000)
    r_l_mu  = class_obj.get_r_l_mu(np.cos(theta))
    N       = np.shape(r_l_mu)[0]

    #Define variables for plotting:
    X = np.zeros((N+1, 1000)); Y = np.ones ((N+1, 1000)); Z = np.ones((N+1, 1000))                   

    X[:N,:] = r_l_mu/np.max(class_obj.li) #ensures a radius 0 entry is present
    Y *= theta/(2*np.pi)*360
    Z[:N,:] = np.outer(class_obj.rhoi/np.max(class_obj.rhoi), np.ones_like(theta))

    #Create new figure:
    class_obj.fig, class_obj.ax  = plt.subplots(layout='constrained')

    #Do the plotting:
    con         = class_obj.ax.contourf(X, Y, Z, levels=opts['contourf_levels'], cmap=opts['contourf_cmap'])
    con_line    = class_obj.ax.contour (X, Y, Z, levels=opts['contour_levels'], colors=opts['contour_colors'])
    cbar        = class_obj.fig.colorbar(con, ticks=opts['colorbar_ticks'])
    
    #Do labeling:
    class_obj.ax.clabel(con_line, inline=True, fontsize=opts['contour_fontsize'])
    cbar.ax.tick_params(left=True, right=True)
    cbar.set_label(r'$\rho$ [$\cdot$' + str(np.round(np.max(class_obj.rhoi/1000),2)) + ' g/cm$^3$]')
    class_obj.ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360], labels=['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE', 'N'])
    class_obj.ax.set_xlabel(r'spheroid shell radius / average surface radius')
    
    #Save the plot if needed:
    if opts['do_save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '_cartesian.' + opts['format'], format=opts['format'], transparent=opts['transparent'])

    #Create new figure:
    class_obj.fig, class_obj.ax = plt.subplots(layout='constrained', subplot_kw={'projection': 'polar'})

    #Do the plotting:
    con     = class_obj.ax.contourf(Y*(2*np.pi)/360, X, Z, levels=opts['contourf_levels'], cmap=opts['contourf_cmap'])
    cbar    = class_obj.fig.colorbar(con, ticks=opts['colorbar_ticks'])
    line    = class_obj.ax.plot(theta, np.ones_like(theta), color=(0.8, 0.8, 0.8, 1.0), label=r'average surface radius')
    
    #Do labeling:
    cbar.ax.tick_params(left=True, right=True)
    cbar.set_label(r'$\rho$ [$\cdot$' + str(np.round(np.max(class_obj.rhoi/1000),2)) + ' g/cm$^3$]')
    class_obj.ax.set_theta_zero_location("N")
    class_obj.ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315, 360], labels=['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE', 'N'])
    class_obj.ax.set_rgrids([], labels=[])
    class_obj.ax.spines['polar'].set_visible(False)
    class_obj.ax.legend(handles=line, loc='center', bbox_to_anchor=(0.5, -0.15), borderaxespad=0, frameon=False)

    #Save the plot if needed:
    if opts['do_save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '_polar.' + opts['format'], format=opts['format'], transparent=opts['transparent'])

def plot_ss(class_obj, **kwargs):

    #Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts(), **kwargs}; _apply_mpl_opts(mpl_opts)
    opts     = {**_default_plot_ss() , **kwargs}

    #Create new figure:
    class_obj.fig, class_obj.ax  = plt.subplots(layout='constrained')

    #Get data for plotting:
    ss = class_obj.ss

    for i in range(len(ss)):

        if i != 0:   

            x = np.append(class_obj.li/class_obj.li[0], 0)
            y = np.append(np.flip(ss[i]/np.max(abs(ss[i]))), ss[i][0]/np.max(abs(ss[i]))) 

            #Do the plotting:
            val, exp = '{:.1e}'.format(np.max(abs(ss[i]))).split('e')
            class_obj.ax.plot(x, y, label=rf'${float(val):.1f} \cdot 10^{{{int(exp)}}} s_{{{2*i}}}$')
    
    #Do the labeling:
    class_obj.ax.set_xlabel(r'average $r/R$')
    class_obj.ax.set_ylabel(r'normalised figure functions')

    #Do the legend if needed:
    if opts['do_legend']:

        class_obj.ax.legend(loc=opts['loc_legend'], ncol=opts['ncol_legend'], frameon=opts['frameon_legend'], bbox_to_anchor=opts['bbox_to_anchor_legend'])

    #Save the plot if needed:
    if opts['do_save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], format=opts['format'], transparent=opts['transparent'])

def plot_state_xy(class_obj, x, y, state, what_model, literature=None, inset_plot_xy=None, colorbar=None, **kwargs):

    #Apply default and user supplied options:
    mpl_opts    = {**_default_mpl_opts()    , **kwargs}; _apply_mpl_opts(mpl_opts)
    opts        = {**_default_plot_xy_opts(), **kwargs}

    #Add a colorbar if needed and set colors to use:
    if colorbar is not None:

        idxs    = np.argsort(colorbar.sort_by)
        colors  = colorbar.cmap((colorbar.sort_by-min(colorbar.sort_by)) / (max(colorbar.sort_by) - min(colorbar.sort_by)))[idxs]

    else:

        idxs   = range(len(state[:,0]))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*(len(idxs)//10+1)

    #Initialize iteration:
    j = 0; do_save = opts['do_save']; do_legend = opts['do_legend']

    #Do the iteration:
    for i in idxs:

        #Call cost functions depending on what model was used:
        if what_model == 'baro':

            costJs = class_obj.baro_cost_function(state[i,:], return_sum=False)

        elif what_model == 'dens':

            costJs = class_obj.dens_cost_function(state[i,:], return_sum=False)
            
        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_model! Use \'baro\' or \'dens\'.' + c.ENDC)

        #Check for data consistency according to cost function results:
        if (np.abs(costJs)<1e0).all():

            color           = colors[j]
            opts['zorder']  = j
            j              += 1
            
        else:

            color           = (0.8, 0.8, 0.8, 0.3)
            opts['zorder']  = 0

            if colorbar is not None:

                j += 1

        #Prepare arguments for plot_xy():
        if i == idxs[0] and len(idxs) == 1:

            opts['color'] = color; opts['do_save'] = do_save; do_new_figure_arg = True ; literature_arg = literature; opts['do_legend'] = do_legend

        elif i == idxs[0]:

            opts['color'] = color; opts['do_save'] = False  ; do_new_figure_arg = True ; literature_arg = None      ; opts['do_legend'] = False

        elif i != idxs[-1]:

            opts['color'] = color; opts['do_save'] = False  ; do_new_figure_arg = False; literature_arg = None      ; opts['do_legend'] = False

        else:

            opts['color'] = color; opts['do_save'] = do_save; do_new_figure_arg = False; literature_arg = literature; opts['do_legend'] = do_legend

        #Call plot_xy():
        plot_xy(class_obj, x, y, do_new_figure=do_new_figure_arg, literature=literature_arg, inset_plot_xy=inset_plot_xy, colorbar=colorbar, **opts)

def plot_state_corr_xy(class_obj, x, y, state, what_model, sigma_lim=np.inf, literature=None, colorbar=None, **kwargs):

    #Apply default and user supplied options:
    mpl_opts    = {**_default_mpl_opts()    , 'axes.xmargin': 0.05, 'axes.ymargin': 0.05, **kwargs}; _apply_mpl_opts(mpl_opts)
    opts        = {**_default_plot_xy_opts(), **kwargs}

    #Define storage lists:
    Js_list = []; Js_err_list = []; color_list = []; match_list = []

    #Add a colorbar if needed and set colors to use:
    if colorbar is not None:

        idxs    = np.argsort(colorbar.sort_by)
        colors  = colorbar.cmap((colorbar.sort_by-min(colorbar.sort_by)) / (max(colorbar.sort_by) - min(colorbar.sort_by)))[idxs]

    else:

        idxs   = range(len(state[:,0]))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*(len(idxs)//10+1)
    
    #Initialize iteration:
    j = 0

    #Do the iteration:
    for i in idxs:

        #Call cost functions depending on what model was used:
        if what_model == 'baro':

            costJs = class_obj.baro_cost_function(state[i,:], return_sum=False)

        elif what_model == 'dens':

            costJs = class_obj.dens_cost_function(state[i,:], return_sum=False)
            
        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_model! Use \'baro\' or \'dens\'.' + c.ENDC)

        #Check for data consistency according to cost function results:
        if (np.abs(costJs)<1e0).all():
            
            match_list.append(True)

            color           = colors[j]
            opts['zorder']  = j
            j              += 1
            
        else:

            match_list.append(False)

            color           = (0.8, 0.8, 0.8, 0.3)
            opts['zorder']  = 0

            if colorbar is not None:

                j += 1

        #Save results:
        Js_list.append(class_obj.Js)
        Js_err_list.append(class_obj.Js_error)
        color_list.append(color)
    
    #Convert storage lists to arrays:
    Js_list = np.array(Js_list); Js_err_list = np.array(Js_err_list)
    
    #Create new figure:
    fig = plt.figure(figsize=(mpl_opts['figure.figsize'][0], mpl_opts['figure.figsize'][0]), layout='constrained')
    gs = fig.add_gridspec(2, 3,   width_ratios=(4, 1.25, 0.01), height_ratios=(1.25, 4), wspace=0.00, hspace=0.00)

    #Define axes:
    ax          = fig.add_subplot(gs[1, 0])
    ax_leg      = fig.add_subplot(gs[0, 1])
    ax_histx    = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy    = fig.add_subplot(gs[1, 1], sharey=ax)

    #Set custom axes options:
    ax_leg.set_xticks([]); ax_leg.set_yticks([])
    ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=True , right=False, labelbottom=False, labelleft=True )
    ax_histy.tick_params(axis="both", which="both", bottom=True , top=False, left=False, right=False, labelbottom=True , labelleft=False)

    for spine in ['top', 'right']:
        ax_histx.spines[spine].set_visible(False)
        ax_histy.spines[spine].set_visible(False)

    for spine in ax_leg.spines.values():
        spine.set_visible(False)

    #Define data to be plotted along the x dimension according to user input:
    if x >= len(Js_list[0,:]):

        x_label = r'$b_{' + str(x-len(Js_list[0,:])) + r'}$'
        x_array = state[:,x-len(Js_list[0,:])]
        x_err   = np.zeros_like(x_array)
        
    else:

        x_label = r'$J_{' + str(2*x) + r'}$'
        x_array = Js_list[:,x]
        x_err   = Js_err_list[:,x]

    #Define data to be plotted along the y dimension according to user input:
    if y >= len(Js_list[0,:]):

        y_label = r'$b_{' + str(y-len(Js_list[0,:])) + r'}$'
        y_array = state[:,y-len(Js_list[0,:])]
        y_err   = np.zeros_like(y_array)
        
    else:

        y_label = r'$J_{' + str(2*y) + r'}$'
        y_array = Js_list[:,y]
        y_err   = Js_err_list[:,y]

    #Find order of magnitude for the data to be plotted:
    if np.any(abs(x_array)):

        x_scale = math.floor(math.log(np.max(abs(x_array)), 10))+1

    else:

        x_scale = 0

    if np.any(abs(y_array)):

        y_scale = math.floor(math.log(np.max(abs(y_array)), 10))+1

    else:

        y_scale = 0

    #Normalise the data to be plotted:
    x_array /= 10**x_scale; y_array /= 10**y_scale
    x_err   /= 10**x_scale; y_err   /= 10**y_scale

    #Define and set x and y labels:
    x_label = r' $10^{'+str(-x_scale)+'}$' + x_label
    y_label = r' $10^{'+str(-y_scale)+'}$' + y_label
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)

    #Apply the sigma_lim potentially supplied by the user, sigma_lim=np.inf has no effect:
    mask_x = (x_array - np.average(x_array))**2/np.std(x_array)**2 < sigma_lim**2
    mask_y = (y_array - np.average(y_array))**2/np.std(y_array)**2 < sigma_lim**2
    mask = np.logical_and(mask_x, mask_y)

    x_array = x_array[mask]; y_array = y_array[mask]; x_err = x_err[mask]; y_err = y_err[mask]
    color_list = [c for c, m in zip(color_list, mask) if m]

    #Plot the data:
    for xa, ya, xe, ye, c, match in zip(x_array, y_array, x_err, y_err, color_list, match_list):
        if not match:
            xe = np.nan; ye=np.nan
        ax.errorbar(xa, ya, xerr=xe, yerr=ye, color=c, fmt='o')

    #Plot the x-data histogram:
    bins_x = np.linspace(x_array.min(), x_array.max(), 20)
    ax_histx.hist(x_array, bins=bins_x, alpha=0.5, color='black')
    counts, bin_edges = np.histogram(x_array, bins=bins_x)
    x_outline_x = np.repeat(bin_edges, 2)
    y_outline_x = np.hstack(([0], np.repeat(counts, 2), [0])) 
    ax_histx.plot(x_outline_x, y_outline_x, color='black')

    #Plot the y-data histogram:
    bins_y = np.linspace(y_array.min(), y_array.max(), 20)
    ax_histy.hist(y_array, bins=bins_y, alpha=0.5, orientation='horizontal', color='black')
    counts, bin_edges = np.histogram(y_array, bins=bins_y)
    y_outline_y = np.repeat(bin_edges, 2)
    x_outline_y = np.hstack(([0], np.repeat(counts, 2), [0]))
    ax_histy.plot(x_outline_y, y_outline_y, color='black')

    #Denote spearman correlation coefficient with confidence interval value:
    p1, = ax.plot(np.nan, np.nan); p2, = ax.plot(np.nan, np.nan)
    ax_leg.legend([p1,p2], [r"$\rho=${:.2f}".format(scipy.stats.spearmanr(x_array, y_array, nan_policy='omit')[0]), r"$p=${:.2f}".format(scipy.stats.spearmanr(x_array, y_array, nan_policy='omit')[1])], handletextpad=0, handlelength=0, loc='center', ncol=1, frameon=False, bbox_to_anchor=(0.0,0.0,1.0,1.0))

    #Plot target gravitational moments:
    if (x < len(Js_list[0,:])) and (y < len(Js_list[0,:])):

        ax.errorbar(class_obj.opts['Target_Js'][x-1]/10**x_scale, class_obj.opts['Target_Js'][y-1]/10**y_scale, xerr = class_obj.opts['Sigma_Js'][x-1]/10**x_scale, yerr = class_obj.opts['Sigma_Js'][y-1]/10**y_scale, color='k', fmt='o', capsize=3.0, capthick=3.0)
 
    #Add a colorbar if needed:
    if colorbar is not None:

        cax = fig.add_subplot(gs[1, 2]); gs.set_width_ratios([4, 1.25, colorbar.width])
        fig.set_size_inches(mpl_opts['figure.figsize'][0]*colorbar.stretch, mpl_opts['figure.figsize'][0], forward=True)

        cbar = fig.colorbar(colorbar.mappable, cax=cax, orientation='vertical')
        cbar.set_label(colorbar.label)
        cbar.ax.tick_params(left=True, right=True)

        if hasattr(colorbar, "tick_lab"):
            cbar.set_ticks(colorbar.tick_pos); cbar.set_ticklabels(colorbar.tick_lab)

        if hasattr(colorbar, "tick_show"):
            if not colorbar.tick_show:
                cbar.ax.tick_params(left=False, right=False)

        cbar.ax.tick_params(which='minor', left=False, right=False)

    #Ensure that the scatter plot focuses on the data points:
    ax.set_xlim((np.min(x_array) - mpl_opts['axes.xmargin']*(np.max(x_array) - np.min(x_array)), np.max(x_array) + mpl_opts['axes.xmargin']*(np.max(x_array) - np.min(x_array))))
    ax.set_ylim((np.min(y_array) - mpl_opts['axes.ymargin']*(np.max(y_array) - np.min(y_array)), np.max(y_array) + mpl_opts['axes.ymargin']*(np.max(y_array) - np.min(y_array))))

    #Do the legend if needed:
    if opts['do_legend']:

        ax_histx.legend([p1,p2], [r"$\mu=${:.3f}".format(np.average(x_array)), r"$\sigma=${:.0e}".format(np.std(x_array))], 
                  handletextpad=0, handlelength=0, loc=opts['loc_legend'], ncol=opts['ncol_legend'], frameon=opts['frameon_legend'], bbox_to_anchor=opts['bbox_to_anchor_legend'])

        ax_histy.legend([p1,p2], [r"$\mu=${:.3f}".format(np.average(y_array)), r"$\sigma=${:.0e}".format(np.std(y_array))], 
                  handletextpad=0, handlelength=0, loc=opts['loc_legend'], ncol=opts['ncol_legend'], frameon=opts['frameon_legend'], bbox_to_anchor=opts['bbox_to_anchor_legend'])

    #Save the plot if needed:
    if opts['do_save']:

        class_obj.fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], format=opts['format'], transparent=opts['transparent'])

def plot_autocorr(autocorr, **kwargs):

    #Apply default and user supplied options:
    mpl_opts    = {**_default_mpl_opts()    , 'axes.xmargin': 0.05, 'axes.ymargin': 0.05, **kwargs}; _apply_mpl_opts(mpl_opts)
    opts        = {**_default_plot_autocorr(), **kwargs}

    #Create new figure:
    fig,ax  = plt.subplots(layout='constrained')

    #Plot the data:
    ax.plot(100*np.arange(1, len(autocorr)+1), np.arange(1, len(autocorr)+1), '--', label=r'$\tau=N/100$', color='black')
    ax.plot(100*np.arange(1, len(autocorr)+1), autocorr                     , 'o-', label=r'estimated $\tau$')

    #Set labels for x- and y-axes:
    ax.set_xlabel(r'number of steps $N$')
    ax.set_ylabel(r'integrated autocorrelation time')
    
    #Do the legend if needed:
    if opts['do_legend']:

        ax.legend(loc=opts['loc_legend'], ncol=opts['ncol_legend'], frameon=opts['frameon_legend'], bbox_to_anchor=opts['bbox_to_anchor_legend'])

    #Save the plot if needed:
    if opts['do_save']:

        fig.savefig(opts['path_name'] + '/' + opts['fig_name'] + '.' + opts['format'], format=opts['format'], transparent=opts['transparent'])