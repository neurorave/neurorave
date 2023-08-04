import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def hdr_plot_style():
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Courier New'
    # mpl.rcParams['text.hinting'] = False
    # Set colors cycle
    colors = mpl.cycler('color', ['#3388BB', '#EE6666', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    # plt.rc('figure', facecolor='#00000000', edgecolor='black')
    # plt.rc('axes', facecolor='#FFFFFF88', edgecolor='white', axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')


# define function that allows to generate a number of sub plots in a single line with the given titles
def prep_plots(titles, fig_size, fig_num=1):
    """
    create a figure with the number of sub_plots given by the number of totles, and return all generated subplot axis
    as a list
    """
    # first close possibly existing old figures, if you dont' do this Juyter Lab will coplain after a while when it collects more than 20 existing ficgires for the same cell
    # plt.close(fig_num)
    # create a new figure
    hdr_plot_style()
    fig = plt.figure(fig_num, figsize=fig_size)
    ax_list = []
    for ind, title in enumerate(titles, start=1):
        ax = fig.add_subplot(1, len(titles), ind)
        ax.set_title(title)
        ax_list.append(ax)
    return ax_list


def finalize_plots(axes_list, legend=True, fig_title=None):
    """
    adds grid and legend to all axes in the given list
    """
    if fig_title:
        fig = axes_list[0].figure
        fig.suptitle(fig_title, y=1)
    for ax in axes_list:
        ax.grid(True)
        if legend:
            ax.legend()
