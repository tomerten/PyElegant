import fractions as frac
import math
import os
from pathlib import Path

import dask.delayed as delay
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask import dataframe as dd
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


class PhaseSpaceAnimation:
    """Class to generate phase-space animation"""

    def __init__(self, data, nmax, xcol="x", pcol="xp"):
        self.data = data.reset_index()
        self.nmax = nmax
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=2)
        self.xmin = 1.1 * data[xcol].min()
        self.pmin = 1.1 * data[pcol].min()
        self.xmax = 1.1 * data[xcol].max()
        self.pmax = 1.1 * data[pcol].max()
        self.x = []
        self.p = []
        self.intensity = []
        self.iterations = len(data)
        self.t = list(range(nmax))
        self.xcol = xcol
        self.pcol = pcol

    def get_new_vals(self, t):
        """
        Method to get data
        updates.
        """
        x = [self.data[self.xcol].iloc[t + 1]]
        p = [self.data[self.pcol].iloc[t + 1]]

        return list(x), list(p)

    def animate(self, i):
        self.ax.set_title(f"Turn: {i:3}")
        #         self.x = self.data.loc[:i,self.xcol].values
        #         self.p = self.data.loc[:i,self.pcol].values

        new_xvals, new_pxvals = self.get_new_vals(i)
        self.x.extend(new_xvals)
        self.p.extend(new_pxvals)

        # Put new values in your plot
        self.scatter.set_offsets(np.c_[self.x, self.p])

        # calculate new color values
        self.intensity = np.concatenate((np.array(self.intensity) * 0.99, np.ones(1)))

        self.scatter.set_array(self.intensity)

    #         return self.scatter,

    def start(self):
        #         Writer = animation.writers['ffmpeg']
        #         writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
        self.ax.set_xlabel("x", size=12)
        self.ax.set_ylabel("px", size=12)
        self.ax.grid()

        colors = [[0, 0, 1, 0], [0, 0, 1, 0.5], [0, 0.2, 0.4, 1]]
        cmap = LinearSegmentedColormap.from_list("", colors)

        self.xlimlist = [(self.xmin, self.xmax)]
        self.ylimlist = [(self.pmin, self.pmax)]

        self.scatter = self.ax.scatter(self.x, self.p, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
        self.ax.set_xlim(self.xlimlist[0][0], self.xlimlist[0][1])
        self.ax.set_ylim(self.ylimlist[0][0], self.ylimlist[0][1])
        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, self.animate, frames=self.t, interval=10, repeat=False, blit=True
        )
        plt.show()

    def save(self, fn):
        if self.anim is None:
            self.start()
        self.anim.save(fn)


def make_movie(data6d, particleID, maxturn, save=False, filename="test.mp4", plotranges=None):
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=24, metadata=dict(artist="Me"), bitrate=1800)
    nmax = maxturn

    # data
    sdata1 = data6d.loc[data6d["particleID"] == particleID]

    if plotranges is None:
        # calc plot limits
        xmin = 1.1 * sdata1["x"].min()
        pxmin = 1.1 * sdata1["xp"].min()
        ymin = 1.1 * sdata1["y"].min()
        pymin = 1.1 * sdata1["yp"].min()
        tmin = 1.2 * sdata1["dt"].min()
        pmin = 0.99 * sdata1["p"].min()

        xmax = 1.1 * sdata1["x"].max()
        pxmax = 1.1 * sdata1["xp"].max()
        ymax = 1.1 * sdata1["y"].max()
        pymax = 1.1 * sdata1["yp"].max()
        tmax = 1.1 * sdata1["dt"].max()
        pmax = 1.01 * sdata1["p"].max()
        xlimlist = [(xmin, xmax), (ymin, ymax), (tmin, tmax), (xmin, xmax)]
        ylimlist = [(pxmin, pxmax), (pymin, pymax), (pmin, pmax), (ymin, ymax)]
    else:
        xlimlist = [plotranges["xlim"], plotranges["ylim"], plotranges["tlim"], plotranges["xlim"]]
        ylimlist = [
            plotranges["pxlim"],
            plotranges["pylim"],
            plotranges["plim"],
            plotranges["pxlim"],
        ]

    # init variables to store plot data
    x_vals = []
    px_vals = []
    y_vals = []
    py_vals = []
    dt_vals = []
    p_vals = []

    intensity = []
    intensityp = []
    iterations = len(sdata1)

    # init frame steps
    t_vals = list(range(nmax))

    # plot setup
    fig = plt.figure(figsize=(8, 8))

    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

    axlist = [ax1, ax2, ax3, ax4]

    ax1.set_xlabel("x", size=12)
    ax1.set_ylabel("px", size=12)
    ax2.set_xlabel("y", size=12)
    ax2.set_ylabel("py", size=12)
    ax3.set_xlabel("dt", size=12)
    ax3.set_ylabel("p", size=12)
    ax4.set_xlabel("x", size=12)
    ax4.set_ylabel("y", size=12)

    for j, ax in enumerate(axlist):
        ax.grid()
        if xlimlist[j][0] == xlimlist[j][1]:
            ax.set_xlim(-1, 1)
        else:
            ax.set_xlim(xlimlist[j][0], xlimlist[j][1])
        if ylimlist[j][0] == ylimlist[j][1]:
            ax.set_ylim(-1, 1)
        else:
            ax.set_ylim(ylimlist[j][0], ylimlist[j][1])

    colors = [[0, 0, 1, 0], [0, 0, 1, 0.5], [0, 0.2, 0.4, 1]]
    cmap = LinearSegmentedColormap.from_list("", colors)

    # init plots
    scatter1 = ax1.scatter(x_vals, px_vals, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
    scatter2 = ax2.scatter(y_vals, py_vals, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
    scatter3 = ax3.scatter(dt_vals, p_vals, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
    scatter4 = ax4.scatter(x_vals, y_vals, s=2, c=[], cmap=cmap, vmin=0, vmax=1)

    def get_new_vals(t):
        """
        Method to get data
        updates.
        """
        x = [sdata1["x"].iloc[t]]
        px = [sdata1["xp"].iloc[t]]
        y = [sdata1["y"].iloc[t]]
        py = [sdata1["yp"].iloc[t]]
        dt = [sdata1["dt"].iloc[t]]
        p = [sdata1["p"].iloc[t]]
        return list(x), list(px), list(y), list(py), list(dt), list(p)

    def update(t):
        """
        Update the data storage
        variable, intensity and color.
        """
        global x_vals, px_vals, y_vals, py_vals, dt_vals, p_vals, intensity, intensityp

        # Get intermediate points
        new_xvals, new_pxvals, new_yvals, new_pyvals, new_dtvals, new_pvals = get_new_vals(t)
        x_vals.extend(new_xvals)
        px_vals.extend(new_pxvals)
        y_vals.extend(new_yvals)
        py_vals.extend(new_pyvals)
        dt_vals.extend(new_dtvals)
        p_vals.extend(new_pvals)

        # Put new values in your plot
        scatter1.set_offsets(np.c_[x_vals, px_vals])
        scatter2.set_offsets(np.c_[y_vals, py_vals])
        scatter3.set_offsets(np.c_[dt_vals, p_vals])
        scatter4.set_offsets(np.c_[x_vals, y_vals])

        # calculate new color values
        intensity = np.concatenate((np.array(intensity) * 0.98, np.ones(len(new_xvals))))
        intensityp = np.concatenate((np.array(intensityp) * 0.99, np.ones(len(new_xvals))))

        # update color variables
        scatter1.set_array(intensity)
        scatter2.set_array(intensity)
        scatter3.set_array(intensityp)
        scatter4.set_array(intensity)

        # Set title to show turn number
        for ax in axlist:
            ax.set_title(f"Turn: {t:3}")

        for j, ax in enumerate(axlist):
            ax.grid()
            if xlimlist[j][0] == xlimlist[j][1]:
                ax.set_xlim(-1, 1)
            else:
                ax.set_xlim(xlimlist[j][0], xlimlist[j][1])
            if ylimlist[j][0] == ylimlist[j][1]:
                ax.set_ylim(-1, 1)
            else:
                ax.set_ylim(ylimlist[j][0], ylimlist[j][1])

    # create the animation
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=t_vals, interval=1, repeat=False)

    # saving ?
    if save:
        ani.save(filename)

    plt.tight_layout()
    plt.show()


class ParticleAnimation6D:
    """Class for 6D phase space and 2D transverse particle animation."""

    def __init__(
        self, data, nmax, xcol="x", pxcol="xp", ycol="y", pycol="yp", tcol="dt", deltacol="p"
    ):
        self.data = data.reset_index()
        self.nmax = nmax

        self.xcol = xcol
        self.pxcol = pxcol
        self.ycol = ycol
        self.pycol = pycol
        self.tcol = tcol
        self.ptcol = deltacol

        self.fig = plt.figure(figsize=(8, 8))
        self.ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
        self.ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        self.ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
        self.ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

        self.xmin = 1.1 * data[xcol].min()
        self.pxmin = 1.1 * data[pxcol].min()
        self.ymin = 1.1 * data[ycol].min()
        self.pymin = 1.1 * data[pycol].min()
        self.tmin = 1.1 * data[tcol].min()
        self.ptmin = 1.1 * data[deltacol].min()

        self.xmax = 1.1 * data[xcol].max()
        self.pxmax = 1.1 * data[pxcol].max()
        self.ymax = 1.1 * data[ycol].max()
        self.pymax = 1.1 * data[pycol].max()
        self.tmax = 1.1 * data[tcol].max()
        self.ptmax = 1.1 * data[deltacol].max()

        self.x = []
        self.px = []
        self.y = []
        self.py = []
        self.dt = []
        self.pt = []

        self.intensity = []
        self.iterations = len(data)
        self.t = list(range(nmax))

    def _get_new_vals(self, t):
        """
        Method to get data
        updates.
        """
        x = [self.data[self.xcol].iloc[t]]
        px = [self.data[self.pxcol].iloc[t]]
        y = [self.data[self.ycol].iloc[t]]
        py = [self.data[self.pycol].iloc[t]]
        dt = [self.data[self.tcol].iloc[t]]
        p = [self.data[self.ptcol].iloc[t]]

        return list(x), list(px), list(y), list(py), list(dt), list(p)

    def animate(self, i):
        self.ax1.set_title(f"Turn: {i:3}")
        self.ax2.set_title(f"Turn: {i:3}")
        self.ax3.set_title(f"Turn: {i:3}")
        self.ax4.set_title(f"Turn: {i:3}")

        # Get intermediate points
        new_xvals, new_pxvals, new_yvals, new_pyvals, new_dtvals, new_pvals = self._get_new_vals(i)
        self.x.extend(new_xvals)
        self.px.extend(new_pxvals)
        self.y.extend(new_yvals)
        self.py.extend(new_pyvals)
        self.dt.extend(new_dtvals)
        self.pt.extend(new_pvals)

        # Put new values in your plot
        self.scatter1.set_offsets(np.c_[self.x, self.px])
        self.scatter2.set_offsets(np.c_[self.y, self.py])
        self.scatter3.set_offsets(np.c_[self.dt, self.pt])
        self.scatter4.set_offsets(np.c_[self.x, self.y])

        # calculate new color values
        self.intensity = np.concatenate((np.array(self.intensity) * 0.99, np.ones(1)))

        self.scatter1.set_array(self.intensity)
        self.scatter2.set_array(self.intensity)
        self.scatter3.set_array(self.intensity)
        self.scatter4.set_array(self.intensity)

    def start(self):
        self.ax1.set_xlabel("x", size=12)
        self.ax1.set_ylabel("px", size=12)
        self.ax2.set_xlabel("y", size=12)
        self.ax2.set_ylabel("py", size=12)
        self.ax3.set_xlabel("dt", size=12)
        self.ax3.set_ylabel("p", size=12)
        self.ax4.set_xlabel("x", size=12)
        self.ax4.set_ylabel("y", size=12)

        self.ax1.grid()
        self.ax2.grid()
        self.ax3.grid()
        self.ax4.grid()

        colors = [[0, 0, 1, 0], [0, 0, 1, 0.5], [0, 0.2, 0.4, 1]]
        cmap = LinearSegmentedColormap.from_list("", colors)

        self.xlimlist = [
            (self.xmin, self.xmax),
            (self.ymin, self.ymax),
            (self.tmin, self.tmax),
            (self.xmin, self.xmax),
        ]
        self.ylimlist = [
            (self.pxmin, self.pxmax),
            (self.pymin, self.pymax),
            (self.ptmin, self.ptmax),
            (self.ymin, self.ymax),
        ]

        # set axes limits if not zero
        for j, ax in enumerate([self.ax1, self.ax2, self.ax3, self.ax4]):
            if self.xlimlist[j][0] != self.xlimlist[j][1]:
                ax.set_xlim(self.xlimlist[j][0], self.xlimlist[j][1])
            if self.ylimlist[j][0] != self.ylimlist[j][1]:
                ax.set_ylim(self.ylimlist[j][0], self.ylimlist[j][1])

        self.scatter1 = self.ax1.scatter(self.x, self.px, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
        self.scatter2 = self.ax2.scatter(self.y, self.py, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
        self.scatter3 = self.ax3.scatter(self.dt, self.pt, s=2, c=[], cmap=cmap, vmin=0, vmax=1)
        self.scatter4 = self.ax4.scatter(self.x, self.y, s=2, c=[], cmap=cmap, vmin=0, vmax=1)

        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, self.animate, frames=self.t, interval=10, repeat=False, blit=True
        )
        plt.show()


SPINE_COLOR = "gray"


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + MAX_HEIGHT_INCHES
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES

    params = {
        "backend": "ps",
        "text.latex.preamble": "\\usepackage{gensymb}",
        "axes.labelsize": 20,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 20,
        "font.size": 20,  # was 10
        "legend.fontsize": 20,  # was 10
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "text.usetex": False,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "axes.formatter.limits": [-3, 3],
        "axes.formatter.use_mathtext": True,
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)

    return ax


def PlotTrackingTSA(
    pdata, nturns, npart, col="x", auto=True, vmax=250, vmin=-150, save=False, fn=None, **kwargs
):
    import matplotlib.ticker as ticker

    x = pdata[col].values.copy()
    x = x.reshape(2000, 24)

    if auto:
        norm = plt.cm.colors.Normalize(vmax=x.max(), vmin=x.min())
    else:
        norm = plt.cm.colors.Normalize(vmax=vmax, vmin=vmin)

    plt.figure(figsize=(12, 8))
    # be carefull - this is normalized to actual see a signal
    plt.imshow(abs(np.fft.rfft(x, axis=0)), aspect="auto", norm=norm)
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 2000))
    plt.gca().yaxis.set_major_formatter(ticks_y)

    plt.grid()
    plt.xlabel("ParticleID")
    plt.ylabel(r"$\nu_{frac}$")

    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        print(ylim)
        plt.ylim(tuple(np.array(ylim) * nturns))

    if save:
        plt.savefig(fn)


def updateOrder(lines, leftY, rightY, order, label):
    if leftY in lines:
        if (rightY in lines[leftY]) and (lines[leftY][rightY]["order"] < order):
            lines[leftY][rightY] = {"order": order, "label": label}
        else:
            lines[leftY][rightY] = {"order": order, "label": label}
    else:
        lines[leftY] = {}
        lines[leftY][rightY] = {"order": order, "label": label}


def farey_gen(N):
    """a generator to produce the ascending Farey sequence of order N,
    according to the algorithm described on
    http://en.wikipedia.org/wiki/Farey_sequence#Next_term"""

    a, b, c, d = 0, 1, 1, N
    yield frac.Fraction(a, b)
    while float(c) / d <= 1:
        k = (N + b) // d
        a, c = c, k * c - a
        b, d = d, k * d - b
        yield frac.Fraction(a, b)


def simplify(f1, f2, f3):
    """simplify the linear equation
    f1*x + f2*y + f3 = 0
    to
    a*x + b*y + c = 0,
    where f1, f2 and f3 are irreducible fractions,
    a, b and c are coprime integers"""

    def lcd(a, b, c):
        """lowest common denominator of three integers"""
        d = a * b // math.gcd(a, b)
        return c * d // math.gcd(c, d)

    def gcd(a, b, c):
        """greatest common divisor of three integers"""
        d = math.gcd(a, b)
        return math.gcd(c, d)

    d = frac.Fraction(lcd(f1.denominator, f2.denominator, f3.denominator))

    a = (f1 * d).numerator
    b = (f2 * d).numerator
    c = (f3 * d).numerator
    d = gcd(a, b, c)
    return a / d, b / d


def sibling(line):
    """find all the sibling of a line segment which have symmetric relation
    about the following axes: x = 1/2 or/and y = 1/2.
    The known line is given by a pair of points (4 coordinates).
    This function returns a set of point-pairs defining the related lines."""

    lines = set([line])
    x1, y1, x2, y2 = line
    lines.add((1 - x1, y1, 1 - x2, y2))
    lines.add((x2, 1 - y2, x1, 1 - y1))
    lines.add((1 - x2, 1 - y2, 1 - x1, 1 - y1))
    return lines


def generatelines(N, ax=None, freq=False, ref_freq=0):
    # N = int(raw_input("specify the order of the resonance: ")) # python3 -> input
    #     ax1.clear()
    if ax is None:
        traces = []

    N = int(N)
    if N > 0:
        Orders = range(1, N + 1)
        fareys = [[] for x in Orders]
        lines = [set() for x in Orders]

    for Ni in range(len(Orders)):

        fareys[Ni] = [f for f in farey_gen(Orders[Ni])]
        lines[Ni] = set()

        # select line segments which belong to resonance of order N
        for i, f1 in enumerate(fareys[Ni]):  # 0 <= f1 <= 1
            lines[Ni].add((f1, 0, f1, 1))  # vertical line
            lines[Ni].add((0, f1, 1, f1))  # horizontal line
            if f1 > 0 and 2 * f1.denominator <= Orders[Ni]:  # diagonal line, e.g. x + y - f1 = 0
                lines[Ni] = lines[Ni].union(sibling((f1, 0, 0, f1)))

            for f2 in fareys[Ni][1:i]:  # 0 < f2 < f1
                a, b = simplify(1, abs(f2 - f1), f1)
                if a + b <= Orders[Ni]:  # e.g. x + (f2-f1)*y - f1 = 0
                    lines[Ni] = lines[Ni].union(sibling((f1, 0, f2, 1)))
                    lines[Ni] = lines[Ni].union(sibling((1, f2, 0, f1)))  # flip about y = x

                a, b = simplify(f2, f1, f1 * f2)
                if a + b <= Orders[Ni]:  # e.g. f2*x + f1*y - f1*f2 = 0
                    lines[Ni] = lines[Ni].union(sibling((f1, 0, 0, f2)))
                    lines[Ni] = lines[Ni].union(sibling((f2, 0, 0, f1)))  # flip about y = x

    lines = list(reversed(lines))

    if len(lines) > 1:
        for i in range(len(lines) - 1):
            lines[i] = lines[i].difference(lines[i + 1])

    Ncount = 1
    linecolors = ["cyan", "red", "magenta", "blue", "green"]
    linecolors = plt.cm.tab10(np.linspace(0, 1, 10))

    if ax is None:
        linecolors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # safety orange
            "#2ca02c",  # cooked asparagus green
            "#d62728",  # brick red
            "#9467bd",  # muted purple
            "#8c564b",  # chestnut brown
            "#e377c2",  # raspberry yogurt pink
            "#7f7f7f",  # middle gray
            "#bcbd22",  # curry yellow-green
            "#17becf",  # blue-teal
        ]

    if freq:
        for i, x in enumerate(list(reversed(lines))):
            while x:
                x1, y1, x2, y2 = x.pop()
                if Ncount > 10:
                    if not ax is None:
                        ax.plot(
                            [x1 * ref_freq, x2 * ref_freq],
                            [y1 * ref_freq, y2 * ref_freq],
                            linewidth=np.divide(7, Ncount),
                            color="black",
                            alpha=0.5,
                        )
                    else:
                        traces.append(
                            dict(
                                type="scatter",
                                x=[x1 * ref_freq, x2 * ref_freq],
                                y=[y1 * ref_freq, y2 * ref_freq],
                                line=dict(color="black", width=np.divide(7, Ncount)),
                                showlegend=False,
                                opacity=0.5,
                            )
                        )
                else:
                    if not ax is None:
                        ax.plot(
                            [x1 * ref_freq, x2 * ref_freq],
                            [y1 * ref_freq, y2 * ref_freq],
                            linewidth=np.divide(7, Ncount),
                            alpha=0.5,
                            color=linecolors[Ncount - 1],
                        )
                    else:
                        traces.append(
                            dict(
                                type="scatter",
                                x=[x1 * ref_freq, x2 * ref_freq],
                                y=[y1 * ref_freq, y2 * ref_freq],
                                line=dict(
                                    color=linecolors[Ncount - 1], width=np.divide(7, Ncount)
                                ),
                                showlegend=False,
                                opacity=0.5,
                            )
                        )
            Ncount += 1
    else:
        for i, x in enumerate(list(reversed(lines))):
            while x:
                x1, y1, x2, y2 = x.pop()
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                if Ncount > 10:
                    if not ax is None:
                        ax.plot(
                            [x1, x2],
                            [y1, y2],
                            linewidth=np.divide(7, Ncount),
                            alpha=0.5,
                            color="black",
                        )
                    else:
                        traces.append(
                            dict(
                                type="scatter",
                                x=[x1, x2],
                                y=[y1, y2],
                                line=dict(color="black", alpha=0.5, width=np.divide(7, Ncount)),
                                showlegend=False,
                                opacity=0.5,
                            )
                        )
                else:
                    if not ax is None:
                        ax.plot(
                            [x1, x2],
                            [y1, y2],
                            linewidth=np.divide(7, Ncount),
                            alpha=0.5,
                            color=f"rgba({tuple(linecolors[Ncount - 1])}",
                        )
                    else:
                        traces.append(
                            dict(
                                type="scatter",
                                x=[x1, x2],
                                y=[y1, y2],
                                line=dict(
                                    color=linecolors[Ncount - 1], width=np.divide(7, Ncount)
                                ),
                                showlegend=False,
                                opacity=0.5,
                            )
                        )

            Ncount += 1
    if ax is None:
        return traces


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = "$%s$" % matplotlib.ticker._mathdefault(self.format)
