import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


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
