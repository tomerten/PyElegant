import multiprocessing
from collections import ChainMap

import dask.delayed as delay
import numpy as np
import PyNAFF as pnf
from dask import dataframe as dd
from joblib import Parallel, delayed
from tqdm import tqdm


def divide_list_in_n_equal_chunks(_list, n):
    """
    Method to divide a list
    in n equal length chunks.

    Arguments:
    ----------
        - _list : list to divide
        - n : chunk size

    Returns:
    --------
        - chunk generator
    """
    for i in range(0, len(_list), n):
        yield _list[i : i + n]


def NaffOnGroup(name, group, Nturn):
    """
    Function that applies the pynaff
    algorithm on groupby object.
    Nturn should be half of number of
    rows in group.

    Arguments:
    ----------
        - name : particleID
        - group : corresponding groupby group
        - Nturn : half the length of group

    Returns:
    --------
        - dict: name : [diffusion_rate, outx0, outx1, outy0, outy1]
    """
    try:
        group.loc[:, "dx"] = group["x"] - group["x"].mean()
        group.loc[:, "dy"] = group["y"] - group["y"].mean()

        signal = group["dx"].values[:Nturn]
        outx0 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

        signal = group["dx"].values[Nturn:]
        outx1 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

        signal = group["dy"].values[:Nturn]
        outy0 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

        signal = group["dy"].values[Nturn:]
        outy1 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

        diffusion_rate = np.log10(
            np.sqrt((outx1 - outx0) ** 2 + (outy1 - outy0) ** 2) / (2 * Nturn)
        )
        return {name: [diffusion_rate, outx0, outx1, outy0, outy1]}
    except:
        print(name)
        return {name: [0, 0, 0, 0, 0]}


def NaffWindowScan(name, group, Nturn, nmax=20000):
    """
    Function that applies the pynaff
    algorithm on groupby object.
    Nturn should be half of number of
    window size.

    Arguments:
    ----------
        - name : particleID
        - group : corresponding groupby group
        - Nturn : half the length of the windowsize
        - nmax : length of the group

    Returns:
    --------
        - dict: particleID : list of [diffusion_rate, outx0, outx1, outy0, outy1]
    """
    diffusionlist = []

    for window in range(0, nmax - 2 * Nturn, 2 * Nturn):
        try:
            grp = group.reset_index(drop=True)

            dx = (
                grp.loc[window : window + 2 * Nturn, "x"]
                - grp.loc[window : window + 2 * Nturn, "x"].mean()
            )
            dy = (
                grp.loc[window : window + 2 * Nturn, "y"]
                - grp.loc[window : window + 2 * Nturn, "y"].mean()
            )

            signal = dx.values[:Nturn]
            outx0 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

            signal = dx.values[Nturn:]
            outx1 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

            signal = dy.values[:Nturn]
            outy0 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

            signal = dy.values[Nturn:]
            outy1 = pnf.naff(signal, Nturn, 1, 0, False)[0][1]

            diffusion_rate = np.log10(
                np.sqrt((outx1 - outx0) ** 2 + (outy1 - outy0) ** 2) / (2 * Nturn)
            )

            diffusionlist.append(np.array([diffusion_rate, outx0, outx1, outy0, outy1]))
        except:
            diffusionlist.append(np.array([0, 0, 0, 0, 0]))

    return {name: diffusionlist}


def applyParallel(dfGrouped, func, Nturn):
    """
    Function to apply func,
    in parallel on dfGrouped (groupby object).

    Arguments:
    ----------
        - name : particleID
        - group : corresponding groupby group
        - Nturn : half the length of group

    Returns:
    --------
        - dict: {particleID : [diffusion_rate, outx0, outx1, outy0, outy1]}
    """
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(name, group, Nturn) for name, group in tqdm(dfGrouped)
    )
    return dict(ChainMap(*retLst))
