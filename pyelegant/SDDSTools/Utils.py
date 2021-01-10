import numpy as np


def GenerateNDimCoordinateGrid(N, NPOINTS, pmin=1e-6, pmax=1e-4, man_ranges=None):
    """
    Method to generate an N dimensional coordinate grid for tracking,
    with fixed number of point in each dimension.
    The final shape is printed at creation.

    IMPORTANT:
        Number of grid points scales with N * NPOINTS**N, i.e.
        very large arrays are generated already with
        quite some small numbers for NPOINTS and N.

        Example: NPOINTS = 2, N = 6 -> 6*2*6 = 384 elements

    Arguments:
    ----------
    N: int
        dimension of the coordinate grid
    NPOINTS: int
        number of points in each dimension
    pmin: float
        min coordinate value in each dim
    pmax: float
        max coordinate value in each dim

    """
    rangelist = [np.linspace(pmin, pmax, NPOINTS)] * N
    if man_ranges is not None:
        print(man_ranges)
        for k, v in man_ranges.items():
            rangelist[int(k)] = v
        print(rangelist)
    grid = np.meshgrid(*rangelist)
    coordinate_grid = np.array([*grid])
    npart = coordinate_grid.size // N
    coordinate_grid = coordinate_grid.reshape(N, npart).T
    print("Shape: {} - Number of paritcles: {} ".format(coordinate_grid.shape, npart))
    # add particle id
    coordinate_grid = np.hstack((coordinate_grid, np.array(range(1, npart + 1)).reshape(npart, 1)))
    return coordinate_grid
