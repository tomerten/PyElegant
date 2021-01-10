import os
import shlex
import subprocess as subp
from io import StringIO

import numpy as np
import pandas as pd

from .ElegantCommand import ElegantCommandFile
from .SDDSTools.SDDS import SDDS


class ElegantRun:
    _REQUIRED_KWARGS = ["use_beamline", "energy"]

    def __init__(self, sif, lattice: str, parallel=False, **kwargs):
        self.sif = sif
        self.lattice = lattice
        self.parallel = parallel
        self.kwargs = kwargs
        self.check()
        self.commandfile = ElegantCommandFile("temp.ele")

    def check(self):
        """
        Check if all necessary info
        is given for Elegant to be able to run.
        """
        if not all(req in self.kwargs.keys() for req in self._REQUIRED_KWARGS):
            print("Missing required kwargs...")
            print("Minimum required are:")
            for r in self._REQUIRED_KWARGS:
                print(r)

    def run(self, parallel=False):
        """
        Run the commandfile.

        Arguments:
        ----------
        parallel    : Bool
            run serial or parallel Elegant
        """
        if len(self.commandfile.commandlist) == 0:
            print("Commandfile empty - nothing to do.")
            return

        # check if commandfile is not empty
        self.commandfile.write()

        # set cmdstr
        if parallel:
            pass
        else:
            cmdstr = "{} elegant temp.ele".format(self.sif)

        # run
        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

    def add_basic_setup(self, **kwargs):
        """
        Add basic setup command.
        """
        self.commandfile.clear()
        self.commandfile.addCommand(
            "run_setup",
            lattice=self.lattice,
            use_beamline=self.kwargs.get("use_beamline", None),
            p_central_mev=self.kwargs.get("energy", 1700.00),
            centroid="%s.cen",
            default_order=kwargs.get("default_order", 3),
            concat_order=kwargs.get("concat_order", 3),
            rootname="temp",
            parameters="%s.params",
            semaphore_file="%s.done",
            magnets="%s.mag",
        )

    def add_basic_controls(self):
        # add controls
        self.commandfile.addCommand("run_control")
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand("track")

    def findtwiss(self, **kwargs):
        """
        Run Twiss and return Twiss parameters
        together with Twiss data.

        Arguments:
        ----------
        kwargs  : dict
            twiss command options
        """
        # TODO: add matched = 0 case
        matched = kwargs.get("matched", 1)
        initial_optics = kwargs.get("initial_optics", [])
        alternate_element = kwargs.get("alternate_elements", {})
        closed_orbit = kwargs.get("closed_orbit", 1)

        # make sure not residual is there
        self.commandfile.clear()

        # add setup command
        self.commandfile.addCommand(
            "run_setup",
            lattice=self.lattice,
            use_beamline=self.kwargs.get("use_beamline", None),
            rootname="temp",
            p_central_mev=self.kwargs.get("energy"),
            centroid="%s.cen",
            default_order=3,
            concat_order=3,
        )

        # add twiss calc
        self.commandfile.addCommand(
            "twiss_output",
            matched=matched,
            output_at_each_step=0,
            filename="%s.twi",
            radiation_integrals=1,
        )

        # add controls
        self.commandfile.addCommand("run_control")
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand("track")

        # write command file
        self.commandfile.write()

        # set cmdstr
        cmdstr = "{} elegant temp.ele".format(self.sif)
        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

        # load twiss output
        twifile = SDDS(self.sif, "temp.twi", 0)
        twiparams = twifile.getParameterValues()
        twidata = twifile.getColumnValues()

        twiparams["length"] = np.round(twidata.iloc[-1]["s"], 3)

        return twidata, twiparams

    def find_matrices(self, **kwargs):
        """
        Find element by element matrix and map elements (depending on given order).
        Constant vector and R matrix are returned as numpy arrays, the maps are
        returned as dicts.

        Arguments:
        ----------
        kwargs  :
            - SDDS_output_order : order of maps (max is 3)

        Returns:
        --------
        C       : np.array
            constant vector
        R       : np.array
            R matrix
        T_dict  : dict
            T map Tijk as key
        Q_dict  : dict
            U map Qijkl as key
        """

        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand(
            "matrix_output",
            SDDS_output="%s.sdds",
            SDDS_output_order=kwargs.get("SDDS_output_order", 1),
            printout="%s.mat",
            printout_order=kwargs.get("SDDS_output_order", 1),
            full_matrix_only=kwargs.get("full_matrix_only", 0),
            individual_matrices=kwargs.get("individual_matrices", 1),
            output_at_each_step=kwargs.get("output_at_each_step", 1),
        )

        # add controls
        self.add_basic_controls()

        # write command file
        self.commandfile.write()

        # set cmdstr
        cmdstr = "{} elegant temp.ele".format(self.sif)
        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

        with open("temp.mat", "r") as f:
            mdata = f.read()

        # get full turn matrix and
        dfmat = pd.read_csv(
            StringIO("\n".join(mdata.split("full", 1)[1].splitlines()[1:])),
            delim_whitespace=True,
            names=[1, 2, 3, 4, 5, 6],
        )
        C = dfmat.loc[dfmat.index == "C:"].values.T
        R = dfmat.loc[dfmat.index.str.contains("R")].values
        T = dfmat.loc[dfmat.index.str.contains("T")]
        Q = dfmat.loc[dfmat.index.str.contains("Q")]

        T_dict = {}
        for _, row in T.iterrows():
            _basekey = row.name[:-1]
            for c in T.columns:
                _key = _basekey + str(c)
                _value = row[c]
                if not pd.isna(_value):
                    T_dict[_key] = _value

        Q_dict = {}
        for _, row in Q.iterrows():
            _basekey = row.name[:-1]
            for c in Q.columns:
                _key = _basekey + str(c)
                _value = row[c]
                if not pd.isna(_value):
                    Q_dict[_key] = _value

        sddsmat = SDDS(self.sif, "temp.sdds", 0)
        ElementMatrices = sddsmat.getColumnValues()

        return C, R, ElementMatrices, T_dict, Q_dict

    def simple_single_particle_track(self, coord=np.zeros((6, 1)), **kwargs):
        """
        Track a single particle with given initial coordinates.
        """
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand("run_control", n_passes=kwargs.get("n_passes", 2 ** 8))
        self.commandfile.addCommand("bunched_beam")

    def generate_particle_lattice(self):
        """
        Generate a lattice of particle coordinates,
        based on boundaries and number of points per
        dimension.
        """
        pass

    def manual_vary_input_table(self):
        """
        Create a vary table to use with elegant tracking,
        generated from manual input values.
        """
        pass

    def track_simple(self):
        """
        Track a set of particles.
        """
        if self.parallel:
            pass
        else:
            pass

    def track_vary(self):
        """
        Track a set of particles in combination with a
        very command.
        """
        pass

    def fma(self):
        """
        Run Elegant fma.
        """
        pass

    def dynap(self):
        """
        Run Elegant's Dynamic Aperture.
        """
        pass

    def dynapmom(self):
        """
        Run Elegant's Dynamic Momentum Aperture.
        """
