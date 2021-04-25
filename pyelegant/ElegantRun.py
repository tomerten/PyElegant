import os
import shlex
import subprocess as subp
from io import StringIO

import numpy as np
import pandas as pd
from scipy import constants as const

from .ElegantCommand import ElegantCommandFile
from .SDDSTools.SDDS import SDDS, SDDSCommand
from .SDDSTools.Utils import GenerateNDimCoordinateGrid


def write_parallel_elegant_script():
    """
    Method to generate a script that runs
    pelegant from bash.
    """

    # list of strings to write
    bashstrlist = [
        "#!/usr/bin/env bash",
        "if [ $# == 0 ] ; then",
        '   echo "usage: run_Pelegant <inputfile>"',
        "   exit 1",
        "fi",
        "n_cores=`grep processor /proc/cpuinfo | wc -l`",
        "echo The system has $n_cores cores.",
        "n_proc=$((n_cores-1))",
        "echo $n_proc processes will be started.",
        "if [ ! -e ~/.mpd.conf ]; then",
        '  echo "MPD_SECRETWORD=secretword" > ~/.mpd.conf',
        "  chmod 600 ~/.mpd.conf",
        "fi",
        "mpiexec -host $HOSTNAME -n $n_proc Pelegant  $1 $2 $3 $4 $5 $6 $7 $8 $9",
    ]

    bashstr = "\n".join(bashstrlist)

    # write to file
    with open("temp_run_pelegant.sh", "w") as f:
        f.write(bashstr)


def write_parallel_run_script(sif):
    """
    Method to generate parallel elegant run
    script.
    """
    bashstrlist = [
        "#!/bin/bash",
        "pele={}".format(sif),
        'cmd="bash temp_run_pelegant.sh"',
        "",
        "$pele $cmd $1",
    ]
    bashstr = "\n".join(bashstrlist)

    # write to file
    with open("run_pelegant.sh", "w") as f:
        f.write(bashstr)


class ElegantRun:
    """
    Class to interact with Elegant and Parallel Elegant from Python.
    """

    _REQUIRED_KWARGS = ["use_beamline", "energy"]

    def __init__(self, sif, lattice: str, parallel=False, **kwargs):
        self.sif = sif
        self.lattice = lattice
        self.parallel = parallel
        self.kwargs = kwargs
        self.check()
        self.commandfile = ElegantCommandFile("temp.ele")

        # setting up executable
        if parallel:
            self._write_parallel_script()
            self.exec = "bash {}".format(self.pelegant)
        else:
            self.exec = "{} elegant ".format(self.sif)

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

    def _write_parallel_script(self):
        """
        Generate the script to run parallel elegant.
        Method sets self.pelegant to the script file.
        """
        write_parallel_elegant_script()
        write_parallel_run_script(self.sif)
        self.pelegant = "run_pelegant.sh"

    def run(self):
        """
        Run the commandfile.
        """
        # check if commandfile is not empty
        if len(self.commandfile.commandlist) == 0:
            print("Commandfile empty - nothing to do.")
            return

        # write Elegant command file to disk
        self.commandfile.write()

        # generate command string
        print(self.exec)
        cmdstr = "{} temp.ele".format(self.exec)
        print(cmdstr)

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
            magnets="%s.mag",  # for plotting profile
        )

    def add_basic_twiss(self):
        """
        Add basic twiss.
        """
        self.commandfile.addCommand(
            "twiss_output", filename="%s.twi", matched=1, radiation_integrals=1
        )

    def add_vary_element(self, **kwargs):
        """
        Add single vary element line.
        """
        self.commandfile.addCommand(
            "vary_element",
            name=kwargs.get("name", "*"),
            item=kwargs.get("item", "L"),
            intial=kwargs.get("initial", 0.0000),
            final=kwargs.get("final", 0.0000),
            index_number=kwargs.get("index_number", 0),
            index_limit=kwargs.get("index_limit", 1),
        )

    def add_very_element_from_file(self, **kwargs):
        """
        Add single vary element line, loading value from
        dataset file.
        """
        if "enumeration_file" not in kwargs.keys():
            print("External filename missing.")
        else:
            self.commandfile.addCommand(
                "vary_element",
                name=kwargs.get("name", "*"),
                item=kwargs.get("item", "L"),
                index_number=kwargs.get("index_number", 0),
                index_limit=kwargs.get("index_limit", 1),
                enumeration_file=kwargs.get("enumeration_file"),
                enumeration_column=kwargs.get("enumeration_column"),
            )

    def add_basic_controls(self):
        # add controls
        self.commandfile.addCommand("run_control")
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand("track")

    def add_watch(self, **kwargs):
        self.commandfile.addCommand(
            "insert_elements",
            name=kwargs.get("name", ""),
            type=kwargs.get("type", ""),
            exclude="",
            s_start=kwargs.get("s_start", -1),
            s_end=kwargs.get("s_end", -1),
            skip=kwargs.get("skip", 1),
            insert_before=kwargs.get("insert_before", 0),
            add_at_end=kwargs.get("add_at_end", 0),
            add_at_start=kwargs.get("add_at_start", 0),
            element_def=kwargs.get(
                "element_def", 'WQ: WATCH, FILENAME="%s-%03ld.wq", mode=coordinates'
            ),
        )

    def add_watch_at_start(self):
        self.add_watch(
            name="W",
            add_at_start=1,
            element_def=r"\"W: WATCH, FILENAME=\"%s-%03ld.wq\", mode=\"coordinates\"\"",
        )

    def add_fma_command(self, **kwargs):
        """
        Add elegant standard fma command.
        """

        self.commandfile.addCommand(
            "frequency_map",
            output="%s.fma",
            xmin=kwargs.get("xmin", -0.1),
            xmax=kwargs.get("xmax", 0.1),
            ymin=kwargs.get("ymin", 1e-6),
            ymax=kwargs.get("ymax", 0.1),
            delta_min=kwargs.get("delta_min", 0),
            delta_max=kwargs.get("delta_max", 0),
            nx=kwargs.get("nx", 21),
            ny=kwargs.get("ny", 21),
            ndelta=kwargs.get("ndelta", 1),
            verbosity=0,
            include_changes=kwargs.get("include_changes", 1),
            quadratic_spacing=kwargs.get("quadratic_spacing", 0),
            full_grid_output=kwargs.get("full_grid_output", 1),
        )

    def add_DA_command(self, **kwargs):
        """
        Add DA find aperture command.
        """
        self.commandfile.addCommand(
            "find_aperture",
            output="%s.aper",
            mode=kwargs.get("mode", "n-line"),
            verbosity=0,
            xmin=kwargs.get("xmin", -0.1),
            xmax=kwargs.get("xmax", 0.1),
            xpmin=kwargs.get("xpmin", 0.0),
            xpmax=kwargs.get("xpmax", 0.0),
            ymin=kwargs.get("ymin", 0.0),
            ymax=kwargs.get("ymax", 0.1),
            ypmin=kwargs.get("ypmin", 0.0),
            ypmax=kwargs.get("ypmax", 0.0),
            nx=kwargs.get("nx", 21),
            ny=kwargs.get("ny", 11),
            n_lines=kwargs.get("n_lines", 11),
            split_fraction=kwargs.get("split_fraction", 0.5),
            n_splits=kwargs.get("n_splits", 0),
            desired_resolution=kwargs.get("desired_resolution", 0.01),
            offset_by_orbit=kwargs.get("offset_by_orbit", 0),
            full_plane=kwargs.get("full_plane", 1),
        )

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
        self.add_basic_setup()

        # add twiss calc
        self.commandfile.addCommand(
            "twiss_output",
            matched=matched,
            output_at_each_step=0,
            filename="%s.twi",
            radiation_integrals=1,
        )

        # add controls
        self.add_basic_controls()

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

    def generate_sdds_particle_inputfile(self, **kwargs):
        """
        Generates an SDDS file containing initial
        particle coordinates on a grid. The grid
        can be defined through the kwargs.

        Arguments:
        ----------
        kwargs      :
            - pmin: min value of grid on each dim
            - pmax: max value of grid on each dim
            - pcentralmev: particle energy (code converts it to beta * gamma )
            - man_ranges: dict containing as key dim num - in order x xp y yp s p
                          and as values an array of values to be used
                          For p this is autoset to beta gamma based on pcentralmev
            - NPOINTS: number of linear spaced points in each dim for the grid

        Returns:
        --------
        None, writes the data to pre-defined named file.

        """
        npoints_per_dim = kwargs.get("NPOINTS", 2)
        pmin = kwargs.get("pmin", 0)
        pmax = kwargs.get("pmax", 1e-4)
        pcentral = kwargs.get("pcentralmev", 1700.00)
        # convert to beta * gamma
        pcentral = np.sqrt(
            (pcentral / const.physical_constants["electron mass energy equivalent in MeV"][0]) ** 2
            - 1
        )
        man_ranges = kwargs.get("man_ranges", {"5": np.array([pcentral])})
        if "5" not in man_ranges.keys() and 5 not in man_ranges.keys():
            man_ranges["5"] = np.array([pcentral])
        # example : man_ranges={'0':np.array([1e-6,1e-5]),'1':[0]})

        # generate coordinate grid, with particle id as last column
        # and save it as plain data table seperated by a whitespace
        particle_df = pd.DataFrame(
            GenerateNDimCoordinateGrid(
                6, npoints_per_dim, pmin=pmin, pmax=pmax, man_ranges=man_ranges
            )
        )
        particle_df.to_csv("temp_plain_particles.dat", sep=" ", header=None, index=False)

        # cleanup kwargs
        kwargs.pop("NPOINTS", None)
        kwargs.pop("pmin", None)
        kwargs.pop("pmax", None)
        kwargs.pop("pcentralmev", None)
        kwargs.pop("man_ranges", None)

        # Create sddscommand object
        sddscommand = SDDSCommand(self.sif)

        # update the command parameters
        if self.parallel:
            outputmode = "binary"
        else:
            outputmode = "ascii"
        kwargs["outputMode"] = outputmode
        kwargs["file_2"] = (
            "temp_particles_input.txt" if not self.parallel else "temp_particles_input.bin"
        )

        # load the pre-defined  convert plain data to sdds command
        cmd = sddscommand.get_particles_plain_2_SDDS_command(**kwargs)

        # run the sdds command
        sddscommand.runCommand(cmd)

        self.sdds_beam_file = kwargs["file_2"]

    def simple_single_particle_track(self, coord=np.zeros((5, 1)), **kwargs):
        """
        Track a single particle with given initial coordinates.

        Important:
        ----------
        Be careful with giving the 6th coordinate, this is beta * gamma. If not
        given it will be calculated automatically either using standard 1700 MeV
        or kwargs["pcentralmev"].

        """
        # generate particle input file
        self.generate_sdds_particle_inputfile(
            man_ranges={k: v for k, v in zip(range(coord.shape[0] + 1), coord)}, **kwargs
        )

        # construct command file
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand("run_control", n_passes=kwargs.get("n_passes", 2 ** 8))
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand(
            "sdds_beam",
            input=self.sdds_beam_file,
            input_type='"elegant"',
        )
        self.commandfile.addCommand("track")

        # run will write command file and execute it
        self.run()

    def manual_vary_input_table(self):
        """
        Create a vary table to use with elegant tracking,
        generated from manual input values.
        """
        pass

    def track_simple(self, **kwargs):
        """
        Track a set of particles.
        """
        # construct command file
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand("run_control", n_passes=kwargs.get("n_passes", 2 ** 8))
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand(
            "sdds_beam",
            input=self.sdds_beam_file,
            input_type='"elegant"',
        )
        self.commandfile.addCommand("track")

        # run will write command file and execute it
        self.run()

    def track_vary(self):
        """
        Track a set of particles in combination with a
        very command.
        """
        pass

    def fma(self, **kwargs):
        """
        Run Elegant fma.
        """
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand("run_control", n_passes=kwargs.pop("n_passes", 2 ** 8))
        self.add_basic_twiss()
        self.add_fma_command(**kwargs)

        self.run()

    def dynap(self, **kwargs):
        """
        Run Elegant's Dynamic Aperture.
        """
        self.commandfile.clear()
        self.add_basic_setup()

        self.commandfile.addCommand("twiss_output", filename="%s.twi", output_at_each_step=1)
        self.commandfile.addCommand("run_control", n_passes=kwargs.pop("n_passes", 2 ** 9))
        self.add_DA_command(**kwargs)

        self.run()

    def dynapmom(self):
        """
        Run Elegant's Dynamic Momentum Aperture.
        """
        pass

    def table_scan(self, scan_list_of_dicts, **kwargs):
        """"""
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand(
            "run_control",
            n_passes=kwargs.get("n_passes", 2 ** 8),
            n_indices=len(scan_list_of_dicts),
        )
        for i, l in enumerate(scan_list_of_dicts):
            self.commandfile.addCommand(
                "vary_element",
                naem=l.get("name"),
                item=l.get("item"),
                initial=l.get("initial"),
                final=l.get("final"),
                index_number=i,
                index_limit=l.get("index_limit"),
            )

        self.commandfile.addCommand(
            "sdds_beam",
            input=self.sdds_beam_file,
            input_type='"elegant"',
        )
        self.commandfile.addCommand("track")

        # run will write command file and execute it
        self.run()
