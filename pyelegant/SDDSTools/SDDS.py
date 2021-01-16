import os
import shlex
import subprocess as subp
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
from dask import dataframe as dd

from .SDDSTools import sddsconvert2ascii, sddsconvert2binary


class SDDSCommand:
    _COMMANDLIST = [
        "plaindata2sdds",
        "sdds2stream",
        "sddsconvert",
        "sddsquery",
        "sddsprocess",
        "sddsplot",
        "sddsprintout",
        "sddsmakedataset",
        "sddsnaff",
    ]

    def __init__(self, sif):
        self.sif = sif
        self.command = {}

    def createCommand(self, command, note="", **params):
        """
        Method to add a command to the command file.
        Generates a dict to reconstruct command string,
        that can be added to the command file.

        Arguments:
        ----------
        command     : str
            valid Elegant command
        note        :

        params      :

        """
        # check if it is a valid Elegant command
        if self.checkCommand(command) == False:
            print("The command {} is not recognized.".format(command))
            return

        # init command dict
        thiscom = {}
        thiscom["NAME"] = command

        # add command parameters to the command dict
        for k, v in params.items():
            thiscom[k] = v

        # add the command dict to the command list
        self.command = thiscom

    def checkCommand(self, typename):
        """
        Check if a command is a valid
        Elegant command.

        Arguments:
        ----------
        typename    : str
            command type

        """
        for tn in self._COMMANDLIST:
            if tn == typename.lower():
                return True
        return False

    def clearCommand(self):
        """
        Clear the command list.

        """
        self.command = {}

    def getCommand(self, command, **params):
        self.createCommand(command, **params)
        cmdstr = "{} {}".format(self.sif, self.command["NAME"])

        for k, v in params.items():
            if "file" in k.lower():
                cmdstr += " {}".format(v)
            elif "separator" in k.lower():
                cmdstr += ' "-separator= {}"'.format(v)
            else:
                if v is not None:
                    cmdstr += " -{}={}".format(k.split("_")[0], v)
                else:
                    cmdstr += " -{}".format(k.split("_")[0])

        return cmdstr

    def runCommand(self, commandstring):
        print("Running command {}".format(commandstring))
        subp.run(commandstring, check=True, shell=True)

    def get_particles_plain_2_SDDS_command(self, **params):
        """
        Returns sdds command to turn plain data table, separated
        by a space into SDDS particle initial coordinates SDDS file.
        Easy to generate plain data with pandas.

        Example:
            pd.DataFrame([
            {
                'x':0.000,
                'px':1.
            },
             {
                'x':5.000,
                'px':1.
            }
            ]).to_csv('testplain.dat',sep=' ',header=None, index=False)

        Arguments:
        ----------
        kwargs      :
            - outputMode: set ascii for serial elegant and binary for Pelegant

        """
        return self.getCommand(
            "plaindata2sdds",
            file_1=params.get("file_1", "temp_plain_particles.dat"),
            file_2=params.get("file_2", "temp_particles_input.txt"),
            inputMode=params.get("inputMode", "ascii"),
            outputMode=params.get("outputMode", "ascii"),
            separator=" ",
            column_1="x,double,units=m",
            column_2="xp,double",
            column_3="y,double,units=m",
            column_4="yp,double",
            column_5="t,double,units=s",
            column_6='p,double,units="m$be$nc"',
            columns_7="particleID,long",
            noRowCount=None,
        )


class SDDS:
    """
    Class for interacting with SDDS files.
    """

    def __init__(self, sif: str, filename: str, filetype: int):
        self.sif = sif
        self.filetype = filetype
        self._filename = filename
        self.columnlist = None
        self.commandlist = []
        self.command_history = {}

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    def addCommand(self, command, **params):
        """
        Method to add a command to the command file.

        Arguments:
        ----------
        command     : str
            valid Elegant command
        params      :

        """
        sddscommand = SDDSCommand(self.sif)
        cmdstr = sddscommand.getCommand(command, **params)

        # add command to current commandlist
        self.commandlist.append(cmdstr)

    def clearCommandList(self):
        """
        Clear the command list.

        """
        self.commandlist = {}

    def clearHistory(self):
        """
        Clear command history.
        """
        self.command_history = {}

    def _addHistory(self, cmdstr):
        """
        Add commands to history.
        """
        _key = str(len(self.command_history))
        _value = cmdstr
        self.command_history[_key] = _value

    def runCommand(self):
        """
        Run all the sddscommands in the command list.
        """
        if len(self.commandlist) > 0:
            for cmd in range(len(self.commandlist)):
                # add commands to history
                self._addHistory(cmd)

                # run command
                print("Executing : \n{}".format(cmd))

                p = subp.Popen(cmd, stdout=subp.PIPE, shell=True)
                (output, err) = p.communicate()
                p_status = p.wait()

        else:
            print("No commands entered - nothing to do!")

        self.clearCommandList()

    def runHistory(self, output=False):
        """
        Rerun all command in command history.
        """
        print("Excuting History")
        print("----------------")
        for _, v in self.command_history.items():
            print("Executing : \n{}".format(v))
            p = subp.Popen(v, stdout=subp.PIPE, shell=True)
            (output, err) = p.communicate()
            p_status = p.wait()

    def load(self):
        if self.filetype == 1:
            "ASCII FORMAT"
            with open(self.filename, "r") as f:
                self.raw_content = f.read()

        else:
            "BINARY FORMAT"
            with open(self.filename, "rb") as f:
                self.raw_content = f.read()

        # process

    def convert(self, outfile):
        if self.filetype == 0:
            converted_filename = self.filename + ".txt"

            if outfile is not None:
                converted_filename = outfile

            cmdstr = "{} sddsconvert -ascii {} {}".format(
                self.sif, self.filename, converted_filename
            )
        else:
            converted_filename = self.filename + ".bin"

            if outfile is not None:
                converted_filename = outfile

            cmdstr = "{} sddsconvert -binary {} {}".format(
                self.sif, self.filename, converted_filename
            )
        # add to command history
        self._addHistory(cmdstr)

        # reset filename
        print("Warning - auto filename set")
        print("Changed from {} to {}".format(self.filename, converted_filename))
        self.filename = converted_filename
        print("Warning - auto filetype set")
        print("Changed from {} to {}".format(self.filetype, abs(1 - self.filetype)))
        self.filetype = abs(1 - self.filetype)

        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

    def process_scan(self):
        self.addCommand(
            "sddsprocess",
            define="column,step,Step {}".format(self.filename),
            outfile="{}_processed.{}".format(
                self.filename.split(".")[0], self.filename.split(".")[1]
            ),
        )
        self.runCommand()

        # cmdstr = "{} sddsprocess -define=column,step,Step {} {}_processed.{}".format(
        #    self.sif, self.filename, *self.filename.split(".")
        # )
        # print(cmdstr)
        # p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        newfilename = "{}_processed.{}".format(*self.filename.split("."))
        print("Warning - auto filename set")
        print("Changed from {} to {}".format(self.filename, newfilename))

        self.filename = newfilename

    def getColumnList(self):
        cmdstr = "{} sddsquery -columnList {}".format(self.sif, self.filename)
        p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        self.columnlist = [l.decode("utf-8") for l in output.splitlines()]
        return self.columnlist

    def getParameterList(self):
        cmdstr = "{} sddsquery -parameterList {}".format(self.sif, self.filename)
        p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        return [l.decode("utf-8") for l in output.splitlines()]

    def getParameterValues(self):
        cmdstr = "{} sddsprintout -parameters=* -spreadsheet {}".format(self.sif, self.filename)
        p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        df = pd.read_csv(
            StringIO(output.decode("utf-8")),
            error_bad_lines=False,
            delim_whitespace=True,
            skip_blank_lines=True,
            skiprows=1,
            names=["ParameterName", "ParameterValue"],
            index_col=False,
        )

        df = df.set_index("ParameterName", drop=True)
        df = pd.to_numeric(df.drop(["SVNVersion", "Stage"])["ParameterValue"])
        self.ParameterName = df
        return df

    def getColumnValues(self):
        if self.columnlist is None:
            self.getColumnList()

        if os.path.getsize(self.filename) > 100e6:
            print(
                "File is large, output redirected to file {}".format(
                    self.filename + "_columnvalues.dat"
                )
            )
            cmdstr = "{} sdds2stream -col={} {} > {}".format(
                self.sif,
                ",".join(self.columnlist),
                self.filename,
                self.filename + "_columnvalues.dat",
            )
            return self.filename + "_columnvalues.dat"
        else:
            cmdstr = "{} sdds2stream -col={} {}".format(
                self.sif, ",".join(self.columnlist), self.filename
            )
            p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
            (output, err) = p.communicate()
            p_status = p.wait()
            output = output.decode("utf-8")
            output = pd.read_csv(StringIO(output), names=self.columnlist, delim_whitespace=True)
            return output

    def readParticleData(self, vary=False):
        """"""
        print(vary)
        print(self.filename)

        if vary:
            self.process_scan()

        self.getColumnList()

        print(self.filename)
        print(self.columnlist)

        if self.filetype == 0:
            self.convert(outfile=None)

        df = self.getColumnValues()
        if isinstance(df, str):
            data = dd.read_csv(df, delimiter=" ", names=self.columnlist, header=None)
        else:
            data = df

        if vary:
            grouped = data.groupby(by="step")

            def f(group):
                return group.join(
                    pd.DataFrame({"Turn": group.groupby("particleID").cumcount() + 1})
                )

            data = grouped.apply(f)

        else:
            data["Turn"] = data.groupby("particleID").cumcount() + 1

        return data

    def sddsplot_with_string(self, **kwargs):
        self.addCommand("sddsplot", string=kwargs.get("string", "-col=s,x"))

    def sddsplot_base(self, **kwargs):
        sddscommand = SDDSCommand(self.sif)
        cmd = sddscommand.getCommand("sddsplot", **kwargs)
        sddscommand.runCommand(cmd)

    def sddsplot(
        self,
        columnNames=["x", "xp"],
        markerstyle="sym",
        vary="subtype",
        scalemarker=1,
        fill=True,
        order="spectral",
        split="page",
        scale="0,0,0,0",
        **kwargs,
    ):

        if fill:
            strfill = ",fill"
        else:
            strfill = ""
        # TODO extra options
        # extra = " ".join(["-{}={}".format(k, v) for k, v in kwargs.items()])
        cmd = f"{self.sif} sddsplot -columnNames={','.join(columnNames)} {self.filename} "
        cmd += f"-graph={markerstyle},vary={vary}{strfill},scale={str(scalemarker)} -order={order} -split={split} -scale={scale}"
        subp.run(cmd, check=True, shell=True)

    def generate_scan_dataset(self, datasetdict):
        """
        Generates a file called "temp_scan.sdds" containing columns of values
        to be used by elegant to scan over using vary_element method.

        Arguments:
        ----------
        datadict: dict
            dictionary where the keys are the column headers and values are list of values to scan over
            Note: all dict values need to have the same length

        """
        cmd = f"{self.sif}  sddsmakedataset temp_scan.sdds "

        for k, v in datasetdict.items():
            cmd += f"-column={k},type=double -data=" + ",".join([str(vv) for vv in v]) + " "

        subp.run(cmd, check=True, shell=True)
        self.sdds_scan = "temp_scan.sdds"
