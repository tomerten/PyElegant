class ElegantCommandFile:
    _COMMANDLIST = (
        "alter_elements",
        "amplification_factors",
        "analyze_map",
        "aperture_data",
        "bunched_beam",
        "change_particle",
        "chromaticity",
        "closed_orbit",
        "correct",
        "correction_matrix_output",
        "correct_tunes",
        "coupled_twiss_output",
        "divide_elements",
        "error_element",
        "error_control",
        "find_aperture",
        "floor_coordinates",
        "frequency_map",
        "global_settings",
        "insert_elements ,insert_sceffects",
        "linear_chromatic_tracking_setup",
        "link_control",
        "link_elements",
        "load_parameters",
        "matrix_output",
        "modulate_elements",
        "moments_output",
        "momentum_aperture",
        "optimize",
        "optimization_constraint",
        "optimization_covariable",
        "optimization_setup",
        "parallel_optimization_setup",
        "optimization_term",
        "optimization_variable",
        "print_dictionary",
        "ramp_elements",
        "rf_setup",
        "replace_elements",
        "rpn_expression",
        "rpn_load",
        "run_control",
        "run_setup",
        "sasefel",
        "save_lattice",
        "sdds_beam",
        "semaphores",
        "slice_analysis",
        "subprocess",
        "steering_element",
        "touschek_scatter",
        "transmute_elements",
        "twiss_analysis",
        "twiss_output",
        "track",
        "tune_shift_with_amplitude",
        "vary_element",
    )

    def __init__(self, filename):
        self.commandlist = []
        self.filename = filename

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

    def addCommand(self, command, note="", **params):
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

        thiscom["NAME"] = command.lower()
        thiscom["NOTE"] = note

        # add command parameters to the command dict
        for k, v in params.items():
            thiscom[k] = v

        # add the command dict to the command list
        self.commandlist.append(thiscom)

    def modifyCommand(self, commandname, mode="last", **params):
        """
        Method to modify a command already in the command list.
        If the command is used multiple times, the command you
        want to change can be select by using the strings "first" or "last"
        or by giving the integer index of the command.

        Arguments:
        ----------
        commandname     : str
            name of the command to modify
        mode            : str
            if multiple commands select which one to change
        params          : dict
            command parameters - updated

        """
        # create command index list to be able
        # to choose which one to change
        indlist = []
        i = 0

        for command in self.commandlist:
            if command["NAME"] == commandname.lower():
                indlist.append(i)
            i += 1

        # find command to updated given in mode
        # allowed are : first, last or integer
        if len(indlist) == 1:

            for k, v in params.items():
                self.commandlist[indlist[0]][k] = v
            return

        elif len(indlist) == 0:
            print("No such command {} can be found".format(commandname))

        else:
            if mode == "last":
                for k, v in params.items():
                    self.commandlist[indlist[-1]][k] = v
                return
            elif mode == "first":
                for k, v in params.items():
                    self.commandlist[indlist[0]][k] = v
                return

            elif isinstance(mode, int):
                for k, v in params.items():
                    self.commandlist[indlist[mode]][k] = v
                return

            else:
                print("The mode is invalid")

    def repeatCommand(self, commandname, mode="last"):
        """
        Method to repeat a commend already in the
        command list.


        Arguments:
        ----------
        commandname     : str
            name of the command to modify
        mode            : str
            if multiple commands select which one to change

        """
        # create index list for the command to be repeated
        indlist = []
        i = 0
        for command in self.commandlist:
            if command["NAME"] == commandname.lower():
                indlist.append(i)
            i += 1

        if len(indlist) == 1:
            self.commandlist.append(self.commandlist[indlist[0]])
            return

        elif len(indlist) == 0:
            print("No such command {} can be found".format(commandname))

        else:
            if mode == "last":
                self.commandlist.append(self.commandlist[indlist[-1]])
            elif mode == "first":
                self.commandlist.append(self.commandlist[indlist[0]])
            elif isinstance(mode, int):
                self.commandlist.append(self.commandlist[indlist[mode]])
            else:
                print("The mode is invalid")

    def clear(self):
        self.commandlist = []

    def optimize(
        self,
        lattice,
    ):
        pass

    def write(self, outputfilename="", mode="w"):
        """
        Method to write the command file to external file.


        """
        if outputfilename != "":
            self.filename = outputfilename

        # writing the command file
        with open(self.filename, mode) as outfile:
            # looping over the commands
            for command in range(len(self.commandlist)):
                outfile.write("&{}\n".format(self.commandlist[command]["NAME"]))

                if self.commandlist[command]["NOTE"] != "":
                    outfile.write("! {}\n".format(self.commandlist[command]["NOTE"]))

                for k, v in self.commandlist[command].items():
                    if k != "NAME" and k != "NOTE":
                        if len(k) <= 19:
                            outfile.write("\t{:20s}= {},\n".format(k, v))
                        elif len(k) <= 29:
                            outfile.write("\t{:30s}= {},\n".format(k, v))
                        else:
                            outfile.write("\t{:40s}= {},\n".format(k, v))
                outfile.write("&end\n\n")
