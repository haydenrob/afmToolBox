# Author:   Hayden Robertson
# Date:     24/10/2021

import pandas as pd
import numpy as np
from scipy import constants
import re
import matplotlib.pyplot as plt
import lmfit
import os
from tqdm.notebook import tqdm
import statistics
import pathlib

import warnings

warnings.simplefilter("ignore", np.RankWarning)
np.seterr(divide="ignore", invalid="ignore")

#### The below constants are used for later calcs ######
k_B = constants.k  # Boltzman constant
N_a = constants.Avogadro  # /mol


class AFM:

    """
    Parameters:
    ___________
    plot : bool, to produce plots 'as you go'.
    h_dry : float, dry thickness of the brush in m
    rho : float, density of the polymer in g/m3
    a : float, length of monomer units (i.e., width of side chain) in m
    MM : float, molecular mass of monomer in g/mol
    T : float, temperature in K

    """

    def __init__(
        self,
        plot=False,
        h_dry=10e-9,
        rho=1.26 * (100**3),
        a=0.44e-9,
        MM=150,
        T=296,
        chi_tol=1e-19,
        start_data=0,
        SprCons=0,
    ):
        self.plot = plot
        self.h_dry = h_dry
        self.rho = rho
        self.a = a
        self.MM = MM
        self.T = T
        self.chi_tol = chi_tol
        self.start_data = start_data
        self.SprCons = SprCons

    fc = []
    not_fc = []

    def import_file(self, f):
        """
        Function to import files from Bruker AFM. Files need to be exported as
        as text files with column headers and x, y data for approach and retract.

        Parameters:
        _____________
        f : string, file name/directory of file to import.

        Returns:
        _____________
        data : This contains all the meta data in the afm file (e.g. spring const)


        """

        self.f = f

        data = open(f, "r", errors="ignore")
        self.d = f[:7]
        self.con = f[7:-4]
        self.stop = False

        return data

    def find_start(self):
        """
        This function finds the row number where AFM data starts
        """

        data = open(self.f, "r", errors="ignore")

        for i, ln in enumerate(data.readlines()):
            if "\*Force file list end" in ln:
                self.start_data = i + 2

    def read_const(self, data):
        """
        This function uses the data loaded from AFM.import_files and saves
        important information to variables to use for analysis.
        """

        line = data.read().strip()

        ln = re.findall(r"\@Sens. DeflSens: .*", line)[0]
        self.DeflSens = float(
            re.split("(\d+)", ln)[1] + "." + re.split("(\d+)", ln)[3]
        )

        if self.SprCons == 0:
            ln = re.findall(r"\Spring Constant: .*", line)[0]
            self.SprCons = float(
                re.split("(\d+)", ln)[1] + "." + re.split("(\d+)", ln)[3]
            )

        ln = re.findall(r"\@4:Z scale: .*", line)[0]
        self.zScale = float(
            re.split("(\d+)", ln)[3] + "." + re.split("(\d+)", ln)[5]
        )

        ln = re.findall(r"\Samps/line:.*", line)[1]
        self.samps_line = int(re.split("(\d+)", ln)[1])

        data.close()

        del data

    def read_data(self, f):
        """
        Reads the actual data from the AFM data file and saves as a pd.DataFrame.
            To increase speed of code this should be changed to np.array.

        If any NAN exist, they're dropped. y-data is also flipped to better
            represent a force curve.

        Parameters:
        _____________
        f : string, file name/directory of file to import.

        Returns:
        _____________
        data_app : pd.DataFrame containing approach z data in nm, deflection
            in V and in nm.
        data_ret : pd.DataFrame containing retract z data in nm, deflection
            in V and in nm.

        """

        try:
            data = pd.read_csv(
                f,
                skiprows=self.start_data - 1,
                sep="\t",
                usecols=[0, 1, 2, 3],
                encoding="ISO-8859-1",
            )
        except:
            AFM.find_start(self)
            data = pd.read_csv(
                f,
                skiprows=self.start_data - 1,
                sep="\t",
                usecols=[0, 1, 2, 3],
                encoding="ISO-8859-1",
            )

        data_ret = data[["Calc_Ramp_Ex_nm", "Defl_V_Ex"]]
        data_app = data[["Calc_Ramp_Rt_nm", "Defl_V_Rt"]]

        data_ret = data_ret.rename(
            columns={"Calc_Ramp_Ex_nm": "z_nm", "Defl_V_Ex": "defl_V"}
        )
        data_app = data_app.rename(
            columns={"Calc_Ramp_Rt_nm": "z_nm", "Defl_V_Rt": "defl_V"}
        )

        data_app = data_app.dropna(axis="rows")
        data_ret = data_ret.dropna(axis="rows")

        data_app["defl_V"] = np.flip(np.array(data_app["defl_V"]))
        data_ret["defl_V"] = np.flip(np.array(data_ret["defl_V"]))

        data_app.insert(2, "defl_nm", data_app["defl_V"] * self.DeflSens)
        data_ret.insert(2, "defl_nm", data_ret["defl_V"] * self.DeflSens)

        del data

        return data_ret, data_app

    def load_data(self, fh):
        """
        This function automates the import process and saves a tuple of
            data_app and data_ret to self.

        Parameters:
        _____________
        f : string, file name/directory of file to import.

        Returns:
        _____________
        data : tuple containing two pd.DataFrame's, each containing z data
            in nm, deflection in V and in nm.

        """

        data = AFM.import_file(self, fh)

        AFM.read_const(self, data)

        data = AFM.read_data(self, fh)

        nam = fh[:-4]
        self.nam = nam[-3:]

        self.data = data

        return data

    def baseline_corr(self, lower=-200, upper=-50, plot=False):
        """
        Determines what the baseline is of an AFM curve and then zeroes
            that baseline. Some data is flipped as approach and retract curves
            travel in opposite directions. AFM.data tuple is updated.
            Was 400 to 850, but change depending on ramp size and brush thickness.
            This is also scanner dependent. Default is now to use the last 150 data points.

        Parameters:
        _____________
        lower : int, lower bound for the index for baseline calculation.
            Default is -200.
        upper : int, upper bound for the index for baseline calculation.
            Default is -50.

        """

        # AFM.plotting(self)
        if plot is True:
            fig, ax = plt.subplots(1, 2, figsize=(7, 4))

        # approach and retraction curves go in different directions,
        # so indicies are different.
        flip = False

        for fh in self.data:
            xn = np.array(fh["z_nm"]).astype(float)
            yn = np.array(fh["defl_nm"]).astype(float)

            if flip:
                xn = np.flip(xn)
                yn = np.flip(yn)

            # Try and fit a straight line to the 'baseline'
            try:
                m_bl, b_bl = np.polyfit(xn[lower:upper], yn[lower:upper], 1)
            except:
                print("Not a real force curve")
                # If you can't fit a straight line, it's not a real force curve
                # and hence stop the fitting process and it's a bad curve.
                if not flip:
                    AFM.not_fc.append(self.nam)
                self.stop = True
                self.bad = True
                return xn, yn

            if flip:
                y = yn - (m_bl * xn + b_bl)
                self.data[1]["defl_cal_nm"] = y
                fh = self.data[1]
            else:
                y = np.flip(yn - (m_bl * xn + b_bl))
                self.data[0]["defl_cal_nm"] = y
                fh = self.data[0]

            if plot is True:
                ax[0].plot(xn, yn)
                ax[0].plot(xn[lower:upper], m_bl * xn[lower:upper] + b_bl)
                ax[1].plot(fh["z_nm"].values, fh["defl_cal_nm"].values)
                for a in ax:
                    a.set_xlabel("Distance (nm)")
                ax[0].set_ylabel("Deflection (nm)")

            flip = True

        return

    def x_cal(self, i_skp=10, e_skp=10, plot=False):
        """
        Function that calibrates the x data to be apparent separation.
            AFM.data tuple is updated.

        Parameters:
        _____________
        i_skp : int, number of initial data points in the constant compliance
            region to skip when fitting a straight line. Default is 10.
        e_skp : int, number of initial data points in the constant compliance
            region to skip when fitting a straight line. Default is 10.
        plot : bool, if True, then plot the data.

        """
        # AFM.plotting(self)
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        flip = False

        for idx, fh in enumerate(self.data):
            if flip:
                y_data = (fh["defl_cal_nm"]).astype(float)
                z_data = (fh["z_nm"]).astype(float)
            else:
                y_data = np.flip(fh["defl_cal_nm"]).astype(float)
                z_data = np.flip(fh["z_nm"]).astype(float)

            for i in range(len(y_data[i_skp:])):
                r2 = (
                    np.corrcoef(
                        z_data[i_skp : i + i_skp + 2].astype(float),
                        y_data[i_skp : i + i_skp + 2].astype(float),
                    )[0, 1]
                    ** 2
                )
                if r2 < 0.99:
                    break

            # In the above np.corrcoef, it was +1 in the upper index, but changed to +2 for the np errs

            lb = i_skp
            ub = i + i_skp - e_skp

            # But we don't want to go too far out of the constant compliance region, so
            # let's just take 50% of the ub.
            ub = int(ub * 0.5)

            x = z_data[lb:ub].astype(float)
            y = y_data[lb:ub].astype(float)

            # TODO
            # add to list

            # Try to fit a straight line to the constant compliance region. If it's not
            # successful, then it's a bad force curve.

            try:
                m, b = np.polyfit(x, y, 1)
                if (
                    not flip
                ):  # only want to append it once, not both times for app and ret
                    AFM.fc.append(self.nam)
            except:
                if not flip:
                    AFM.not_fc.append(self.nam)
                self.stop = True
                self.bad = True
                return x, y

            z_cal = abs(z_data - ((y_data - b.astype(float)) / m.astype(float)))

            if plot:
                ax[0].plot(z_data, y_data)
                ax[0].plot(x, m * x + b)
                ax[1].plot(z_cal, y_data * self.SprCons)
                ax[0].set_xlabel("Distance (nm)")
                ax[1].set_xlabel("Distance (nm)")
                ax[0].set_ylabel("Deflection (nN)")
                fig.tight_layout()
                # fig.savefig('plots/' + self.d + self.con + '_fc_all.png')

            # if self.plot:
            #     ax1.plot(z_cal, y_data * self.SprCons)
            #     fig1.tight_layout()
            #     fig1.savefig('plots/' + self.d + self.con + '_fc.png')

            x = z_cal
            y = y_data * self.SprCons

            self.data[idx]["z_cal_nm"] = x
            self.data[idx]["defl_cal_nN"] = y

            flip = True

        return

    # TODO fix
    # def plotting(self, plot = False):
    #     """
    #     Function which dictates if a plot is made.

    #     """
    #     self.plot = plot

    def final_pulloff(self):
        """
        This snazzy function determines the indicies of the final pull-off location,
            which is used in the WLC model analysis.

        The function first walks from *right* to *left* on a normal force curve (note
            the np.flip here), and finds the location where the next y value drops
            to below -0.025 AND the difference between y[i] and y[i+1] is > 0.06.
            These parameters (-0.025 and 0.06) were optimised for this data set.
            This i represents the y_start location.
            N.B. if the function doesn't find a trough/dip, then there's no
            pull-off event and we won't use it.

        After finding y_start we then find y_end - this is trickier. Many methods were
            tried, but the best method was using a rolling-average of 'gradients' (note
            that here we're actually using the negative gradient - could fix). Gradients
            are calculated between y[i] and y[i+1], y[i+1] and y[i+2], etc, and stored in
            a np.array. The average is then calculated between these gradients. If the
            average gradient is negative, then it's tipped over, and we've reached another
            dip, representing the end of the pull-off event. To triple check this, the
            2nd last gradient must be less than the last gradient. We then move back 4 spots
            and this is our y_end value. The size of the np.array (or number of gradients to
            use) was optimised to 16.

        Note: That what constitutes a bad fit:
            - if force curve doesnt snap back to baseline
            - if the 'pull-off' has a gradient with the wrong sign

        Upping 0.06 to 0.065 and -0.025 to -0.03 seems to work for optimising for new datasets

        """
        y = np.array(self.data[1]["defl_cal_nN"])
        x = np.array(self.data[1]["z_cal_nm"])

        yi = np.flip(y)
        xi = np.flip(x)

        for idx, f in enumerate(yi):
            if idx == len(yi) - 1:
                print("bad bad")
                self.stop = True
                self.bad = True
                return

            if (f < -0.03) & (abs(f) - yi[idx + 1] > 0.065):
                # Change the -1 here to change where the end position is
                y_start = len(y) - idx - 1
                self.xmax = xi[idx]
                break

        yi = np.flip(np.array(y[: len(y) - idx]))

        # ROLLING AVERAGE

        yn = np.flip(y[:y_start])
        xn = np.flip(x[:y_start])
        grad_av = []

        # TODO Improve this - do not need to calculate every gradient every iteration
        # in the for loop! Just need to move indicies.
        for i in range(len(yn)):
            grads = np.zeros(16)
            for s in range(len(grads)):
                grads[s] = (yn[i + s + 1] - yn[i + s]) / (
                    xn[i + s] - xn[i + s + 1]
                )

            av = statistics.mean(grads)
            grad_av.append(av)
            # N.B. Conditions here for grads[-4:-1] < grads[0] needs to be up to -4,
            # otherwise y_end could be premature as it rolls over too soon
            if (
                (av < 0)
                & (grads[-1] < grads[0])
                & (grads[-2] < grads[0])
                & (grads[-3] < grads[0])
                & (grads[-4] < grads[0])
            ):
                # Change the +4 here to change where the start position is
                y_end = y_start - i - len(grads) + 4

                if abs(np.mean(y[y_start : y_start + 4])) > 0.05:
                    print(
                        f"didnt snap back to baseline: {y[y_start]:.03f}",
                        self.f,
                    )
                    self.bad = True
                    # self.stop = True
                elif y[y_start] - y[y_end] > 0:
                    # TODO clever way for this
                    # We want to move y_start across if ystart > yend. Should iterate, not just a static move.
                    # while y[y_start] - y[y_end] > 0:
                    y_start -= 2

                break

        self.y_start = y_start
        self.y_end = y_end

    def wlc(x, L_p, L_c, T):
        return ((k_B * T) / L_p) * (
            (x / L_c) - (1 / (4 * (1 - (x / L_c)) ** 2)) - (1 / 4)
        )

    def resid(params, x, ydata):
        """
        Determines the residuals between the WLC model and data.
        """

        L_p = params["L_p"].value
        L_c = params["L_c"].value
        T = params["T"].value

        ymodel = AFM.wlc(x, L_p, L_c, T)
        return ymodel - ydata

    def auto_params(self, L_p=0.8e-9, L_c=300e-9):
        """
        Automatically defines parameters for used by `limfit` for the WLC optimisation.
        """

        # Change this depending on monomer backbone size
        self.Lp_min = (
            0.1e-9  # TODO shouldnt be too much less than monomer legnth...
        )
        self.Lp_max = 3e-9

        # Change these depending on brush thickness
        self.Lc_min = 20e-9
        self.Lc_max = 800e-9

        params = lmfit.Parameters()
        params.add("L_p", L_p, min=self.Lp_min, max=self.Lp_max)
        params.add("L_c", L_c, min=self.Lc_min, max=self.Lc_max)
        params.add("T", self.T, vary=False)

        return params

    def wlc_fit(
        self,
        params,
        plot=True,
        method="differential_evolution",
        show_plot=False,
    ):
        """
        This function fits the Wormlike chain model to the final pull off event
        determined in the `final_pulloff` function.
        """

        # * 1e-9 to convert to SI units
        x_data = self.data[1]["z_cal_nm"][self.y_end : self.y_start] * 1e-9
        y_data = self.data[1]["defl_cal_nN"][self.y_end : self.y_start] * 1e-9

        op = lmfit.minimize(
            AFM.resid, params, args=(x_data, y_data), method=method
        )
        self.chisqr = op.chisqr

        x1 = np.array(x_data)[0]
        x2 = np.array(x_data)[-1]
        y1 = AFM.wlc(
            x1,
            op.params["L_p"].value,
            op.params["L_c"].value,
            op.params["T"].value,
        )
        y2 = AFM.wlc(
            x2,
            op.params["L_p"].value,
            op.params["L_c"].value,
            op.params["T"].value,
        )

        if y2 - y1 > 0:
            print("wrong way", self.f)
            self.bad = True
            self.stop = True

        # TODO modify this for optmising
        if self.chisqr > self.chi_tol:
            self.bad = True

        path = self.f
        fh = path.split("/")[-1]
        try:
            fh = fh.replace(".", "_")
        except:
            print("maybe a weird file")

        # TODO fix this plot - save it and label. use self.plot?
        if plot:
            fig, ax = plt.subplots()
            ax.plot(
                x_data.values * 1e9,
                np.array(
                    AFM.wlc(
                        x_data,
                        op.params["L_p"].value,
                        op.params["L_c"].value,
                        op.params["T"].value,
                    )
                )
                * 1e9,
            )

            x = self.data[1]["z_cal_nm"][
                self.y_end - 100 : self.y_start + 100
            ].values
            y = self.data[1]["defl_cal_nN"][
                self.y_end - 100 : self.y_start + 100
            ].values
            ax.plot(x, y)

            try:
                ax.set_ylim(1.2 * min(y), 0.1)
            except:
                ax.set_ylim(-5, 0.1)
            ax.set_ylabel("Deflection (nN)")
            ax.set_xlabel("Distance (nm)")
            ax.axhline(y=0, c="r", linestyle="--")
            ax.set_title(fh)
            fig.tight_layout()
            plot_path = pathlib.Path("processed_data", fh[:-4] + "_wlc.png")
            fig.savefig(plot_path, dpi=300)

            if not show_plot:
                plt.cla()
                plt.close(fig)

        self.L_p = op.params["L_p"].value
        self.L_c = op.params["L_c"].value

    def molecular_weight(self):
        """
        Calculates the polymer molecular weight based on the calulated contour length

        Parameters:
        _____________
        MM : float, molecular mass of monomer in g/mol
        a : float, length of monomer unit (i.e. width of side chain) in m

        Returns:
        _____________
        molecular_weight : Polymer molecular weight in g/mol
        """

        MW = self.MM * self.L_c / self.a
        self.MW = MW
        return MW

    def grafting_density(self):
        """
        Calculates the grafting density of the polymer brush in m

        Parameters:
        _____________
        rho : float, Density of polymer in g/m3
        h_dry : float, Dry thickness of the polymer brush in m

        Returns:
        _____________
        grafting_density : Polymer brush grafting density
        """

        sigma = self.h_dry * self.rho * N_a / self.MW
        # self.sigma2 = self.sigma * (1e-9)**2

        # 1 chain per ___ nm
        # print(1/np.sqrt(sigma)*1e9)

        return sigma

    def radius_gyration(self):
        """
        Calculates the radius of gyration of the brush in m

        Parameters:
        _____________
        a : float, length of monomer unit (i.e. width of side chain) in m

        Returns:
        _____________
        radius_gyration : Polymer radius of gyration in m
        """

        R_g = np.sqrt(
            (25 / 176) * (self.a**2) * (self.L_c / self.a) ** (6 / 5)
        )

        return R_g

    def reduced_grafting_density(self):
        """
        Calculates the reduced grafting density
        """

        Sigma = np.pi * self.sigma * self.R_g**2
        return Sigma

    def summary_stats(self, prt=False):
        self.MW = AFM.molecular_weight(self)
        self.sigma = AFM.grafting_density(self) * (1e-9) ** 2
        self.R_g = AFM.radius_gyration(self) * 1e9
        self.Sigma = AFM.reduced_grafting_density(self)

        if prt:
            print(
                f"Polymer MW: {self.MW:.2f} g/mol\nGrafting Density: {self.sigma:.4f} chains/m$^2$\nChain density: 1 chain per {1/np.sqrt(self.sigma):.2f} nm\nRadius of gyration: {self.R_g:.2f} nm\nReduced grafting density: {self.Sigma:.2f}"
            )

    def save_data(self, path, plot=True):
        fh = path.split("/")[-1]
        try:
            fh = fh.replace(".", "_")
        except:
            print("maybe a weird file")

        pathA = pathlib.Path("processed_data/")
        pathB = pathlib.Path(fh + ".xlsx")
        out_dir = pathlib.Path(pathA, pathB)

        if out_dir.parent.exists():
            pass
        else:
            print("creating dir")
            pathlib.Path(out_dir.parent).mkdir(parents=True, exist_ok=True)

        writer = pd.ExcelWriter(out_dir, engine="xlsxwriter")
        self.data[0].to_excel(writer, sheet_name="out")
        self.data[1].to_excel(writer, sheet_name="in")
        writer.save()

        # TODO use self.plot? maybe have condiiton on if to show in NB? i.e. plt.close?
        if plot:
            fig, ax = plt.subplots()
            ax.plot(
                self.data[1]["z_cal_nm"].values,
                self.data[1]["defl_cal_nN"].values,
                label="Retract",
            )
            ax.plot(
                self.data[0]["z_cal_nm"].values,
                self.data[0]["defl_cal_nN"].values,
                label="Approach",
            )
            ax.set_ylabel("Force (nN)")
            ax.set_xlabel("Apparent Separation (nm)")
            ax.set_title(fh)
            ax.legend()
            fig.tight_layout()
            plot_path = pathlib.Path("processed_data", fh + ".png")
            fig.savefig(plot_path, dpi=300)
            plt.cla()
            plt.close(fig)

    def output_dir(f_dir):
        pathA = pathlib.Path("processed_data/")
        pathB = pathlib.Path(f_dir[:-4] + ".xlsx")
        out_dir = pathlib.PurePath(pathA, pathB)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        return out_dir

    def potentially_make_wlc(f):
        """
        Check if the `wlc.csv` file exists. If not, make a one with appropriate headings.

        Parameters:
        ___________
        f : string, file name to save WLC summary as

        Returns:
        ________
        wlc_summary : pd.DataFrame, relating to (potentially previous) WLC summary data
        file : string, the file name of the WLC summary file.

        """

        f = pathlib.Path(f)
        file = f"wlc_{f.parent}.csv"

        if not os.path.isfile(file):
            print("making wlc file")
            headers = np.array(
                [
                    "Chain_density",
                    "L_c",
                    "L_p",
                    "MW",
                    "Rg",
                    "Sigma",
                    "bad_fit",
                    "chi_sq",
                    "run_number",
                    "sigma",
                    "xmax",
                ],
                dtype="str",
            )

            np.savetxt(
                file,
                headers.reshape(1, len(headers)),
                fmt="%s",
                delimiter=",",
            )

        wlc_summary = pd.read_csv(file)

        return wlc_summary, file

    def batch_process(
        self, files, update=False, WLC=False, method="differential_evolution"
    ):
        done = False
        for i, f in zip(tqdm(range(len(files))), files):
            self.bad = False
            if update:
                print(f)
            # while not AFM.stop:
            AFM.load_data(self, f)
            AFM.baseline_corr(self)

            if not self.stop:
                AFM.x_cal(self)
            if not self.stop:
                AFM.save_data(self, f[:-4])

            if WLC:
                wlc_summary, fh = AFM.potentially_make_wlc(f)
                if not self.stop:
                    AFM.final_pulloff(self)

                if not self.stop:
                    AFM.wlc_fit(self, AFM.auto_params(self), method=method)

                    tol_ub = 0.9
                    tol_lb = 1.1

                    # if (not done) & (self.L_p > tol * self.Lp_max) or (self.L_p < tol * self.Lp_min):
                    #     print('attempting to improve fit')
                    #     AFM.wlc_fit(self, AFM.auto_params(self), method = method)

                    #     if (self.L_p > tol * self.Lp_max) or (self.L_p < tol * self.Lp_min):
                    #         print('refitting didnt help')
                    #         self.bad = True

                    # elif done:
                    #     self.bad = True

                    # if (not done) & (self.L_c > tol * self.Lc_max) or (self.L_c < tol * self.Lc_min):
                    #     print('attempting to improve fit')
                    #     AFM.wlc_fit(self, AFM.auto_params(self), method = method)

                    #     if (self.L_c > tol * self.Lc_max) or (self.L_c < tol * self.Lc_min):
                    #         print('refitting didnt help')
                    #         self.bad = True

                    # elif done:
                    #     self.bad = True

                    if (self.L_p > tol_ub * self.Lp_max) or (
                        self.L_p < tol_lb * self.Lp_min
                    ):
                        self.bad = True

                    if (self.L_c > tol_ub * self.Lc_max) or (
                        self.L_c < tol_lb * self.Lc_min
                    ):
                        self.bad = True

                    # We don't want contour length less than than pull-off distance (in nm)
                    if self.L_c < 0.75 * self.xmax * 1e-9:
                        self.bad = True

                    AFM.summary_stats(self)

                    to_append = {
                        "run_number": self.f[:-4],
                        "bad_fit": self.bad,
                        "L_p": self.L_p,
                        "L_c": self.L_c,
                        "xmax": self.xmax,
                        "chi_sq": self.chisqr,
                        "MW": self.MW,
                        "sigma": self.sigma,
                        "Rg": self.R_g,
                        "Sigma": self.Sigma,
                        "Chain_density": 1
                        / np.sqrt(self.sigma),  # : 1 chain per ___ nm
                    }

                    wlc_summary = wlc_summary.append(
                        to_append, ignore_index=True
                    )

            wlc_summary.to_csv(fh, index=False)

        print(
            f"{len(AFM.fc)} curves processed and {len(AFM.not_fc)} curves discarded."
        )

    def plot_raw(self):
        fig, ax = plt.subplots()
        ax.plot(
            self.data[1]["z_nm"].values,
            self.data[1]["defl_V"].values,
            label="Retract",
        )
        ax.plot(
            self.data[0]["z_nm"].values,
            self.data[0]["defl_V"].values,
            label="Approach",
        )
        ax.set_ylabel("Deflection (V)")
        ax.set_xlabel("Distance (nm)")
        ax.legend()
        fig.tight_layout()
        # plt.cla()
        # plt.close(fig)

        # TODO
        # self.bad
        # make sure it has at least x data points (20?)
        # fix as you go plotting
        # list of bullet points as to why self.bad = True
        # be consistent with all units

        # TODO
        # Fix T and self.T so wlc can access T
