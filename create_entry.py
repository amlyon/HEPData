from __future__ import print_function
print('-> import hepdata_lib')
import hepdata_lib
from hepdata_lib import Submission
from hepdata_lib import Table
from hepdata_lib import Variable
from hepdata_lib import Uncertainty
from hepdata_lib import RootFileReader
from hepdata_lib.c_file_reader import CFileReader
import numpy as np
import importlib
import random
import sys
sys.path.append('./inputs')
print('-> successful import')


class TernaryPlot(object):
    def __init__(self, mass, mass_str, scenario, scenario_name, figure_name, position):
        self.mass = mass
        self.mass_str = mass_str
        self.scenario = scenario
        self.scenario_name = scenario_name
        self.figure_name = figure_name
        self.position = position

    def print(self):
        print('Ternary Plot - {} GeV, {}'.format(self.mass, self.scenario_name))


class LimitPlot(object):
    def __init__(self, scenario, scenario_name, re, ru, rt, re_str, ru_str, rt_str, figure_number, figure_name, position):
        self.scenario = scenario
        self.scenario_name = scenario_name
        self.re = re
        self.ru = ru
        self.rt = rt
        self.re_str = re_str
        self.ru_str = ru_str
        self.rt_str = rt_str
        self.figure_number = figure_number
        self.figure_name = figure_name
        self.position = position

    def print(self):
        print('2D Limit Plot - {}, (re, ru, rt) = ({}, {}, {})'.format(self.scenario_name, self.re, self.ru, self.rt))


class pNNPlot(object):
    def __init__(self, quantity_name, units, position, figure_name, rootfile):
        self.quantity_name = quantity_name
        self.units = units
        self.position = position
        self.figure_name = figure_name
        self.rootfile = rootfile

    def print(self):
        print('-> pNN feature distribution - {}'.format(self.quantity_name))


class PrefitPlot(object):
    def __init__(self, figure_name, input_data):
        self.figure_name = figure_name
        self.input_data = input_data

    def print(self):
        print('-> Prefit Plot')


class PostfitPlot(object):
    def __init__(self, mass, input_data, figure_name, position):
        self.mass = mass
        self.input_data = input_data
        self.figure_name = figure_name
        self.position = position

    def print(self):
        print('-> Postfit Plot - {} GeV'.format(self.mass))


class HEPDataEntryCreator(object):
    def __init__(self, pNN_plots, prefit_plots, postfit_plots, limit_plots, ternary_plots):
        self.pNN_plots = pNN_plots
        self.prefit_plots = prefit_plots
        self.postfit_plots = postfit_plots
        self.limit_plots = limit_plots
        self.ternary_plots = ternary_plots


    def prepare_pNN_plot_table(self, pNN_plot):
        pNN_plot.print()

        table_name = 'pNN feature - {}'.format(pNN_plot.quantity_name)
        table = Table(table_name)

        table.description = 'The distribution of {qte} is shown for data, as well as for a signal hypothesis of {mN} = 2 GeV and $c\\tau$ =100 mm. The data correspond to an integrated luminosity of 5.2 {fb} and are selected in the mass window of size $10\sigma$ around {mN} = 2 GeV. The distributions, which are normalized to unit area, are shown for the dimuon channel in the category with high {lxysig}, OS, and low {bmass} mass.'.format(qte=pNN_plot.quantity_name, mN='$m_\mathrm{N}$', fb='fb$^{-1}$', lxysig='$L_{xy}/\sigma_{L_{xy}}$', bmass='$\ell_\mathrm{B}\ell^\pm\pi^\mp$')

        table.location = 'Data from Figure 3 ({})'.format(pNN_plot.position)

        reader = RootFileReader('inputs/{}'.format(pNN_plot.rootfile))
        data = reader.read_hist_1d('hist_bkg0p0')
        signal = reader.read_hist_1d('hist_sig0p0')
        #print(data.keys())

        quantity = Variable('{}'.format(pNN_plot.quantity_name), is_independent=True, is_binned=False, units=pNN_plot.units)
        quantity.values = data['x']

        data_entries = Variable('Number of data events', is_independent=False, is_binned=False, units='')
        data_entries.values = data['y']

        signal_entries = Variable('Number of signal events (2 GeV, 100 mm)', is_independent=False, is_binned=False, units='')
        signal_entries.values = signal['y']

        unc_data = Uncertainty('Stat. uncert.', is_symmetric=True)
        unc_data.values = data['dy']
        data_entries.add_uncertainty(unc_data)

        unc_signal = Uncertainty('Stat. uncert.', is_symmetric=True)
        unc_signal.values = signal['dy']
        signal_entries.add_uncertainty(unc_signal)

        table.add_variable(quantity)
        table.add_variable(data_entries)
        table.add_variable(signal_entries)

        table.add_image('inputs/{}'.format(pNN_plot.figure_name))
        #table.add_additional_resource('Data file', 'inputs/2d_limits_{}_{}_{}_{}.txt'.format(limit_plot.scenario, limit_plot.re_str, limit_plot.ru_str, limit_plot.rt_str), copy_file=True)

        return table


    def prepare_prefit_plot_table(self, prefit_plot):
        prefit_plot.print()

        table_name = 'Prefit plot'
        table = Table(table_name)

        table.description = 'Distribution of the $\mu^\pm\pi^\mp$ invariant mass in the mass window around 1.5 GeV in the high {lxysig}, OS, and low {bmass} category in the dimuon channel. The result of the background-only fit to the data is shown together with the mass distribution expected from a Majorana signal with {mN} =1.5 GeV and $c\\tau$ = 500 mm, for the case in which the N mixes with the muon sector only.'.format(mN='$m_\mathrm{N}$', lxysig='$L_{xy}/\sigma_{L_{xy}}$', bmass='$\ell_\mathrm{B}\ell^\pm\pi^\mp$')

        table.location = 'Data from Figure 7'

        # get data
        raw_data = importlib.import_module(prefit_plot.input_data)
        if len(raw_data.y) != len(raw_data.x) or len(raw_data.y_err_down) != len(raw_data.x) or len(raw_data.y_err_up) != len(raw_data.x):
            raise RuntimeError('-> ERROR: lists of different lengths')

        m = Variable('$m(\mu^\pm\pi^\mp)$', is_independent=True, is_binned=False, units='GeV')
        m.values = raw_data.x 

        data = Variable('data', is_independent=False, is_binned=False, units='')
        data.values = raw_data.y

        unc_data = Uncertainty('uncert.', is_symmetric=False)
        unc_data.set_values_from_intervals(zip(-np.array(raw_data.y_err_down)+np.array(raw_data.y), np.array(raw_data.y_err_up)+np.array(raw_data.y)), nominal=data.values)
        data.add_uncertainty(unc_data)

        ## get fits
        #reader = CFileReader('inputs/hnl_mass_m_1p5_ctau_500p0_cat_lxysiggt150_OS_prefit.C')
        #graphs = reader.get_graphs()
        ##print(graphs.keys())

        #background = Variable('background prediction', is_independent=False, is_binned=False, units='')
        #background.values = graphs['pdf_binlxysiggt150_OS_Norm[hnl_mass_muon_channel_m_1p5]_Comp[shapeBkg*]']['y']

        #signal = Variable('signal', is_independent=False, is_binned=False, units='')
        #signal.values = graphs['pdf_binlxysiggt150_OS_Norm[hnl_mass_muon_channel_m_1p5]_Comp[shapeSig*]']['y']

        ## shrink size to that of data
        #x_background = Variable('x (background)', is_independent=True, is_binned=False, units='GeV')
        #x_background.values = graphs['pdf_binlxysiggt150_OS_Norm[hnl_mass_muon_channel_m_1p5]_Comp[shapeBkg*]']['x']
        #indices_background = random.sample(range(0, len(x_background.values)), len(x_background.values)-len(m.values))
        #background.values = np.delete(background.values, indices_background)
        #print(len(background.values))

        ##x_signal = Variable('x (signal)', is_independent=True, is_binned=False, units='GeV')
        ##x_signal.values = graphs['pdf_binlxysiggt150_OS_Norm[hnl_mass_muon_channel_m_1p5]_Comp[shapeSig*]']['x']
        ##indices_signal = random.sample(range(0, len(x_signal.values)), len(x_signal.values)-len(m.values))
        ##signal.values = np.delete(signal.values, indices_signal)

        table.add_variable(m)
        table.add_variable(data)
        #table.add_variable(background)
        #table.add_variable(signal)

        table.add_image('inputs/{}'.format(prefit_plot.figure_name))

        return table


    def prepare_postfit_plot_table(self, postfit_plot):
        postfit_plot.print()

        table_name = 'Postfit plot - {} GeV'.format(postfit_plot.mass)
        table = Table(table_name)

        table.description = 'Mass distribution for a signal of mass {mN} = {m} GeV in the high {lxysig}, OS, and low {bmass} category of the dimuon channel.'.format(mN='$m_\mathrm{N}$', m=postfit_plot.mass, lxysig='$L_{xy}/\sigma_{L_{xy}}$', bmass='$\ell_\mathrm{B}\ell^\pm\pi^\mp$')

        table.location = 'Data from Figure 8 ({})'.format(postfit_plot.position)

        raw_data = importlib.import_module(postfit_plot.input_data)

        if len(raw_data.y) != len(raw_data.x) or len(raw_data.y_err_down) != len(raw_data.x) or len(raw_data.y_err_up) != len(raw_data.x):
            raise RuntimeError('-> ERROR: lists of different lengths')

        m = Variable('$m(\mu^\pm\pi^\mp)$', is_independent=True, is_binned=False, units='GeV')
        m.values = raw_data.x 

        data = Variable('data', is_independent=False, is_binned=False, units='')
        data.values = raw_data.y

        unc_data = Uncertainty('uncert.', is_symmetric=False)
        unc_data.set_values_from_intervals(zip(-np.array(raw_data.y_err_down)+np.array(raw_data.y), np.array(raw_data.y_err_up)+np.array(raw_data.y)), nominal=data.values)
        data.add_uncertainty(unc_data)

        table.add_variable(m)
        table.add_variable(data)

        table.add_image('inputs/{}'.format(postfit_plot.figure_name))

        return table


    def prepare_limit_plot_table(self, limit_plot):
        limit_plot.print()

        table_name = 'Limits - {}, (re, ru, rt) = ({}, {}, {})'.format(limit_plot.scenario_name, limit_plot.re, limit_plot.ru, limit_plot.rt) # using scenario instead of scenario_name to respect character limit
        table = Table(table_name)

        table.description = 'Expected and observed 95% CL upper limits on {} as a function of {} for the mixing scenario ($r_e$, $r_\mu$, $r_\\tau$) = ({}, {}, {}) and in the {} scenario.'.format('$|V_\mathrm{N}|^2$', '$m_\mathrm{N}$', limit_plot.re, limit_plot.ru, limit_plot.rt, limit_plot.scenario_name)

        table.location = 'Data from Figure {} ({})'.format(limit_plot.figure_number, limit_plot.position)

        data = np.loadtxt('inputs/2d_limits_{}_{}_{}_{}.txt'.format(limit_plot.scenario, limit_plot.re_str, limit_plot.ru_str, limit_plot.rt_str))

        #print(data)

        m = Variable('$m_\mathrm{N}$', is_independent=True, is_binned=False, units='GeV')
        m.values = data[:,0]

        obs = Variable('observed', is_independent=False, is_binned=False, units='')
        obs.values = data[:,6]
        obs.add_qualifier('Limit', 'Observed')
        obs.add_qualifier('SQRT(S)', 13, 'TeV')
        obs.add_qualifier('LUMINOSITY', 41.6, 'fb$^{-1}$')

        exp = Variable('expected', is_independent=False, is_binned=False, units='')
        exp.values = data[:,3]
        exp.add_qualifier('Limit', 'Expected')
        exp.add_qualifier('SQRT(S)', 13, 'TeV')
        exp.add_qualifier('LUMINOSITY', 41.6, 'fb$^{-1}$')

        unc_1s = Uncertainty("1$\sigma$", is_symmetric=False)
        unc_1s.set_values_from_intervals(zip(data[:,2], data[:,4]), nominal=exp.values)
        exp.add_uncertainty(unc_1s)

        unc_2s = Uncertainty("2$\sigma$", is_symmetric=False)
        unc_2s.set_values_from_intervals(zip(data[:,1], data[:,5]), nominal=exp.values)
        exp.add_uncertainty(unc_2s)

        table.add_variable(m)
        table.add_variable(obs)
        table.add_variable(exp)

        table.add_image('inputs/{}'.format(limit_plot.figure_name))
        table.add_additional_resource('Data file', 'inputs/2d_limits_{}_{}_{}_{}.txt'.format(limit_plot.scenario, limit_plot.re_str, limit_plot.ru_str, limit_plot.rt_str), copy_file=True)

        return table


    def prepare_ternary_plot_table(self, ternary_plot):
        ternary_plot.print()
    
        table_name = 'Ternary Plot - {} GeV, {}'.format(ternary_plot.mass, ternary_plot.scenario_name)
        table = Table(table_name)
    
        table.description = 'Observed 95% CL lower limits on {} as functions of the mixing ratios ($r_e$, $r_\mu$, $r_\\tau$) for a fixed N mass of {} GeV in the {} scenario.'.format('$c\\tau_\mathrm{N}$', ternary_plot.mass, ternary_plot.scenario_name)
    
        table.location = 'Data from Figure 11 ({})'.format(ternary_plot.position)
    
        data = np.loadtxt('inputs/exclusion_{}_m_{}_ctau.txt'.format(ternary_plot.scenario, ternary_plot.mass_str))#, skiprows=2)
    
        #print(data)
    
        re = Variable('$r_{e}$', is_independent=True, is_binned=False, units='')
        re.values = data[:,0]
    
        ru = Variable('$r_{\mu}$', is_independent=True, is_binned=False, units='')
        ru.values = data[:,1]
    
        rt = Variable('$r_{\\tau}$', is_independent=True, is_binned=False, units='')
        rt.values = data[:,2]
    
        obs = Variable('obs', is_independent=False, is_binned=False, units='m')
        obs.values = data[:,3]
        obs.add_qualifier('Limit', 'Observed')
        obs.add_qualifier('SQRT(S)', 13, 'TeV')
        obs.add_qualifier('LUMINOSITY', 41.6, 'fb$^{-1}$')
    
        table.add_variable(re)
        table.add_variable(ru)
        table.add_variable(rt)
        table.add_variable(obs)
    
        table.add_image('inputs/{}'.format(ternary_plot.figure_name))
        table.add_additional_resource('Data file', 'inputs/exclusion_{}_m_{}_ctau.txt'.format(ternary_plot.scenario, ternary_plot.mass_str), copy_file=True)
    
        return table



    def process(self):
        submission = Submission()

        #print ('-> reading abtract')
        #submission.read_abstract("inputs/abstract.txt")
        #submission.add_link("Webpage with all figures and tables", "https://cms-results.web.cern.ch/cms-results/public-results/publications/B2G-16-029/")
        #submission.add_link("arXiv", "http://arxiv.org/abs/arXiv:1802.09407")
        #submission.add_record_id(1657397, "inspire")
        #submission.add_additional_resource("Original abstract file", "example_inputs/abstract.txt", copy_file=True)  # for illustration, probably not useful

        for pNN_plot in pNN_plots:
            pNN_plot_table = self.prepare_pNN_plot_table(pNN_plot=pNN_plot)
            submission.add_table(pNN_plot_table)

        for prefit_plot in prefit_plots:
            prefit_plot_table = self.prepare_prefit_plot_table(prefit_plot=prefit_plot)
            submission.add_table(prefit_plot_table)

        for postfit_plot in postfit_plots:
            postfit_plot_table = self.prepare_postfit_plot_table(postfit_plot=postfit_plot)
            submission.add_table(postfit_plot_table)

        for limit_plot in limit_plots:
            limit_plot_table = self.prepare_limit_plot_table(limit_plot=limit_plot)
            submission.add_table(limit_plot_table)

        for ternary_plot in ternary_plots:
            ternary_plot_table = self.prepare_ternary_plot_table(ternary_plot=ternary_plot)
            submission.add_table(ternary_plot_table)

        #for table in submission.tables:
        #    table.keywords["cmenergies"] = [13000]

        outdir = "output"
        submission.create_files(outdir, remove_old=True)


if __name__ == '__main__':

    ternary_plots = []
    ternary_plots.append(
            TernaryPlot(
                mass = '1.0',
                mass_str = '1p0',
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                figure_name = 'Figure_011-a.pdf',
                position = 'upper left',
                )
            )

    ternary_plots.append(
            TernaryPlot(
                mass = '1.5',
                mass_str = '1p5',
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                figure_name = 'Figure_011-c.pdf',
                position = 'middle left',
                )
            )

    ternary_plots.append(
            TernaryPlot(
                mass = '2.0',
                mass_str = '2p0',
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                figure_name = 'Figure_011-e.pdf',
                position = 'lower left',
                )
            )

    ternary_plots.append(
            TernaryPlot(
                mass = '1.0',
                mass_str = '1p0',
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                figure_name = 'Figure_011-b.pdf',
                position = 'upper right',
                )
            )

    ternary_plots.append(
            TernaryPlot(
                mass = '1.5',
                mass_str = '1p5',
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                figure_name = 'Figure_011-c.pdf',
                position = 'middle right',
                )
            )

    ternary_plots.append(
            TernaryPlot(
                mass = '2.0',
                mass_str = '2p0',
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                figure_name = 'Figure_011-f.pdf',
                position = 'lower right',
                )
            )

    limit_plots = []
    limit_plots.append(
            LimitPlot(
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                re = '0.0',
                ru = '1.0',
                rt = '0.0',
                re_str = '0p0',
                ru_str = '1p0',
                rt_str = '0p0',
                figure_number = '9',
                figure_name = 'Figure_009-a.pdf',
                position = 'upper left',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                re = '0.0',
                ru = '0.5',
                rt = '0.5',
                re_str = '0p0',
                ru_str = '0p5',
                rt_str = '0p5',
                figure_number = '9',
                figure_name = 'Figure_009-b.pdf',
                position = 'upper right',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                re = '0.5',
                ru = '0.5',
                rt = '0.0',
                re_str = '0p5',
                ru_str = '0p5',
                rt_str = '0p0',
                figure_number = '9',
                figure_name = 'Figure_009-c.pdf',
                position = 'lower left',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Majorana',
                scenario_name = 'Majorana',
                re = '0.33',
                ru = '0.33',
                rt = '0.33',
                re_str = '0p3',
                ru_str = '0p3',
                rt_str = '0p3',
                figure_number = '9',
                figure_name = 'Figure_009-d.pdf',
                position = 'lower right',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                re = '0.0',
                ru = '1.0',
                rt = '0.0',
                re_str = '0p0',
                ru_str = '1p0',
                rt_str = '0p0',
                figure_number = '10',
                figure_name = 'Figure_010-a.pdf',
                position = 'upper left',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                re = '0.0',
                ru = '0.5',
                rt = '0.5',
                re_str = '0p0',
                ru_str = '0p5',
                rt_str = '0p5',
                figure_number = '10',
                figure_name = 'Figure_010-b.pdf',
                position = 'upper right',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                re = '0.5',
                ru = '0.5',
                rt = '0.0',
                re_str = '0p5',
                ru_str = '0p5',
                rt_str = '0p0',
                figure_number = '10',
                figure_name = 'Figure_010-c.pdf',
                position = 'lower left',
                )
            )

    limit_plots.append(
            LimitPlot(
                scenario = 'Dirac',
                scenario_name = 'Dirac-like',
                re = '0.33',
                ru = '0.33',
                rt = '0.33',
                re_str = '0p3',
                ru_str = '0p3',
                rt_str = '0p3',
                figure_number = '10',
                figure_name = 'Figure_010-d.pdf',
                position = 'lower right',
                )
            )

    pNN_plots = []
    pNN_plots.append(
            pNNPlot(    
                    quantity_name = '$\pi$ $\mathrm{p}_\mathrm{T}$', 
                    units = 'GeV',
                    position = 'upper left', 
                    figure_name = 'Figure_003-a.pdf',
                    rootfile = 'pi_pt_lxysiggt150_OS_score0p0.root',
                    )
            )

    pNN_plots.append(
            pNNPlot(
                    quantity_name = '$m(\mu_\mathrm{B}\mu^\pm\pi^\mp)$', 
                    units = 'GeV',
                    position = 'upper right', 
                    figure_name = 'Figure_003-b.pdf',
                    rootfile = 'b_mass_lxysiggt150_OS_score0p0.root',
                    )
            )
    pNN_plots.append(
            pNNPlot(
                    quantity_name = '$\cos \\theta$', 
                    units = '',
                    position = 'lower left', 
                    figure_name = 'Figure_003-c.pdf',
                    rootfile = 'hnl_cos2d_lxysiggt150_OS_score0p0.root',
                    )
            )

    pNN_plots.append(
            pNNPlot(
                    quantity_name = '$\pi$ $\mathrm{d}_{xy}|\sigma_{\mathrm{d}_{xy}}$', 
                    units = '',
                    position = 'lower right', 
                    figure_name = 'Figure_003-d.pdf',
                    rootfile = 'pi_dcasig_lxysiggt150_OS_score0p0.root',
                    )
            )

    prefit_plots = []
    prefit_plots.append(
            PrefitPlot(
                figure_name = 'Figure_007.pdf',
                input_data = 'prefit_plot_data',
                )
            )

    postfit_plots = []
    postfit_plots.append(
            PostfitPlot(
                mass = '1.0',
                input_data = 'postfit_plot_data_m_1p0',
                figure_name = 'Figure_008-a.pdf',
                position = 'upper left',
                )
            )

    postfit_plots.append(
            PostfitPlot(
                mass = '1.5',
                input_data = 'postfit_plot_data_m_1p5',
                figure_name = 'Figure_008-b.pdf',
                position = 'upper right',
                )
            )

    postfit_plots.append(
            PostfitPlot(
                mass = '2.0',
                input_data = 'postfit_plot_data_m_2p0',
                figure_name = 'Figure_008-c.pdf',
                position = 'lower left',
                )
            )

    postfit_plots.append(
            PostfitPlot(
                mass = '2.5',
                input_data = 'postfit_plot_data_m_2p5',
                figure_name = 'Figure_008-d.pdf',
                position = 'lower right',
                )
            )

    creator = HEPDataEntryCreator(
                pNN_plots = pNN_plots,
                prefit_plots = prefit_plots,
                postfit_plots = postfit_plots,
                limit_plots = limit_plots, 
                ternary_plots = ternary_plots,
            )

    creator.process()
