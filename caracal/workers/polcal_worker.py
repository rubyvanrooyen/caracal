# -*- coding: future_fstrings -*-
import stimela
import pickle
import sys
import os
import caracal.dispatch_crew.utils as utils
import caracal
import yaml
import stimela.dismissable as sdm
from caracal.workers.utils import manage_flagsets as manflags
from caracal.workers.utils import manage_fields as manfields
from caracal.workers.utils import manage_caltabs as manGtabs
import copy
import re
import json
import glob
import glob

import shutil
import numpy
from casacore.tables import table as tb

NAME = "Polarization calibration"
LABEL = 'polcal'


def get_dir_path(string, pipeline):
    return string.split(pipeline.output)[1][1:]


def exists(outdir, path):
    _path = os.path.join(outdir, path)
    return os.path.exists(_path)


def scan_length(msinfo, field):
    idx = utils.get_field_id(msinfo, field)[0]
    return float(utils.field_observation_length(msinfo, field)) / len(msinfo['SCAN'][str(idx)])


def xcal_model_method(msname, msinfo, recipe, config, pipeline, i, prefix, polarized_calibrators, caltablelist, gainfieldlist,
            interplist, calwtlist):
    gaintables = [prefix + '.Gpol1', prefix + '.Kcrs', prefix + '.Xref', prefix + '.Xf', prefix + '.Dref',
                  prefix + '.Df']
    interps = ['linear', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest']
    fields = ['', '', '', '', '', '']
    calwt = [True, False, False, False, False, False]

    all_gaintables = caltablelist + gaintables
    all_interp = interplist + interps
    all_fields = gainfieldlist + fields
    all_calwt = calwtlist + calwt

    ref = pipeline.refant[i] or '0'
    field = ",".join(getattr(pipeline, config["pol_calib"])[i])
    leak_field = ",".join(getattr(pipeline, config["leakage_calib"])[i])
    avgstring = ',' + config["avg_bw"]
    scandur = scan_length(msinfo, field)
    scandur_leak = scan_length(msinfo, leak_field)
    if config['plotgains']:
        plotdir = os.path.join(pipeline.diagnostic_plots, "polcal")
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)

    docal = config['reuse_existing_tables']
    if docal:
        for cal in gaintables:
            if not os.path.exists(os.path.join(pipeline.caltables, cal)):
                caracal.log.info("No polcal table found in %s" % str(os.path.join(pipeline.caltables, cal)))
                docal = False

    if not docal:
        recipe.add("cab/casa_setjy",
                   "set_model_%d" % 0,
                   {
                       "msname": msname,
                       "usescratch": True,
                       "field": field,
                       "standard": polarized_calibrators[field]["standard"],
                       "fluxdensity": polarized_calibrators[field]["fluxdensity"],
                       "spix": polarized_calibrators[field]["spix"],
                       "reffreq": polarized_calibrators[field]["reffreq"],
                       "polindex": polarized_calibrators[field]["polindex"],
                       "polangle": polarized_calibrators[field]["polangle"],
                   },
                   input=pipeline.input, output=pipeline.output,
                   label="set_model_%d" % 0)

        gain_opts ={
            "vis": msname,
            "caltable": prefix + '.Gpol1:output',
            "field": field,
            "uvrange": config["uvrange"],
            "refant": ref,
            "solint": scandur,
            "combine": "",
            "parang": True,
            "gaintype": "G",
            "calmode": "p",
            "spw": '',
        }
        if caltablelist != []:
            gain_opts.update({
                "gaintable": ["%s:output" % ct for ct in caltablelist],
                "gainfield": gainfieldlist,#["%s" % ct for ct in gainfieldlist],
                "interp": interplist,#["%s" % ct for ct in interplist],
            })
        # Phaseup diagonal of crosshand cal if available
        recipe.add("cab/casa_gaincal",
                   "gain_xcal",
                   gain_opts,
                   input=pipeline.input, output=pipeline.caltables,
                   label="gain_xcal")

        recipe.add("cab/casa_gaincal",
                   "crosshand_delay",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Kcrs:output',
                       "field": field,
                       "uvrange": config["uvrange"],
                       "refant": ref,
                       "solint": scandur,
                       "combine": "",
                       "parang": True,
                       "gaintype": "KCROSS",
                       "spw": '',
                       "gaintable": ["%s:output" % ct for ct in set(caltablelist + [prefix + '.Gpol1'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + [''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['linear'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="crosshand_delay")

        # Solve for the absolute angle (phase) between the feeds
        # (P Jones auto enabled)
        # of the form [e^{2pi.i.b} 0 0 1]

        # remove the DC of the frequency solutions before
        # possibly joining scans to solve for per-frequency solutions
        # a strongly polarized source is needed with known properties
        # to limit the amount of PA coverage needed
        recipe.add("cab/casa_polcal",
                   "crosshand_phase_ref",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Xref:output',
                       "field": field,
                       "uvrange": config["uvrange"],
                       "solint": scandur,
                       "combine": "",
                       "poltype": "Xf",
                       "refant": ref,
                       "gaintable": ["%s:output" % ct for ct in set(caltablelist + [prefix + '.Gpol1', prefix + '.Kcrs'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + ['',''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['linear','nearest'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="crosshand_phase_ref")

        recipe.add("cab/casa_polcal",
                   "crosshand_phase_freq",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Xf:output',
                       "field": field,
                       "uvrange": config["uvrange"],
                       "solint": 'inf' + avgstring,  # solint to obtain SNR on solutions
                       "combine": "scan",
                       "poltype": "Xf",
                       "refant": ref,
                       "gaintable": ["%s:output" % ct for ct in
                                     set(caltablelist + [prefix + '.Gpol1', prefix + '.Kcrs', prefix + '.Xref'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + ['', '', ''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['linear', 'nearest', 'nearest'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="crosshand_phase_freq")

        # Solve for leakages (off-diagonal terms) using the unpolarized source
        # - first remove the DC of the frequency response and combine scans
        # if necessary to achieve desired SNR
        recipe.add("cab/casa_polcal",
                   "leakage_ref",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Dref:output',
                       "field": leak_field,
                       "uvrange": config["uvrange"],
                       "solint": scandur_leak,
                       "combine": "",
                       "poltype": "D",
                       "refant": ref,
                       "spw": '',
                       "gaintable": ["%s:output" % ct for ct in
                                     set(caltablelist + [prefix + '.Kcrs', prefix + '.Xref', prefix + '.Xf'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + ['', '', ''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['nearest', 'nearest', 'nearest'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="leakage_ref")

        recipe.add("cab/casa_polcal",
                   "leakage_freq",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Df:output',
                       "spw": '',
                       "field": leak_field,
                       "uvrange": config["uvrange"],
                       "solint": 'inf' + avgstring,
                       "combine": "scan",
                       "poltype": "Df",
                       "refant": ref,
                       "gaintable": ["%s:output" % ct for ct in
                                     set(caltablelist + [prefix + '.Kcrs', prefix + '.Xref', prefix + '.Xf', prefix + '.Dref'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + ['', '', '', ''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['nearest', 'nearest', 'nearest', 'nearest'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="leakage_freq")

        applycal_recipes = []
        calmodes = []

        for ix, gt in enumerate(gaintables):
            applycal_recipes.append(dict(zip(
                ['caltable', 'fldmap', 'interp', 'calwt'], [gt, fields[ix], interps[ix], bool(calwt[ix])])))
            if '.Gpol1' in gt:
                calmodes.append('xcal_gain')
            elif '.Kcrs' in gt:
                calmodes.append('cross_phase')
            elif '.Xref' in gt:
                calmodes.append('phase ref')
            elif '.Xf' in gt:
                calmodes.append('phase')
            elif '.Dref' in gt:
                calmodes.append('leakage_ref')
            elif '.Df' in gt:
                calmodes.append('leakage_freq')
        callib_dir = "{}/callibs".format(pipeline.caltables)
        if not os.path.exists(callib_dir):
            os.mkdir(callib_dir)

        callib_dict = dict(zip(calmodes, applycal_recipes))

        with open(os.path.join(callib_dir, f'callib_{prefix}_xcal.json'), 'w') as json_file:
            json.dump(callib_dict, json_file)

        target_callib_dict = callib_dict
        del target_callib_dict['xcal_gain']
        with open(os.path.join(callib_dir, f'callib_{prefix}.json'), 'w') as json_file:
            json.dump(target_callib_dict, json_file)

        if config['plotgains']:
            gfields = [field, field, field, field, leak_field, leak_field]
            terms = ['G', 'KCROSS', 'Xref', 'Xf', 'Dref', 'Df']
            for ix, gt in enumerate(gfields):
                plotgains(recipe, pipeline, plotdir, gfields[ix], gaintables[ix], i, terms[ix])
    else:
        caracal.log.info("Reusing existing tables as requested")

    if config['apply_pcal']:
        for fld in config["applyto"]:
            f = ",".join(getattr(pipeline, fld)[i])
            pcal = ",".join(getattr(pipeline, config["pol_calib"])[i])
            if f == pcal:
                recipe.add("cab/casa_applycal", "apply_caltables_"+fld, {
                    "vis": msname,
                    "field": f,
                    "calwt": all_calwt,
                    "gaintable": ["%s:output" % ct for ct in all_gaintables],
                    "gainfield": ["%s" % ct for ct in all_fields],
                    "interp": ["%s" % ct for ct in all_interp],
                    "parang": True,
                },
                   input=pipeline.input, output=pipeline.caltables,
                   label="Apply caltables on "+fld)
            else:
                cc=calwtlist+calwt[1:]
                ccal=caltablelist+gaintables[1:]
                cfield=gainfieldlist+fields[1:]
                cinter=interplist+interps[1:]
                recipe.add("cab/casa_applycal", "apply_caltables_"+fld, {
                    "vis": msname,
                    "field": f,
                    "calwt": cc,
                    "gaintable": ["%s:output" % ct for ct in ccal],
                    "gainfield": ["%s" % ct for ct in cfield],
                    "interp": ["%s" % ct for ct in cinter],
                    "parang": True,
                },
                           input=pipeline.input, output=pipeline.caltables,
                           label="Apply caltables on "+fld)


def pa_coverage_method(msname, msinfo, recipe, config, pipeline, i, prefix, caltablelist, gainfieldlist, interplist, calwtlist):
    gaintables = [prefix + '.Gxyamp', prefix + '.Kcrs', prefix + '.Xfparang', prefix + '.Df0gen']
    interps = ['linear', 'nearest', 'nearest', 'nearest']
    fields = ['', '', '', '']
    calwt = [True, False, False, False]

    all_gaintables = caltablelist + gaintables
    all_interp = interplist + interps
    all_fields = gainfieldlist + fields
    all_calwt = calwtlist + calwt

    ref = pipeline.refant[i] or '0'
    field = ",".join(getattr(pipeline, config["pol_calib"])[i])
    avgstring = ',' + config["avg_bw"]
    scandur = scan_length(msinfo, field)
    if config['plotgains']:
        plotdir = os.path.join(pipeline.diagnostic_plots, "polcal")
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)

    docal = config['reuse_existing_tables']
    if docal:
        for cal in gaintables:
            if not os.path.exists(os.path.join(pipeline.caltables, cal)):
                caracal.log.info("No polcal table found in %s" % str(os.path.join(pipeline.caltables, cal)))
                docal = False

    if not docal:
        # G1
        recipe.add("cab/casa_gaincal",
                   "gain_xcal_1",
                   {
                       "vis": msname,
                       "field": field,
                       "caltable": prefix + '.Gpol1:output',
                       "smodel": ['1', '0', '0', '0'],
                       "refantmode": 'strict',
                       "refant": ref,
                       "gaintype": 'G',
                       "calmode": 'ap',
                       "parang": False,
                       "solint": 'int',
                       "gaintable": ["%s:output" % ct for ct in caltablelist],
                       "gainfield": ["%s" % ct for ct in gainfieldlist],
                       "interp": ["%s" % ct for ct in interplist],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="gain_xcal_1")

        shutil.rmtree(os.path.join(pipeline.caltables, prefix + '.Gpol1a'), ignore_errors=True)
        # QU
        recipe.add("cab/casa_polfromgain",
                   "QU_from_gain",
                   {
                       "vis": msname,
                       "tablein": prefix + '.Gpol1:output',
                       "caltable": prefix + '.Gpol1a:output',
                       "save_result": prefix + '_S1_from_QUfit:output',
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="QU_from_gain")
        recipe.run()
        recipe.jobs = []

        # We search for the scan where the polarization signal is minimum in XX and YY
        # (i.e., maximum in XY and YX):
        with tb(os.path.join(pipeline.caltables, prefix + '.Gpol1')) as t:
            scans = t.getcol('SCAN_NUMBER')
            gains = numpy.squeeze(t.getcol('CPARAM'))
            t.close()
        scanlist = numpy.array(list(set(scans)))
        ratios = numpy.zeros(len(scanlist))

        for si, s in enumerate(scanlist):
            filt = scans == s
            ratio = numpy.sqrt(
                numpy.average(numpy.power(numpy.abs(gains[filt, 0]) / numpy.abs(gains[filt, 1]) - 1.0, 2.)))
            ratios[si] = ratio

        bestscidx = numpy.argmin(ratios)
        bestscan = scanlist[bestscidx]
        caracal.log.info('Scan with highest expected X-Y signal: ' + str(bestscan))

        recipe.run()
        recipe.jobs = []

        # Kcross
        recipe.add("cab/casa_gaincal",
                   "crosshand_delay",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Kcrs:output',
                       "selectdata": True,
                       "field": field,
                       "scan": str(bestscan),
                       "gaintype": 'KCROSS',
                       "solint": 'inf' + avgstring,
                       "refantmode": 'strict',
                       "refant": ref,
                       "smodel": ['1', '0', '1', '0'],
                       "gaintable": ["%s:output" % ct for ct in set(caltablelist + [prefix + '.Gpol1'])],
                       #"gainfield": ["%s" % ct for ct in gainfieldlist],
                       "interp": ["%s" % ct for ct in set(interplist + ['linear'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="crosshand_delay")

        recipe.run()
        recipe.jobs = []

        if os.path.isfile(pipeline.output + '/caltables/' + prefix + '_S1_from_QUfit'):
            with open(pipeline.output + '/caltables/' + prefix + '_S1_from_QUfit', 'rb') as stdr:
                S1 = pickle.load(stdr, encoding='latin1')

            S1 = S1[field]['SpwAve']
            caracal.log.info("First [I,Q,U,V] fitted model (with I=1 and Q, U fractional): %s" % S1)
        else:
            raise RuntimeError("Cannot find S1")  # prefix+'S1_from_QUfit:output'

        # QU abs delay
        recipe.add("cab/casa_polcal",
                   "crosshand_phase_QU_fit",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Xfparang:output',
                       "field": field,
                       "spw": '',
                       "poltype": 'Xfparang+QU',
                       "solint": 'inf' + avgstring,
                       "combine": 'scan,obs',
                       "preavg": scandur,
                       "smodel": S1,
                       "save_result": prefix + '_S2_from_polcal:output',
                       "gaintable": ["%s:output" % ct for ct in set(caltablelist + [prefix + '.Gpol1', prefix + '.Kcrs'])],
                       # "gainfield": ["%s" % ct for ct in gainfieldlist],
                       "interp": ["%s" % ct for ct in set(interplist + ['linear','nearest'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="crosshand_phase_QU_fit")
        recipe.run()
        recipe.jobs = []

        if os.path.isfile(pipeline.output + '/caltables/' + prefix + '_S2_from_polcal'):
            with open(pipeline.output + '/caltables/' + prefix + '_S2_from_polcal', 'rb') as stdr:
                S2 = pickle.load(stdr, encoding='latin1')
            S2 = S2[field]['SpwAve'].tolist()
            caracal.log.info("Second [I,Q,U,V] fitted model (with I=1 and Q, U fractional): %s" % S2)
        else:
            raise RuntimeError("Cannot find " + pipeline.output + "/caltables/" + prefix + "_S2_from_polcal")

        recipe.add("cab/casa_gaincal",
                   "second gaincal",
                   {
                       "vis": msname,
                       "field": field,
                       "caltable": prefix + '.Gpol2:output',
                       "smodel": S2,
                       "refantmode": 'strict',
                       "refant": ref,
                       "gaintype": 'G',
                       "calmode": 'ap',
                       "parang": True,
                       "solint": 'int',
                       "gaintable": ["%s:output" % ct for ct in caltablelist],
                       "gainfield": ["%s" % ct for ct in gainfieldlist],
                       "interp": ["%s" % ct for ct in interplist],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="gain_xcal_2")

        # LEAKAGE
        recipe.add("cab/casa_polcal",
                   "leakage",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Df0gen:output',
                       "field": field,
                       "spw": '',
                       "solint": 'inf' + avgstring,
                       "combine": 'obs,scan',
                       "preavg": scandur,
                       "poltype": 'Dflls',
                       "refant": '',  # solve absolute D-term
                       "smodel": S2,
                       "gaintable": ["%s:output" % ct for ct in set(caltablelist + [prefix + '.Gpol2', prefix + '.Kcrs', prefix + '.Xfparang'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + ['','',''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['linear','nearest','nearest'])],
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="leakage")

        # solve for global normalized gain amp (to get X/Y ratios) on pol calibrator (TO APPLY ON TARGET)
        # amp-only and normalized, so only X/Y amp ratios matter
        recipe.add("cab/casa_gaincal",
                   "norm_gain_for_target",
                   {
                       "vis": msname,
                       "caltable": prefix + '.Gxyamp:output',
                       "field": field,
                       "solint": 'inf',
                       "combine": 'scan,obs',
                       "refant": ref,
                       "refantmode": 'strict',
                       "gaintype": 'G',
                       "smodel": S2,
                       "calmode": 'a',
                       "gaintable": ["%s:output" % ct for ct in set(caltablelist + [prefix + '.Kcrs' , prefix + '.Xfparang' , prefix + '.Df0gen'])],
                       "gainfield": ["%s" % ct for ct in set(gainfieldlist + ['', '', ''])],
                       "interp": ["%s" % ct for ct in set(interplist + ['nearest', 'nearest', 'nearest'])],
                       "solnorm": True,
                       "parang": True,
                   },
                   input=pipeline.input, output=pipeline.caltables,
                   label="norm_gain_for_target")
        recipe.run()
        recipe.jobs = []

        applycal_recipes = []
        calmodes = []
        gaintables = [prefix + '.Gpol2', prefix + '.Kcrs', prefix + '.Xfparang', prefix + '.Df0gen']
        interps = ['linear', 'nearest', 'nearest', 'nearest']
        fields = ['', '', '', '']
        calwt = [True, False, False, False]

        for ix, gt in enumerate(gaintables):
            applycal_recipes.append(dict(zip(
                ['caltable', 'fldmap', 'interp', 'calwt'], [gt, fields[ix], interps[ix], calwt[ix]])))
            if '.Gpol2' in gt:
                calmodes.append('xcal_gain')
            elif '.Kcrs' in gt:
                calmodes.append('cross_phase')
            elif '.Xfparang' in gt:
                calmodes.append('pol_angle')
            elif '.Df0gen' in gt:
                calmodes.append('abs_leakage')
        callib_dir = "{}/callibs".format(pipeline.caltables)
        if not os.path.exists(callib_dir):
            os.mkdir(callib_dir)

        callib_dict = dict(zip(calmodes, applycal_recipes))

        with open(os.path.join(callib_dir, f'callib_{prefix}_xcal.json'), 'w') as json_file:
            json.dump(callib_dict, json_file)

        if config['plotgains']:
            gfields = [field, field, field, field]
            terms = ['G', 'KCROSS', 'Xf', 'Df0gen']
            for ix, gt in enumerate(gfields):
                plotgains(recipe, pipeline, plotdir, gfields[ix], gaintables[ix], i, terms[ix])

        applycal_recipes = []
        calmodes = []
        gaintables = [prefix + '.Gxyamp', prefix + '.Kcrs', prefix + '.Xfparang', prefix + '.Df0gen']
        interps = ['linear', 'nearest', 'nearest', 'nearest']
        fields = ['', '', '', '']
        calwt = [True, False, False, False]

        for ix, gt in enumerate(gaintables):
            applycal_recipes.append(dict(zip(
                ['caltable', 'fldmap', 'interp', 'calwt'], [gt, fields[ix], interps[ix], calwt[ix]])))
            if '.Gxyamp' in gt:
                calmodes.append('cross_gain')
            elif '.Kcrs' in gt:
                calmodes.append('cross_phase')
            elif '.Xfparang' in gt:
                calmodes.append('pol_angle')
            elif '.Df0gen' in gt:
                calmodes.append('abs_leakage')
        callib_dir = "{}/callibs".format(pipeline.caltables)
        if not os.path.exists(callib_dir):
            os.mkdir(callib_dir)

        callib_dict = dict(zip(calmodes, applycal_recipes))

        with open(os.path.join(callib_dir, f'callib_{prefix}.json'), 'w') as json_file:
            json.dump(callib_dict, json_file)

        if config['plotgains']:
            plotgains(recipe, pipeline, plotdir, field, prefix + '.Gxyamp', i, 'G')

    else:
        caracal.log.info("Reusing existing tables as requested")

    if config['apply_pcal']:
        for fld in config["applyto"]:
            f = ",".join(getattr(pipeline, fld)[i])
            pcal = ",".join(getattr(pipeline, config["pol_calib"])[i])
            if f != pcal:
                for n,m in enumerate(all_gaintables):
                    if m == prefix+'.Gpol2':
                        all_gaintables[n] = prefix+'.Gxyamp'
                recipe.add("cab/casa_applycal", "apply_caltables", {
                    "vis": msname,
                    "field": f,
                    "calwt": all_calwt,
                    "gaintable": ["%s:output" % ct for ct in all_gaintables],
                    "gainfield": ["%s" % ct for ct in all_fields],
                    "interp": ["%s" % ct for ct in all_interp],
                    "parang": True,
                },
                   input=pipeline.input, output=pipeline.caltables,
                   label="Apply caltables")
            else:
                for n,m in enumerate(all_gaintables):
                    if m == prefix+'.Gxyamp':
                        all_gaintables[n] = prefix+'.Gpol2'
                recipe.add("cab/casa_applycal", "apply_caltables", {
                        "vis": msname,
                        "field": f,
                        "calwt": all_calwt,
                        "gaintable": ["%s:output" % ct for ct in all_gaintables],
                        "gainfield": ["%s" % ct for ct in all_fields],
                        "interp": ["%s" % ct for ct in all_interp],
                        "parang": True,
                    },
                    input=pipeline.input, output=pipeline.caltables,
                    label="Apply caltables")


def plotgains(recipe, pipeline, plotdir, field_id, gtab, i, term):
    step = "plotgains-%s-%d-%s" % (term, i, gtab)
    recipe.add('cab/ragavi', step,
               {
                   "table": gtab+":msfile",
                   "corr": '',
                   "htmlname": gtab,
                   "field": field_id,
               },
               input=pipeline.input, msdir=pipeline.caltables, output=plotdir,
               label='{0:s}:: Plot gaincal phase'.format(step))


def worker(pipeline, recipe, config):
    label = config["label_cal"]
    label_in = config["label_in"]

    # define pol and unpol calibrators, P&B2017 + updated pol properties from NRAO web site (https://science.nrao.edu/facilities/vla/docs/manuals/obsguide/modes/pol, Table 7.2.7)
    polarized_calibrators = {"3C138": {"standard": "manual",
                                       "fluxdensity": [8.33843],
                                       "spix": [-0.4981, -0.1552, -0.0102, 0.0223],
                                       "reffreq": "1.47GHz",
                                       "polindex": [0.078],
                                       "polangle": [-0.16755]},
                             "3C286": {"standard": "manual",
                                       "fluxdensity": [14.7172],
                                       "spix": [-0.4507, -0.1798, 0.0357],
                                       "reffreq": "1.47GHz",
                                       "polindex": [0.098],
                                       "polangle": [0.575959]},
                             }
    polarized_calibrators["J1331+3030"] = polarized_calibrators["3C286"]
    polarized_calibrators["J0521+1638"] = polarized_calibrators["3C138"]
    unpolarized_calibrators = ["PKS1934-63", "J1939-6342", "J1938-6341", "PKS 1934-638", "PKS 1934-63", "PKS1934-638"]

    # loop over all MSs for this label
    #for i, msbase in enumerate(pipeline.msbasenames):
    for i, (msbase, prefix_msbase) in enumerate(zip(pipeline.msbasenames, pipeline.prefix_msbases)):
        msname = pipeline.form_msname(msbase, label_in)
        msinfo = pipeline.get_msinfo(msname)
        prefix = f"{pipeline.prefix_msbases[i]}-{label}"

        # Check if feeds are linear
        if set(list(msinfo['CORR']['CORR_TYPE'])) & {'XX', 'XY', 'YX', 'YY'} == 0:
            raise RuntimeError(
                "Cannot calibrate polarization! Allowed strategies are for linear feed data but correlation is: " + str(
                    [
                        'XX', 'XY', 'YX', 'YY']))

        # check if cross_callib needs to be applyied
        if config['otfcal']:
            caltablelist, gainfieldlist, interplist, calwtlist = [], [], [], []
            if config['otfcal']['callib']:
                callib = 'caltables/callibs/{}'.format(config['otfcal']['callib'])
                if not os.path.exists(os.path.join(pipeline.output, callib)):
                    raise IOError(
                        "Callib file {0:s} does not exist. Please check that it is where it should be.".format(callib))
                if not os.path.exists(os.path.join('{}/callibs'.format(pipeline.caltables),'{0:s}.json'.format(config['otfcal']['callib'][:-4]))):
                    raise IOError("json version of callib file does not exist. Please provide it.")

                with open(os.path.join('{}/callibs'.format(pipeline.caltables),'{0:s}.json'.format(config['otfcal']['callib'][:-4]))) as f:
                    callib_dict = json.load(f)

                for applyme in callib_dict:
                    caltablelist.append(callib_dict[applyme]['caltable'])
                    gainfieldlist.append(callib_dict[applyme]['fldmap'])
                    interplist.append(callib_dict[applyme]['interp'])
                    calwtlist.append(bool(callib_dict[applyme]['calwt']))

            # write calibration library file for OTF cal
            elif config['otfcal']['label_cal']:
                calprefix = '{0:s}-{1:s}'.format(prefix_msbase,config['otfcal']['label_cal'])

                with open(os.path.join('{}/callibs'.format(pipeline.caltables),'callib_{0:s}.json'.format(calprefix))) as f:
                    callib_dict = json.load(f)

                for applyme in callib_dict:
                    caltablelist.append(callib_dict[applyme]['caltable'])
                    gainfieldlist.append(callib_dict[applyme]['fldmap'])
                    interplist.append(callib_dict[applyme]['interp'])
                    if 'calwt' in callib_dict:
                        calwtlist.append(bool(callib_dict[applyme]['calwt']))
                    else:
                        calwtlist.append(bool(False))

            else:
                caltablelist, gainfieldlist, interplist, calwtlist = [], [], [], []

        # if config['crosscal_callib'] != '' and config['crosscal_callib'][-5:] == '.json':
        #     callib_path = 'caltables/callibs/{}'.format(config['crosscal_callib'])
        #
        #     if not os.path.exists(os.path.join(pipeline.output, callib_path)):
        #         raise RuntimeError("Cannot find cross_cal callib, check crosscal_callib parameter in config file !")
        #
        #     # write calibration library txt file from json file to applycal
        #     caltablelist, gainfieldlist, interplist, calwtlist = [], [], [], []
        #
        #     with open(os.path.join(pipeline.output, callib_path)) as f:
        #         callib_dict = json.load(f)
        #
        #     for applyme in callib_dict:
        #         caltablelist.append(callib_dict[applyme]['caltable'])
        #         gainfieldlist.append(callib_dict[applyme]['fldmap'])
        #         interplist.append(callib_dict[applyme]['interp'])
        #     for k in enumerate(caltablelist):
        #         calwtlist.append(True)
        # elif config['crosscal_callib'] != '' and config['crosscal_callib'][-5:] != '.json':
        #     raise RuntimeError("Cross_cal callib must be a .json file!")
        # else:
        #     caltablelist, gainfieldlist, interplist, calwtlist = [], [], [], []

        # Set -90 deg receptor angle rotation [if we are using MeerKAT data]
        if float(config['feed_angle_rotation']) != '':
            with tb("%s::FEED" % os.path.join(pipeline.msdir, msname), readonly=False) as t:
                ang = t.getcol("RECEPTOR_ANGLE")
                ang[:, 0] = numpy.deg2rad(float(config['feed_angle_rotation']))
                ang[:, 1] = numpy.deg2rad(float(config['feed_angle_rotation']))
                t.putcol("RECEPTOR_ANGLE", ang)
                caracal.log.info('RECEPTOR_ANGLE has been rotated by %s degrees' % config['feed_angle_rotation'])

        pol_calib = ",".join(getattr(pipeline, config["pol_calib"])[i])
        leakage_calib = ",".join(getattr(pipeline, config["leakage_calib"])[i])

        # choose the strategy according to config parameters
        if leakage_calib in set(unpolarized_calibrators):
            if pol_calib in set(polarized_calibrators):
                caracal.log.info(
                    "You decided to calibrate the polarized angle with a polarized calibrator assuming a model for the calibrator and the leakage with an unpolarized calibrator.")
                xcal_model_method(msname, msinfo, recipe, config, pipeline, i, prefix, polarized_calibrators, caltablelist, gainfieldlist, interplist, calwtlist)
            else:
                raise RuntimeError("Unknown pol_calib!"
                                   "Currently only these are known on caracal:\
                                   " + str(polarized_calibrators.keys()) + ". \
                                   You can use one of these source to calibrate polarization \
                                   or if none of them is available you can calibrate both leakage (leakage_calib) and polarization (pol_calib) \
                                   with a source observed at several parallactic angles")
        elif leakage_calib == pol_calib:
            caracal.log.info(
                "You decided to calibrate the polarized angle and leakage with a polarized calibrator without assuming a model for the calibrator.")
            idx = utils.get_field_id(msinfo, leakage_calib)[0]
            if len(msinfo['SCAN'][str(idx)]) >= 3:
                pa_coverage_method(msname, msinfo, recipe, config, pipeline, i,
                           prefix, caltablelist, gainfieldlist, interplist, calwtlist)
            else:
                raise RuntimeError(
                    "Cannot calibrate polarization! Insufficient number of scans for the pol calibrator.")
        else:
            raise RuntimeError("Cannot calibrate polarization! Allowed strategies are: \
                               1. Calibrate leakage with a unpolarized source (i.e. " + str(unpolarized_calibrators) + ") \
                               and polarized angle with a know polarized source (i.e. " + str(polarized_calibrators.keys()) + ") \
                               2. Calibrate both leakage and polarized angle with a (known or unknown) polarized source observed at different parallactic angles.")

        recipe.run()
        recipe.jobs = []
