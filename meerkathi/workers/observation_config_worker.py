import meerkathi.dispatch_crew.utils as utils
import yaml
import meerkathi
import sys
import numpy as np
from os import path

NAME = 'Automatically catergorize observed fields'

def repeat_val(val, n):
    l = []
    for x in range(n):
        l.append(val)
    return l

def worker(pipeline, recipe, config):
    if pipeline.virtconcat:
        msnames = [pipeline.vmsname]
        prefixes = [pipeline.prefix]
        nobs = 1
    else:
        msnames = pipeline.msnames
        prefixes = pipeline.prefixes
        nobs = pipeline.nobs

    for i in range(nobs):
        prefix = prefixes[i]
        msname = msnames[i]
        msroot = msname[:-3]

        if pipeline.enable_task(config, 'obsinfo'):
            if config['obsinfo'].get('listobs'):
                step = 'listobs_{:d}'.format(i)
                recipe.add('cab/casa_listobs', step,
                           {
                               "vis": msname,
                               "listfile": '{0:s}-obsinfo.txt'.format(msroot),
                               "overwrite": True,
                           },
                           input=pipeline.input,
                           output=pipeline.output,
                           label='{0:s}:: Get observation information ms={1:s}'.format(step, msname))

            if config['obsinfo'].get('summary_json'):
                step = 'summary_json_{:d}'.format(i)
                recipe.add('cab/msutils', step,
                           {
                               "msname": msname,
                               "command": 'summary',
                               "display": False,
                               "outfile": '{0:s}-obsinfo.json'.format(msroot),
                           },
                           input=pipeline.input,
                           output=pipeline.output,
                           label='{0:s}:: Get observation information as a json file ms={1:s}'.format(step, msname))

        if config['obsinfo'].get('vampirisms'):
            step = 'vampirisms_{0:d}'.format(i)
            recipe.add('cab/sunblocker', step,
                       {
                           "command": 'vampirisms',
                           "inset": msname,
                           "dryrun": True,
                           "nononsoleil": True,
                           "verb": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='{0:s}:: Note sunrise and sunset'.format(step))

        recipe.run()
        recipe.jobs = []

    # initialse things
    for item in 'xcal fcal bpcal gcal target reference_antenna'.split():
        val = config.get(item)
        for attr in ["", "_ra", "_dec", "_id"]:
            setattr(pipeline, item+attr, repeat_val(val, pipeline.nobs))

    setattr(pipeline, 'nchans', repeat_val(None,pipeline.nobs))
    setattr(pipeline, 'firstchanfreq', repeat_val(None, pipeline.nobs))
    setattr(pipeline, 'lastchanfreq', repeat_val(None, pipeline.nobs))
    setattr(pipeline, 'chanwidth', repeat_val(None, pipeline.nobs))
    setattr(pipeline, 'specframe', repeat_val(None, pipeline.nobs))

    # Set antenna properties
    pipeline.Tsys_eta = config.get('Tsys_eta')
    pipeline.dish_diameter = config.get('dish_diameter')

    for i, prefix in enumerate(prefixes):
        msinfo = '{0:s}/{1:s}-obsinfo.json'.format(pipeline.output, pipeline.dataid[i])
        meerkathi.log.info('Extracting info from {2:s} and (if present, and only for the purpose of automatically setting the reference antenna) the metadata file {0:s}/{1:s}-obsinfo.json'.format(
            pipeline.data_path, pipeline.dataid[i], msinfo))

        # get reference antenna
        if config.get('reference_antenna') == 'auto':
            msmeta = '{0:s}/{1:s}-obsinfo.json'.format(
                pipeline.data_path, pipeline.dataid[i])
            if path.exists(msmeta):
                pipeline.reference_antenna[i] = utils.meerkat_refant(msmeta)
                meerkathi.log.info('Auto selecting reference antenna as {:s}'.format(
                    pipeline.reference_antenna[i]))
            else:
                meerkathi.log.error(
                    'Cannot auto select reference antenna because the metadata file {0:s}, which should have been provided by the observatory, does not exist.'.format(msmeta))
                meerkathi.log.error(
                    'Note that this metadata file is generally available only for MeerKAT-16/ROACH2 data.')
                meerkathi.log.error(
                    'Please set the reference antenna manually in the config file and try again.')
                sys.exit(1)

        # Get channels in MS
        with open(msinfo, 'r') as stdr:
            spw = yaml.safe_load(stdr)['SPW']['NUM_CHAN']
            pipeline.nchans[i] = spw
        meerkathi.log.info('MS has {0:d} spectral windows, with NCHAN={1:s}'.format(
            len(spw), ','.join(map(str, spw))))

        # Get first chan, last chan, chan width
        with open(msinfo, 'r') as stdr:
            chfr = yaml.safe_load(stdr)['SPW']['CHAN_FREQ']
            firstchanfreq = [ss[0] for ss in chfr]
            lastchanfreq = [ss[-1] for ss in chfr]
            chanwidth = [(ss[-1]-ss[0])/(len(ss)-1) for ss in chfr]
            pipeline.firstchanfreq[i] = firstchanfreq
            pipeline.lastchanfreq[i] = lastchanfreq
            pipeline.chanwidth[i] = chanwidth
            meerkathi.log.info('CHAN_FREQ from {0:s} Hz to {1:s} Hz with average channel width of {2:s} Hz'.format(
                ','.join(map(str, firstchanfreq)), ','.join(map(str, lastchanfreq)), ','.join(map(str, chanwidth))))
        if i == len(prefixes)-1 and np.max(pipeline.chanwidth) > 0 and np.min(pipeline.chanwidth) < 0:
            meerkathi.log.info(
                'Some datasets have positive channel increment, some others negative. This will lead to errors. Exiting')
            sys.exit(1)
        # Get spectral frame
        with open(msinfo, 'r') as stdr:
            pipeline.specframe[i] = yaml.safe_load(
                stdr)['SPW']['MEAS_FREQ_REF']

        with open(msinfo, 'r') as stdr:
            targetinfo = yaml.safe_load(stdr)['FIELD']

        intents = utils.categorize_fields(msinfo)
        for term in "fcal bpcal gcal target xcal".split():
            if "auto" in getattr(pipeline, term)[i]:
                label, fields = intents[term]
                if fields in [None, []]:
                    getattr(pipeline, term)[i] = []
                    continue
                getattr(pipeline, term)[i] = fields
                _label = label[0]
            _label = term
            meerkathi.log.info("====================================")
            meerkathi.log.info(_label)
            meerkathi.log.info(" ---------------------------------- ")
            _ra = []
            _dec = []
            _fid = []
            for f in getattr(pipeline, term)[i]:
                fid = utils.get_field_id(msinfo, f)[0]
                targetpos = targetinfo['REFERENCE_DIR'][fid][0]
                ra = targetpos[0]/np.pi*180
                dec = targetpos[1]/np.pi*180
                _ra.append(ra)
                _dec.append(dec)
                _fid.append(fid)
                tobs = utils.field_observation_length(msinfo, f)/60.0
                meerkathi.log.info(
                        '{0:s} (ID={1:d}) : {2:.2f} minutes | RA={3:.2f} deg, Dec={4:.2f} deg'.format(f, fid, tobs, ra, dec))
            getattr(pipeline, term+"_ra")[i] = _ra
            getattr(pipeline, term+"_dec")[i] = _dec
            getattr(pipeline, term+"_id")[i] = _fid

    if pipeline.enable_task(config, 'primary_beam'):
        meerkathi.log.info('Generating primary beam')
        recipe.add('cab/eidos', 'primary_beam',
                   {
                       "diameter": config['primary_beam'].get('diameter'),
                       "pixels": config['primary_beam'].get('pixels'),
                       "freq": config['primary_beam'].get('freq'),
                       "coeff": config['primary_beam'].get('coefficients', 'me'),
                       "prefix": pipeline.prefix,
                       "output-eight": True,
                   },
                   input=pipeline.input,
                   output=pipeline.output,
                   label="generate_primary_beam:: Generate primary beam")

        pipeline.primary_beam = pipeline.prefix + "-$\(xy\)_$\(reim).fits"
        pipeline.primary_beam_l_axis = "X"
        pipeline.primary_beam_m_axis = "Y"
        meerkathi.log.info('Primary beam registered as : \\ Pattern - {0:s}\
                                                         \\ l-axis  - {1:s}\
                                                         \\ m-axis  - {2:s}'.format(pipeline.primary_beam,
                                                                                    pipeline.primary_beam_l_axis,
                                                                                    pipeline.primary_beam_m_axis))
