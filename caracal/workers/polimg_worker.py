import os
import shutil
import glob
import sys
import yaml
import json
import re
import copy
import caracal
import numpy as np
import stimela.dismissable as sdm
from caracal.dispatch_crew import utils
from astropy.io import fits as fits
from stimela.pathformatter import pathformatter as spf
from typing import Any
from caracal.workers.utils import manage_flagsets as manflags
import psutil

NAME = 'Polarization Imaging'
LABEL = 'polimg'

def get_dir_path(string, pipeline):
    return string.split(pipeline.output)[1][1:]

def worker(pipeline, recipe, config):
    wname = pipeline.CURRENT_WORKER
    flags_before_worker = '{0:s}_{1:s}_before'.format(pipeline.prefix, wname)
    flags_after_worker = '{0:s}_{1:s}_after'.format(pipeline.prefix, wname)
    #flag_main_ms = pipeline.enable_task(config, 'calibrate') and config['cal_niter'] >= config['start_iter']
    rewind_main_ms = config['rewind_flags']["enable"] and (config['rewind_flags']['mode'] == 'reset_worker' or config['rewind_flags']["version"] != 'null')
    #rewind_transf_ms = config['rewind_flags']["enable"] and (config['rewind_flags']['mode'] == 'reset_worker' or config['rewind_flags']["transfer_apply_gains_version"] != 'null')
    niter = config['img_niter']
    imgweight = config['img_weight']
    robust = config['img_robust']
    taper = config['img_taper']
    maxuvl = config['img_maxuv_l']
    transuvl = maxuvl*config['img_transuv_l']/100.
    multiscale = config['img_multiscale']
    multiscale_scales = config['img_multiscale_scales']
    if taper == '':
        taper = None

    label = config['label_in']
    min_uvw = config['minuvw_m']
    ncpu = config['ncpu']
    if ncpu == 0:
      ncpu = psutil.cpu_count()
    else:
      ncpu = min(ncpu, psutil.cpu_count())
    nwlayers_factor= config['img_nwlayers_factor']
    nrdeconvsubimg = ncpu if config['img_nrdeconvsubimg'] == 0 else config['img_nrdeconvsubimg']
    if nrdeconvsubimg == 1:
        wscl_parallel_deconv = None
    else:
        wscl_parallel_deconv = int(np.ceil(config['img_npix']/np.sqrt(nrdeconvsubimg)))

    mfsprefix = ["", '-MFS'][int(config['img_nchans'] > 1)]
    all_targets, all_msfile, ms_dict = pipeline.get_target_mss(label)

    i = 0
    for i, m in enumerate(all_msfile):
        # check whether all ms files to be used exist
        if not os.path.exists(os.path.join(pipeline.msdir, m)):
            raise IOError(
                "MS file {0:s} does not exist. Please check that it is where it should be.".format(m))

        # Write/rewind flag versions only if flagging tasks are being
        # executed on these .MS files, or if the user asks to rewind flags
        if rewind_main_ms:
            available_flagversions = manflags.get_flags(pipeline, m)
            if rewind_main_ms:
                if config['rewind_flags']['mode'] == 'reset_worker':
                    version = flags_before_worker
                    stop_if_missing = False
                elif config['rewind_flags']['mode'] == 'rewind_to_version':
                    version = config['rewind_flags']['version']
                    if version == 'auto':
                        version = flags_before_worker
                    stop_if_missing = True
                if version in available_flagversions:
                    if flags_before_worker in available_flagversions and available_flagversions.index(flags_before_worker) < available_flagversions.index(version) and not config['overwrite_flagvers']:
                        manflags.conflict('rewind_too_little', pipeline, wname, m, config, flags_before_worker, flags_after_worker)
                    substep = 'version-{0:s}-ms{1:d}'.format(version, i)
                    manflags.restore_cflags(pipeline, recipe, version, m, cab_name=substep)
                    if version != available_flagversions[-1]:
                        substep = 'delete-flag_versions-after-{0:s}-ms{1:d}'.format(version, i)
                        manflags.delete_cflags(pipeline, recipe,
                            available_flagversions[available_flagversions.index(version)+1],
                            m, cab_name=substep)
                    if version != flags_before_worker:
                        substep = 'save-{0:s}-ms{1:d}'.format(flags_before_worker, i)
                        manflags.add_cflags(pipeline, recipe, flags_before_worker,
                            m, cab_name=substep, overwrite=config['overwrite_flagvers'])
                elif stop_if_missing:
                    manflags.conflict('rewind_to_non_existing', pipeline, wname, m, config, flags_before_worker, flags_after_worker)
                # elif flag_main_ms:
                #     substep = 'save-{0:s}-ms{1:d}'.format(flags_before_worker, i)
                #     manflags.add_cflags(pipeline, recipe, flags_before_worker,
                #         m, cab_name=substep, overwrite=config['overwrite_flagvers'])
            else:
                if flags_before_worker in available_flagversions and not config['overwrite_flagvers']:
                    manflags.conflict('would_overwrite_bw', pipeline, wname, m, config, flags_before_worker, flags_after_worker)
                else:
                    substep = 'save-{0:s}-ms{1:d}'.format(flags_before_worker, i)
                    manflags.add_cflags(pipeline, recipe, flags_before_worker,
                        m, cab_name=substep, overwrite=config['overwrite_flagvers'])

    prefix = pipeline.prefix

    def image(trg, img_dir, mslist, field):
        key = 'image'
        ncpu_img = config[key]['ncpu_img'] if config[key]['ncpu_img'] else ncpu
        absmem = config[key]['absmem']
        caracal.log.info("Number of threads used by WSClean for gridding:")
        caracal.log.info(ncpu_img)
        imcolumn = config[key]['col']

        step = 'image-field{0:d}'.format(trg)
        image_opts = {
            "msname": mslist,
            "prefix": '{0:s}/{1:s}_{2:s}'.format(img_dir, prefix, field),
            "column": imcolumn,
            "weight": imgweight if not imgweight == 'briggs' else 'briggs {}'.format(robust),
            "nmiter": sdm.dismissable(config['img_nmiter']),
            "npix": config['img_npix'],
            "padding": config['img_padding'],
            "scale": config['img_cell'],
            "niter": config['img_niter'],
            "gain": config["img_gain"],
            "mgain": config['img_mgain'],
            "pol": config['img_stokes'],
            "channelsout": config['img_nchans'],
            "joinchannels": config['img_joinchans'],
            "squared-channel-joining": config['img_squared_chansjoin'],
            "join-polarizations": config['img_join_polarizations'],
            "auto-threshold": config[key]['clean_cutoff'],
            "parallel-deconvolution": sdm.dismissable(wscl_parallel_deconv),
            "nwlayers-factor": nwlayers_factor,
            "threads": ncpu_img,
            "absmem": absmem,
        }
        if config['img_join_polarizations'] is False and config['img_specfit_nrcoeff'] > 0:
            image_opts["fit-spectral-pol"] = config['img_specfit_nrcoeff']
        if config['img_niter'] > 0:
            image_opts["savesourcelist"] = True
        if not config['img_mfs_weighting']:
            image_opts["nomfsweighting"] = True
        if maxuvl > 0.:
            image_opts.update({
                "maxuv-l": maxuvl,
                "taper-tukey": transuvl,
            })
        if float(taper) > 0.:
            image_opts.update({
                "taper-gaussian": taper,
            })
        if min_uvw > 0:
            image_opts.update({"minuvw-m": min_uvw})
        if multiscale:
            image_opts.update({"multiscale": multiscale})
            if multiscale_scales:
                image_opts.update({"multiscale-scales": list(map(int,multiscale_scales.split(',')))})

        mask_key = config[key]['cleanmask_method']
        if mask_key == 'wsclean':
            image_opts.update({
                "auto-mask": config[key]['cleanmask_thr'],
                "local-rms": config[key]['cleanmask_localrms'],
              })
            if config[key]['cleanmask_localrms']:
                image_opts.update({
                    "local-rms-window": config[key]['cleanmask_localrms_window'],
                  })
        else:
            fits_mask = 'masking/{0:s}.fits'.format(mask_key)
            if not os.path.isfile('{0:s}/{1:s}'.format(pipeline.output, fits_mask)):
                raise caracal.ConfigurationError("Clean mask {0:s}/{1:s} not found. Please make sure that you have given the correct mask label"\
                    " in cleanmask_method, and that the mask exists.".format(pipeline.output, fits_mask))
            image_opts.update({
                "fitsmask": '{0:s}:output'.format(fits_mask),
                "local-rms": False,
              })

        recipe.add('cab/wsclean', step,
                   image_opts,
                   input=pipeline.input,
                   output=pipeline.output,
                   label='{:s}:: Make wsclean image'.format(step))

    target_iter=0
    for target in all_targets:
        mslist = ms_dict[target]
        field = utils.filter_name(target)
        image_path = "{0:s}/polarization".format(pipeline.continuum)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        image(target_iter, get_dir_path(image_path, pipeline), mslist, field)

        for i, msname in enumerate(mslist):
            if pipeline.enable_task(config, 'flagging_summary'):
                step = 'flagging_summary-selfcal-ms{0:d}'.format(i)
                recipe.add('cab/casa_flagdata', step,
                           {
                               "vis": msname,
                               "mode": 'summary',
                           },
                           input=pipeline.input,
                           output=pipeline.output,
                           label='{0:s}:: Flagging summary  ms={1:s}'.format(step, msname))


