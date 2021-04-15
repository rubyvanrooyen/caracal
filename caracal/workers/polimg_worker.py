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
    rewind_main_ms = config['rewind_flags']["enable"] and (
            config['rewind_flags']['mode'] == 'reset_worker' or config['rewind_flags']["version"] != 'null')
    taper = config['img_taper']
    maxuvl = config['img_maxuv_l']
    transuvl = maxuvl * config['img_transuv_l'] / 100.
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
    nwlayers_factor = config['img_nwlayers_factor']
    nrdeconvsubimg = ncpu if config['img_nrdeconvsubimg'] == 0 else config['img_nrdeconvsubimg']
    if nrdeconvsubimg == 1:
        wscl_parallel_deconv = None
    else:
        wscl_parallel_deconv = int(np.ceil(config['img_npix'] / np.sqrt(nrdeconvsubimg)))

    mfsprefix = ["", '-MFS'][int(config['img_nchans'] > 1)]
    all_targets, all_msfile, ms_dict = pipeline.get_target_mss(label)

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
                    if flags_before_worker in available_flagversions and available_flagversions.index(
                            flags_before_worker) < available_flagversions.index(version) and not config[
                        'overwrite_flagvers']:
                        manflags.conflict('rewind_too_little', pipeline, wname, m, config, flags_before_worker,
                                          flags_after_worker)
                    substep = 'version-{0:s}-ms{1:d}'.format(version, i)
                    manflags.restore_cflags(pipeline, recipe, version, m, cab_name=substep)
                    if version != available_flagversions[-1]:
                        substep = 'delete-flag_versions-after-{0:s}-ms{1:d}'.format(version, i)
                        manflags.delete_cflags(pipeline, recipe,
                                               available_flagversions[available_flagversions.index(version) + 1],
                                               m, cab_name=substep)
                    if version != flags_before_worker:
                        substep = 'save-{0:s}-ms{1:d}'.format(flags_before_worker, i)
                        manflags.add_cflags(pipeline, recipe, flags_before_worker,
                                            m, cab_name=substep, overwrite=config['overwrite_flagvers'])
                elif stop_if_missing:
                    manflags.conflict('rewind_to_non_existing', pipeline, wname, m, config, flags_before_worker,
                                      flags_after_worker)
                # elif flag_main_ms:
                #     substep = 'save-{0:s}-ms{1:d}'.format(flags_before_worker, i)
                #     manflags.add_cflags(pipeline, recipe, flags_before_worker,
                #         m, cab_name=substep, overwrite=config['overwrite_flagvers'])
            else:
                if flags_before_worker in available_flagversions and not config['overwrite_flagvers']:
                    manflags.conflict('would_overwrite_bw', pipeline, wname, m, config, flags_before_worker,
                                      flags_after_worker)
                else:
                    substep = 'save-{0:s}-ms{1:d}'.format(flags_before_worker, i)
                    manflags.add_cflags(pipeline, recipe, flags_before_worker,
                                        m, cab_name=substep, overwrite=config['overwrite_flagvers'])

    prefix = pipeline.prefix

    # rename single stokes fits files
    def rename_single_stokes(img_dir, field, stokes):
        posname = '{0:s}/{1:s}/{2:s}_{3:s}'.format(pipeline.output, img_dir, prefix, field)
        llist = list(set(glob.glob('{0:s}_{1:s}'.format(posname, '*psf.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*I-psf.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*Q-psf.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*U-psf.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*V-psf.fits'))))
        for fname in llist:
            os.rename(fname, fname[:-8] + stokes + "-psf.fits")
        llist = list(set(glob.glob('{0:s}_{1:s}'.format(posname, '*dirty.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*I-dirty.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*Q-dirty.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*U-dirty.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*V-dirty.fits'))))
        for fname in llist:
            os.rename(fname, fname[:-10] + stokes + "-dirty.fits")
        llist = list(set(glob.glob('{0:s}_{1:s}'.format(posname, '*image.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*I-image.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*Q-image.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*U-image.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*V-image.fits'))))
        for fname in llist:
            os.rename(fname, fname[:-10] + stokes + "-image.fits")
        llist = list(set(glob.glob('{0:s}_{1:s}'.format(posname, '*model.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*I-model.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*Q-model.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*U-model.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*V-model.fits'))))
        for fname in llist:
            os.rename(fname, fname[:-10] + stokes + "-model.fits")
        llist = list(set(glob.glob('{0:s}_{1:s}'.format(posname, '*residual.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*I-residual.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*Q-residual.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*U-residual.fits'))) -
                     set(glob.glob('{0:s}_{1:s}'.format(posname, '*V-residual.fits'))))
        for fname in llist:
            os.rename(fname, fname[:-13] + stokes + "-residual.fits")

    def change_header_and_type(filename, headfile, copy_head):
        pblist = fits.open(filename)
        dat = pblist[0].data
        pblist.close()
        if copy_head:
            head = fits.getheader(headfile, 0)
        else:
            head = fits.getheader(filename, 0)
            # delete ORIGIN, CUNIT1, CUNIT2
            if 'ORIGIN' in head:
                del head['ORIGIN']
            if 'CUNIT1' in head:
                del head['CUNIT1']
            if 'CUNIT2' in head:
                del head['CUNIT2']
            # copy CRVAL3 from headfile to filename
            template_head = fits.getheader(headfile, 0)
            if 'crval3' in template_head:
                head['crval3'] = template_head['crval3']
        fits.writeto(filename, dat.astype('int32'), head, overwrite=True)

    def fake_image(trg, num, img_dir, mslist, field):
        ncpu_img = config['ncpu_img'] if config['ncpu_img'] else ncpu

        step = 'image-field{0:d}-iter{1:d}'.format(trg, num)
        fake_image_opts = {
            "msname": mslist,
            "column": config['col'],
            "weight": config['img_weight'] if not config['img_weight'] == 'briggs' else 'briggs {}'.format(
                config['img_robust']),
            "nmiter": sdm.dismissable(config['img_nmiter']),
            "nomfsweighting": config['img_mfs_weighting'],
            "npix": config['img_npix'],
            "padding": config['img_padding'],
            "scale": config['img_cell'],
            "prefix": '{0:s}/{1:s}_{2:s}_{3:d}'.format(img_dir, prefix, field, num),
            "niter": config['img_niter'],
            "gain": config["img_gain"],
            "mgain": config['img_mgain'],
            "pol": config['img_stokes'],
            "channelsout": config['img_nchans'],
            "joinchannels": config['img_joinchans'],
            "squared-channel-joining": config['img_squared_chansjoin'],
            "join-polarizations": config['img_join_polarizations'],
            "local-rms": False,
            "auto-mask": 6,
            "auto-threshold": config['clean_cutoff'],
            "fitbeam": False,
            "parallel-deconvolution": sdm.dismissable(wscl_parallel_deconv),
            "nwlayers-factor": nwlayers_factor,
            "threads": ncpu_img,
            "absmem": config['absmem'],
        }
        if config['img_join_polarizations'] is False and config['img_specfit_nrcoeff'] > 0:
            fake_image_opts["fit-spectral-pol"] = config['img_specfit_nrcoeff']
        if not config['img_mfs_weighting']:
            fake_image_opts["nomfsweighting"] = True
        if maxuvl > 0.:
            fake_image_opts.update({
                "maxuv-l": maxuvl,
                "taper-tukey": transuvl,
            })
        if float(taper) > 0.:
            fake_image_opts.update({
                "taper-gaussian": taper,
            })
        if min_uvw > 0:
            fake_image_opts.update({"minuvw-m": min_uvw})

        recipe.add('cab/wsclean', step,
                   fake_image_opts,
                   input=pipeline.input,
                   output=pipeline.output,
                   label='{:s}:: Make image after first round of calibration'.format(step))

    def sofia_mask(trg, num, img_dir, field, stokes):
        step = 'make-sofia_mask-field{0:d}-iter{1:d}-{2:s}'.format(trg, num, stokes)
        key = 'img_sofia_settings'

        if config['img_joinchans']:
            imagename = '{0:s}/{1:s}_{2:s}_{3:d}-MFS-{4:s}-image.fits'.format(
                img_dir, prefix, field, num, stokes)
        else:
            imagename = '{0:s}/{1:s}_{2:s}_{3:d}-{4:s}-image.fits'.format(
                img_dir, prefix, field, num, stokes)

        if config[key]['fornax_special'] is True and config[key]['fornax_sofia'] is True:
            forn_kernels = [[80, 80, 0, 'b']]
            forn_thresh = config[key]['fornax_thr']

            sofia_opts_forn = {
                "import.inFile": imagename,
                "steps.doFlag": True,
                "steps.doScaleNoise": False,
                "steps.doSCfind": True,
                "steps.doMerge": True,
                "steps.doReliability": False,
                "steps.doParameterise": False,
                "steps.doWriteMask": True,
                "steps.doMom0": False,
                "steps.doMom1": False,
                "steps.doWriteCat": False,
                "parameters.dilateMask": False,
                "parameters.fitBusyFunction": False,
                "parameters.optimiseMask": False,
                "SCfind.kernelUnit": 'pixel',
                "SCfind.kernels": forn_kernels,
                "SCfind.threshold": forn_thresh,
                "SCfind.rmsMode": 'mad',
                "SCfind.edgeMode": 'constant',
                "SCfind.fluxRange": 'all',
                "scaleNoise.method": 'local',
                "scaleNoise.windowSpatial": 51,
                "scaleNoise.windowSpectral": 1,
                "writeCat.basename": 'FornaxA_sofia',
                "merge.radiusX": 3,
                "merge.radiusY": 3,
                "merge.radiusZ": 1,
                "merge.minSizeX": 100,
                "merge.minSizeY": 100,
                "merge.minSizeZ": 1,
            }

        outmask = pipeline.prefix + '_' + field + '_' + str(num+1) + '_' + stokes + '_clean'

        sofia_opts = {
            "import.inFile": imagename,
            "steps.doFlag": True,
            "steps.doScaleNoise": config['cleanmask_localrms'],
            "steps.doSCfind": True,
            "steps.doMerge": True,
            "steps.doReliability": False,
            "steps.doParameterise": False,
            "steps.doWriteMask": True,
            "steps.doMom0": False,
            "steps.doMom1": False,
            "steps.doWriteCat": True,
            "writeCat.writeASCII": False,
            "writeCat.basename": outmask,
            "writeCat.writeSQL": False,
            "writeCat.writeXML": False,
            "parameters.dilateMask": False,
            "parameters.fitBusyFunction": False,
            "parameters.optimiseMask": False,
            "SCfind.kernelUnit": 'pixel',
            "SCfind.kernels": [[kk, kk, 0, 'b'] for kk in config[key]['kernels']],
            "SCfind.threshold": config['cleanmask_thr'],
            "SCfind.rmsMode": 'mad',
            "SCfind.edgeMode": 'constant',
            "SCfind.fluxRange": 'all',
            "scaleNoise.statistic": 'mad',
            "scaleNoise.method": 'local',
            "scaleNoise.interpolation": 'linear',
            "scaleNoise.windowSpatial": config['cleanmask_localrms_window'],
            "scaleNoise.windowSpectral": 1,
            "scaleNoise.scaleX": True,
            "scaleNoise.scaleY": True,
            "scaleNoise.scaleZ": False,
            "scaleNoise.perSCkernel": config['cleanmask_localrms'],
            # work-around for https://github.com/SoFiA-Admin/SoFiA/issues/172, to be replaced by "True" once the next SoFiA version is in Stimela
            "merge.radiusX": 3,
            "merge.radiusY": 3,
            "merge.radiusZ": 1,
            "merge.minSizeX": 3,
            "merge.minSizeY": 3,
            "merge.minSizeZ": 1,
            "merge.positivity": config[key]['pospix'],
        }
        if config[key]['flag']:
            flags_sof = config[key]['flagregion']
            sofia_opts.update({"flag.regions": flags_sof})

        if config[key]['inputmask']:
            mask_fits = 'masking/' + config[key]['inputmask']
            mask_casa = mask_fits.replace('.fits', '.image')
            mask_regrid_casa = mask_fits.replace('.fits', '_regrid.image')
            mask_regrid_fits = mask_fits.replace('.fits', '_regrid.fits')
            imagename_casa = imagename.split('/')[-1].replace('.fits', '.image')

            recipe.add('cab/casa_importfits', step + "-import-image",
                       {
                           "fitsimage": imagename,
                           "imagename": imagename_casa,
                           "overwrite": True,
                       },
                       input=pipeline.output,
                       output=pipeline.output,
                       label='Import image in casa format')

            recipe.add('cab/casa_importfits', step + "-import-mask",
                       {
                           "fitsimage": mask_fits + ':output',
                           "imagename": mask_casa,
                           "overwrite": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Import mask in casa format')

            recipe.add('cab/casa_imregrid', step + "-regrid-mask",
                       {
                           "template": imagename_casa + ':output',
                           "imagename": mask_casa + ':output',
                           "output": mask_regrid_casa,
                           "overwrite": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Regrid mask to image')

            recipe.add('cab/casa_exportfits', step + "-export-mask",
                       {
                           "fitsimage": mask_regrid_fits + ':output',
                           "imagename": mask_regrid_casa + ':output',
                           "overwrite": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Export regridded mask to fits')

            recipe.add(change_header_and_type, step + "-copy-header",
                       {
                           "filename": pipeline.output + '/' + mask_regrid_fits,
                           "headfile": pipeline.output + '/' + imagename,
                           "copy_head": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Copy image header to mask')

            sofia_opts.update({"import.maskFile": mask_regrid_fits})
            sofia_opts.update({"import.inFile": imagename})

        if config[key]['fornax_special'] is True and config[key]['fornax_sofia'] is True:

            recipe.add('cab/sofia', step + "-fornax_special",
                       sofia_opts_forn,
                       input=pipeline.output,
                       output=pipeline.output + '/masking/',
                       label='{0:s}:: Make SoFiA mask'.format(step))

            fornax_namemask = 'masking/FornaxA_sofia_mask.fits'
            sofia_opts.update({"import.maskFile": fornax_namemask})

        elif config[key]['fornax_special'] is True and config[key]['fornax_sofia'] is False:

            # this mask should be regridded to correct f.o.v.

            fornax_namemask = 'masking/Fornaxa_vla_mask_doped.fits'
            fornax_namemask_regr = 'masking/Fornaxa_vla_mask_doped_regr.fits'

            mask_casa = fornax_namemask + '.image'

            mask_regrid_casa = fornax_namemask + '_regrid.image'

            imagename_casa = '{0:s}_{1:d}{2:s}-image.image'.format(prefix, num, mfsprefix)

            recipe.add('cab/casa_importfits', step + "-fornax_special-import-image",
                       {
                           "fitsimage": imagename,
                           "imagename": imagename_casa,
                           "overwrite": True,
                       },
                       input=pipeline.output,
                       output=pipeline.output,
                       label='Image in casa format')

            recipe.add('cab/casa_importfits', step + "-fornax_special-import-image",
                       {
                           "fitsimage": fornax_namemask + ':output',
                           "imagename": mask_casa,
                           "overwrite": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Mask in casa format')

            recipe.add('cab/casa_imregrid', step + "-fornax_special-regrid",
                       {
                           "template": imagename_casa + ':output',
                           "imagename": mask_casa + ':output',
                           "output": mask_regrid_casa,
                           "overwrite": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Regridding mosaic to size and projection of dirty image')

            recipe.add('cab/casa_exportfits', step + "-fornax_special-export-mosaic",
                       {
                           "fitsimage": fornax_namemask_regr + ':output',
                           "imagename": mask_regrid_casa + ':output',
                           "overwrite": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Extracted regridded mosaic')

            recipe.add(change_header_and_type, step + "-fornax_special-change_header",
                       {
                           "filename": pipeline.output + '/' + fornax_namemask_regr,
                           "headfile": pipeline.output + '/' + imagename,
                           "copy_head": True,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='Extracted regridded mosaic')

            sofia_opts.update({"import.maskFile": fornax_namemask_regr})

        recipe.add('cab/sofia', step,
                   sofia_opts,
                   input=pipeline.output,
                   output=pipeline.output + '/masking/',
                   label='{0:s}:: Make SoFiA mask'.format(step))

    def image(trg, num, img_dir, mslist, field):
        ncpu_img = config['ncpu_img'] if config['ncpu_img'] else ncpu
        caracal.log.info("Number of threads used by WSClean for gridding:")
        caracal.log.info(ncpu_img)

        step = 'image-field{0:d}-iter{1:d}'.format(trg, num)
        image_opts = {
            "msname": mslist,
            "prefix": '{0:s}/{1:s}_{2:s}_{3:d}'.format(img_dir, prefix, field, num),
            "column": config['col'],
            "weight": config['img_weight'] if not config['img_weight'] == 'briggs' else 'briggs {}'.format(
                config['img_robust']),
            "nmiter": sdm.dismissable(config['img_nmiter']),
            "npix": config['img_npix'],
            "padding": config['img_padding'],
            "scale": config['img_cell'],
            "niter": config['img_niter'],
            "gain": config["img_gain"],
            "mgain": config['img_mgain'],
            "channelsout": config['img_nchans'],
            "joinchannels": config['img_joinchans'],
            "squared-channel-joining": config['img_squared_chansjoin'],
            #"join-polarizations": config['img_join_polarizations'],
            "auto-threshold": config['clean_cutoff'],
            "parallel-deconvolution": sdm.dismissable(wscl_parallel_deconv),
            "nwlayers-factor": nwlayers_factor,
            "threads": ncpu_img,
            "absmem": config['absmem'],
        }

        mask_key = config['cleanmask_method']

        #join polarization only if they will be imaged together
        joinpol=False
        if mask_key == 'wsclean':
            joinpol = config['img_join_polarizations']

        image_opts["join-polarizations"] = joinpol

        if joinpol is False and config['img_specfit_nrcoeff'] > 0:
            image_opts["fit-spectral-pol"] = config['img_specfit_nrcoeff']
            if config['img_niter'] > 0 and config['img_stokes'] == 'I':
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
                image_opts.update({"multiscale-scales": list(map(int, multiscale_scales.split(',')))})

        if mask_key == 'wsclean':
            image_opts.update({
                "pol": config['img_stokes'],
                "auto-mask": config['cleanmask_thr'],
                "local-rms": config['cleanmask_localrms'],
            })
            if config['cleanmask_localrms']:
                image_opts.update({
                    "local-rms-window": config['cleanmask_localrms_window'],
                })
            recipe.add('cab/wsclean', step,
                       image_opts,
                       input=pipeline.input,
                       output=pipeline.output,
                       label='{:s}:: Make wsclean image (iter {})'.format(step, num))
        else:
            for stokes in config['img_stokes']:
                if mask_key == 'sofia':
                    fits_mask = 'masking/{0:s}_{1:s}_{2:d}_{3:s}_clean_mask.fits'.format(prefix, field, num, stokes)
                    if not os.path.isfile('{0:s}/{1:s}'.format(pipeline.output, fits_mask)):
                        raise caracal.ConfigurationError(
                            "SoFiA clean mask {0:s}/{1:s} not found. Something must have gone wrong with the SoFiA run" \
                            " (maybe the detection threshold was too high?). Please check the logs.".format(pipeline.output,fits_mask))
                else:
                    fits_mask = 'masking/{0:s}.fits'.format(mask_key)
                    if not os.path.isfile('{0:s}/{1:s}'.format(pipeline.output, fits_mask)):
                        raise caracal.ConfigurationError(
                            "Clean mask {0:s}/{1:s} not found. Please make sure that you have given the correct mask label" \
                            " in cleanmask_method, and that the mask exists.".format(pipeline.output, fits_mask))

                image_opts.update({
                    "pol": stokes,
                    "fitsmask": '{0:s}:output'.format(fits_mask),
                    "local-rms": False,
                })
                recipe.add('cab/wsclean', step + "_" + stokes,
                           image_opts,
                           input=pipeline.input,
                           output=pipeline.output,
                           label='{:s}:: Make wsclean image (selfcal iter {})'.format(step, num, stokes))
                recipe.run()
                recipe.jobs = []

    target_iter = 0
    for target in all_targets:
        mslist = ms_dict[target]
        field = utils.filter_name(target)
	image_path = "{0:s}/polarization".format(pipeline.continuum)
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        if config['cleanmask_method'] == 'sofia':
            image_path = "{0:s}/polarization/image_0".format(pipeline.continuum)
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            fake_image(target_iter, 0, get_dir_path(image_path, pipeline), mslist, field)
            #rename single stokes files (wsclean do not label the single stokes outputs)
            alone = ["I", "Q", "U", "V"]
            stokes = config['img_stokes']
            if stokes in alone:
                rename_single_stokes(get_dir_path(image_path, pipeline), field, stokes)
            for s in config['img_stokes']:
                sofia_mask(target_iter, 0, get_dir_path(image_path, pipeline), field, s)
            recipe.run()
            recipe.jobs = []
            image_path = "{0:s}/polarization/image_1".format(pipeline.continuum)
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            image(target_iter, 1, get_dir_path(image_path, pipeline), mslist, field)

        else:
            image_path = "{0:s}/polarization/image_1".format(pipeline.continuum)
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            image(target_iter, 1, get_dir_path(image_path, pipeline), mslist, field)

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
