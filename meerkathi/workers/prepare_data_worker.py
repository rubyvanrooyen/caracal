import os
import sys
from meerkathi.workers.utils import manage_flagsets as manflags

NAME = "Prepare data for calibration"


def worker(pipeline, recipe, config):

    wname = pipeline.CURRENT_WORKER
    for i in range(pipeline.nobs):

        msname = pipeline.msnames[i]
        prefix = pipeline.prefixes[i]

        if pipeline.enable_task(config, 'fixvis'):
            step = 'fixvis_{:d}'.format(i)
            recipe.add('cab/casa_fixvis', step,
                       {
                           "vis": msname,
                           "reuse": False,
                           "outputvis": msname,
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='{0:s}:: Fix UVW coordinates ms={1:s}'.format(step, msname))

        if pipeline.enable_task(config, "manage_flags"):

            if config["manage_flags"].get("remove_flagsets", False):
                step = "remove_flags_{0:s}_{1:d}".format(wname, i)
                manflags.delete_cflags(pipeline, recipe, "all", msname, cab_name=step)

            step = "init_legacy_flags_{0:s}_{1:d}".format(wname, i)
            manflags.add_cflags(pipeline, recipe, "legacy", msname, cab_name=step)

        if config["clear_cal"]:
            step = 'clear_cal_{:d}'.format(i)
            fields = set(pipeline.fcal[i] + pipeline.bpcal[i])
            recipe.add('cab/casa_clearcal', step,
                       {
                           "vis": msname,
                           "field" : ",".join(fields),
                       },
                       input=pipeline.input,
                       output=pipeline.output,
                       label='{0:s}:: Reset MODEL_DATA ms={1:s}'.format(step, msname))

        if pipeline.enable_task(config, "spectral_weights"):
            _config = config['spectral_weights']
            if _config["uniform"]:
                step = 'init_weights_{:d}'.format(i)
                recipe.add('cab/casa_script', step,
                           {
                               "vis": msname,
                               "script" : "vis = os.path.join(os.environ['MSDIR'], '{:s}')\n" \
                                          "initweights(vis=vis, wtmode='weight', dowtsp=True)".format(msname),
                           },
                           input=pipeline.input,
                           output=pipeline.output,
                           label='{0:s}:: Adding Spectral weights using MeerKAT noise specs ms={1:s}'.format(step, msname))

            elif _config["estimate"]["enable"]:
                step = 'estimate_weights_{:d}'.format(i)
                recipe.add('cab/msutils', step,
                           {
                               "msname": msname,
                               "command": 'estimate_weights',
                               "stats_data": _config['estimate'].get('stats_data'),
                               "weight_columns": _config['estimate'].get('weight_columns'),
                               "noise_columns": _config['estimate'].get('noise_columns'),
                               "write_to_ms": _config['estimate'].get('write_to_ms'),
                               "plot_stats": prefix + '-noise_weights.png',
                           },
                           input=pipeline.input,
                           output=pipeline.diagnostic_plots,
                           label='{0:s}:: Adding Spectral weights using MeerKAT noise specs ms={1:s}'.format(step, msname))

            elif _config["delete"]:
                step = 'delete_weight_spectrum{:d}'.format(i)
                recipe.add('cab/casa_script', step,
                           {
                               "vis": msname,
                               "script" : "vis = os.path.join(os.environ['MSDIR'], '{msname:s}') \n" \
                                          "colname = '{colname:s}' \n" \
                                          "tb.open(vis, nomodify=False) \n" \
                                          "try: tb.colnames().index(colname) \n" \
                                          "except ValueError: pass \n" \
                                          "finally: tb.close(); quit \n" \
                                          "tb.open(vis, nomodify=False) \n" \
                                          "try: tb.removecols(colname) \n" \
                                          "except RuntimeError: pass \n" \
                                          "finally: tb.close()".format(msname=msname, colname="WEIGHT_SPECTRUM"),
                           },
                           input=pipeline.input,
                           output=pipeline.output,
                           label='{0:s}:: deleting WEIGHT_SPECTRUM if it exists ms={1:s}'.format(step, msname))
