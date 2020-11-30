from os import pipe
import stimela
import os.path
from collections import OrderedDict

_MODES = dict(
    K       = "delay_cal", 
    B       = "bp_cal", 
    F       = "gain_cal", 
    G       = "gain_cal",   # both F and G serve the same purpose, so same mode
    Gpol    = "gain_xcal",
    Kcrs    = 'cross_delay',
    Xref    = 'cross_phase_ref',
    Xf      = 'cross_phase',
    Dref    = 'leakage_ref',
    Df      = 'leakage',
    Gxyamp  = 'cross_gain',
    Xfparang = 'cross_phase',
    Df0gen   = 'leakage'
    )

def new_callib():
    return dict()

def add_callib_recipe(callib, gt, interp, fldmap, field=None, calwt=False):
    """Adds gaintable to a callib
        gt:     gain table path
        interp: interpolation policy
        fldmap: field mapping policy
        field:  if set, then this table must appy to a specific field. Otherwise set as default.
    """
    # get extension of gain table, and strip off digits at end
    _, ext = os.path.splitext(gt)
    ext = ext and ext[1:].rstrip("0123456789")
    mode = _MODES.get(ext, "unknown")
    cal_entries = callib.setdefault(mode, {})
    entry = dict(caltable=gt, fldmap=fldmap, interp=interp, calwt=bool(calwt))
    # if specific field is set and we have a default entry, check that this one is different
    if field:
        default = cal_entries.get("default")
        if default and all(val == default.get(key) for key, val in entry.items()):
            return
    else: 
        field = "default"

    cal_entries[field] = entry 

def resolve_calibration_library(pipeline, msprefix, cal_lib, cal_label, worker_label=None, output_fields=None):
    """
    Reads callib specified by configuration. Figures out how to apply it to the given set of output fields.
    Writes a CASA-compatible callib.txt file describing same.
    Returns a tupe of:
        callib_filename, (gaintables, gainfields, interps, calwts, fields)
    where the latter are five lists suitable to the CASA applycal task:
        - gain tables
        - field mapping policies (gainfield)
        - interpolation type
        - calwt
        - field (to apply to)
    Arguments:
        pipeline:      worker administrator object
        msprefix:       filename prefix (based on MS name etc.)
        cal_lib:        name of callib given in config, if supplied. Overrides label, if given.
        cal_label:      label given in config, if supplied.
        worker_label:   label of worker. This is used to form up the output .txt filename.
        output_fields:  set of fields that the calibration is applied to. If None, assume target fields.
    """
    cal_lists = [], [], [], [], []         # init 5 empty lists for output values

    # get name from callib name and/or from prefix
    if not cal_lib:
        if cal_label:
            cal_lib = f"{msprefix}-{cal_label}"
        else:
            return None, cal_lists

    caldict = pipeline.load_callib(cal_lib)
    outfile = pipeline.get_callib_name(cal_lib, "txt", pipeline.CURRENT_WORKER)
    with open(outfile, 'w') as stdw:
        for _, cal_entries in caldict.items():
            cal_fields = set(cal_entries.keys())
            # specific fields -- set of fields for which a separate caltable is defined
            # default_fields -- set of fields for which the default caltable is used
            if output_fields is None:
                specific_fields = {}
                default_fields = {""}  # this will turn into '' post-join below, which CASA recognizes as default
            else:
                specific_fields = set(output_fields).intersection(cal_fields)
                default_fields  = set(output_fields).difference(cal_fields)
            # go through all tables, skip the ones that don't apply
            for field, entry in cal_entries.items():
                if field == "default":
                    if not default_fields:
                        continue
                    field = ",".join(default_fields)   
                elif field not in specific_fields:
                    continue
                cal_lists[0].append(entry['caltable'])
                cal_lists[1].append(entry['fldmap'])
                cal_lists[2].append(entry['interp'])
                calwt = entry.get('calwt', False)
                cal_lists[3].append(calwt)
                cal_lists[4].append(field)

                filename = os.path.join(stimela.recipe.CONT_IO["output"], 'caltables', entry['caltable']) 
                stdw.write(f"""caltable="{filename}" calwt={calwt} tinterp='{entry['interp']}' """
                    f"""finterp='linear' fldmap='{entry['fldmap']}' field='{field}' spwmap=0\n""")

    return outfile[len(pipeline.output)+1:], cal_lists
