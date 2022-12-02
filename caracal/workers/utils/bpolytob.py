"""
This is a temporary hack to map the BPOLY table to Bcal table in CASA
It copies an existing table and update the values
"""

import caracal
import os
import sys

import numpy as np

from casacore.tables import table
import casacore.fitting as fitting


def metaref(bpolytable):
    """
    Extract info from reference Measurement Set

    @param bpolytable string : CARACal BPOLY table name

    @return msfile string      : Name of reference MS file
    @return nrows int          : Nr of rows in Bcal table
    @return Xpol_idx int       : Index of XX pol correlator data
    @return Ypol_idx int       : Index of YY pol correlator data
    @return freq_range ndarray : Observation Frequency range [Hz]
    @return antennas ndarray   : List of antenna names in array
    @return flag_mask ndarray  : Boolean array of generally flagged channels
    """

    # get nr of rows for new table
    nrows = table(bpolytable, ack=False).nrows()

    # get the ms file you're gonna need it
    msfile= table(bpolytable+"::CAL_DESC", ack=False).getcol('MS_NAME')[0]

    # Data: Complex value for each of 4 correlations per spectral channel
    # IDs correspond to values RR (5), RL (6), LR (7), LL (8), XX (9), XY (10), YX (11), and YY (12)
    corr_prods_id = {5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
                     9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY'}
    with table(msfile+'::POLARIZATION', ack=False) as tb:
        ncorr_prods = int(tb.getcol("NUM_CORR"))
        corr_prods = tb.getcol("CORR_TYPE").squeeze()
        tb.close()
    # associate corr prod with 4 tuple indices
    lin_prods = np.array([corr_prods_id[prod] for prod in corr_prods])
    Xpol_idx = np.nonzero(lin_prods == 'XX')[0][0]  # H pol
    Ypol_idx = np.nonzero(lin_prods == 'YY')[0][0]  # V pol
    # H = XX
#     caracal.log.info(f'H pol {Xpol_idx} index in corr products')
    print(f'H pol {Xpol_idx} index in corr products')
    # V = YY
#     caracal.log.info(f'V pol {Ypol_idx} index in corr products')
    print(f'V pol {Ypol_idx} index in corr products')

    freq_range = table(msfile+"::SPECTRAL_WINDOW", ack=False).getcol("CHAN_FREQ").squeeze()
    antennas = table(msfile+"::ANTENNA", ack=False).getcol("NAME")

    # find passband from flags
    flags = table(msfile, ack=False).getcol("FLAG")
    [num_rows, num_chans, num_pols] = flags.shape
    flags=flags.sum(axis=0)/num_rows
    flags = flags.sum(axis=1)/num_pols
    flag_mask = flags > 0.5

    return msfile, nrows, Xpol_idx, Ypol_idx, freq_range, antennas, flag_mask


def create_empty_B(btable, template_bcal, nrows=1):
    """
    Create an empty B cal table from BPOLY cal table

    @param btable string        : Output Bcal table name
    @param template_bcal string : Empty Bcal template table name
    @param nrows int            : [Optional] Number of rows in table

    @return None
    """

    if os.path.exists(btable):
        os.system(f'rm -rf {btable}')
    # get old table schema by reading empty templage MS: "",
#     caracal.log.info(f"Creating BCAL table {btable}")
    print(f"Creating BCAL table {btable}")
    table(template_btable, ack=False).copy(btable, deep=True)
    # add the rows _before_ filling the columns.
    table(btable, ack=False, readonly=False).addrows(nrows-1)

    return btable


def addsubtables(msfile, btable):
    """
    Copy Bcal subtables from linked MS file

    @param btable string : CARACal Bcal table name
    @param msfile string : Name of reference MS file

    @return None
    """
    # cp subtables from MS
    with table(btable, ack=False, readonly=False) as tb:
        fieldnames=tb.getkeywords()
        tb.putkeyword("MSName", msfile)
        tb.flush()

    # get list of subtables to create
    subtable_list = []
    for name, value in fieldnames.items():
        if "table" in str(value).lower():
            subtable_list.append([name, value.split()[-1].strip()])

    # subtables are just copies from MS subtables
    for name, path in subtable_list:
        if name == 'HISTORY':
            continue
        intable = f"{msfile}::{name}"
        outtable = f"{btable}::{name}"
        with table(outtable, ack=False, readonly=False) as tbout:
            table(intable, ack=False).copyrows(tbout, startrowin=0, startrowout=0)
            tbout.flush()


def readcolumnasscalar(bpolytable, bp_field):
    """
    CASA varcol data / python dict column values flatten to np.ndarray

    @param bpolytable string : CARACal BPOLY table name
    @param bp_field string   : Name of BPOLY column to read

    @return colvalues ndarray : Column values as flat array
    """
    colvalues = readcolumn(bpolytable, bp_field)
    if isinstance(colvalues, dict):
        flatlist = []
        for key, value in colvalues.items():
            flatlist.append(value.squeeze())
        colvalues = np.array(flatlist)
    return colvalues


def readcolumn(btable, bp_field=None):
    """
    Read column data from BPOLY cal table

    @param btable string   : CARACal B table name
    @param bp_field string : Name of BPOLY column to read

    @return values ndarray   : BPOLY column values 
    """
    with table(btable, ack=False) as tb:
        if tb.isscalarcol(bp_field):
            values = tb.getcol(bp_field)
        elif tb.isvarcol(bp_field):
            values = tb.getvarcol(bp_field)
        else:
            values = tb.getcol(bp_field)
    return values


def makecolumndata(values, arrsize=[], dtype=np.ndarray, varcol=False):
    """
    Create ndarray or dict column values

    @param values float/bool/None : Default value to populate column array
    @param arrsize list           : Array dimensions, e.g. [nrow, ncol]
    @param dtype string/obj       : Dtype of numpy array elements
    @param varcol bool            : [Optional] convert default array to dict

    @return values ndarray/dict   : Default column values
    """
    # create dummy array for CASA
    if dtype == 'boolean': dtype='bool'
    if isinstance(dtype, str):
        dtype=eval(dtype)
    # create empty array
    if values is None:
        values = np.empty(arrsize, dtype=dtype)
    # create dummy array
    if not isinstance(values, np.ndarray):
        values = np.full(arrsize, values, dtype=dtype)
    if varcol:
        # unpack np.darray into varcol dict
        valuedict = {}
        for rowcnt, valueline in enumerate(values):
            valuedict[f"r{rowcnt+1}"] = np.expand_dims(np.array(valueline), axis=0)
        values = valuedict
    return values


def writecolumndata(btable, main_tbl_dict):
    """
    Write column data to table

    @param btable string        : CARACal B table name
    @param mail_tbl_dict dict   : Bcal table main table 

    @return None
    """
    colnames = table(btable, ack=False).colnames()
    [nrows, nchans, npols] = main_tbl_dict['CPARAM'].shape
    with table(btable, ack=False, readonly=False) as tb:
        for colname in colnames:
            # get info from input table
            coldtype = tb.coldatatype(colname)
            if tb.isscalarcol(colname):
                arrshape = tb.getcol(colname).shape
            if tb.isvarcol(colname):
                arrshape = np.shape(tb.getvarcol(colname)['r1'])
            if len(arrshape) == 1:
                # 1d array len nrows
                pass
            elif len(arrshape) == 2:
                arrshape = [nrows,nchans]
            else:
                arrshape = [nrows,nchans, npols]
            # update input table
            if colname in main_tbl_dict:
                values = main_tbl_dict[colname]
                if isinstance(values, dict):
                    tb.putvarcol(colname, values)
                else:
                    tb.putcol(colname, values)
            else:
                if tb.isvarcol(colname):
                    values = makecolumndata(0.,
                                            arrsize=arrshape,
                                            dtype=coldtype,
                                            varcol=True)
                    tb.putvarcol(colname, values)
                else:
                    values = makecolumndata(0.,
                                            arrsize=arrshape,
                                            dtype=coldtype)
                    tb.putcol(colname, values)
            tb.flush()


def readBPOLY(bpolytable):
    """
    Extract polyfit coefficients from bpolytable

    @param bpolytable string        : CARACal BPOLY table name

    @return scaleFactor NDArray     : Amplitude scale factor
    @return antennasBP NDArray      : Antenna IDs
    @return frequencyLimits NDArray : Domain to use. The interval [domain[0], domain[1]]
    @return frequenciesGHz NDArray  : Poly eval x points
    @return nPolyAmp NDArray        : Amplitude poly fit degree
    @return nPolyPhase NDArray      : Phase poly fit degree
    @return polynomialAmplitude NDArray : Amplitude fit Chebyshev coefficients
    @return polynomialPhase NDArray : Phase fit Chebyshev coefficients
    """

    # output for user info
    polyMode = readcolumn(bpolytable, "POLY_MODE")
#     caracal.log.info(f"This is a BPOLY solution = {polyMode[0]}")
    print(f"This is a BPOLY solution = {polyMode[0]}")
    polyType = readcolumn(bpolytable, "POLY_TYPE")
#     caracal.log.info(f"The 'BPOLY' solver fits ({np.unique(polyType)}) polynomials to the amplitude and phase of the calibrator visibilities as a function of frequency.")
    print(f"The 'BPOLY' solver fits ({np.unique(polyType)}) polynomials to the amplitude and phase of the calibrator visibilities as a function of frequency.")

    uniqueTimesBP = np.unique(readcolumn(bpolytable, "TIME"))
    nUniqueTimesBP = len(uniqueTimesBP)
    tsstring = f"There are {nUniqueTimesBP} unique times in the BPOLY solution: "
    for u in uniqueTimesBP:
        tsstring += '%.3f, ' % (u)
#     caracal.log.info(tsstring)

    # table information needed for display
    scaleFactor = readcolumnasscalar(bpolytable, 'SCALE_FACTOR')
    antennasBP = readcolumnasscalar(bpolytable, 'ANTENNA1')
    frequencyLimits = readcolumnasscalar(bpolytable, 'VALID_DOMAIN')

     # poly fit degrees
    nPolyAmp = readcolumnasscalar(bpolytable, 'N_POLY_AMP')
    degamp = int(np.unique(nPolyAmp)) - 1
    nPolyPhase = readcolumnasscalar(bpolytable, 'N_POLY_PHASE')
    degphase = int(np.unique(nPolyPhase)) - 1
#     caracal.log.info(f"BPOLY fit using AMP order {degamp} and PHASE order {degphase}")
    print(f"BPOLY fit using AMP order {degamp} and PHASE order {degphase}")

    # amplitude coefficients for antenna
    if (polyMode[0] == 'A&P' or polyMode[0] == 'A'):
        polynomialAmplitude = readcolumnasscalar(bpolytable, "POLY_COEFF_AMP")

    # phase coefficients for antenna
    if (polyMode[0] == 'A&P' or polyMode[0] == 'P'):
        polynomialPhase = readcolumnasscalar(bpolytable, "POLY_COEFF_PHASE")
    return scaleFactor, antennasBP, frequencyLimits, nPolyAmp, nPolyPhase, polynomialAmplitude, polynomialPhase


def bpolyfit(bpolytable, freq_range, antennas, flag_mask):
    """
    Calculate gain solutions from BPOLY table

    @param bpolytable string   : CARACal BPOLY table name
    @param freq_range ndarray  : Observation Frequency range [Hz]
    @param antennas ndarray    : List of antenna names in array
    @param flag_mask ndarray   : Boolean array of generally flagged channels

    @return bcal_sol ndarray : Bcal solution from poly fit
    """

    # read caltable
    [scaleFactor,
     antennasBP,
     frequencyLimits,
     nPolyAmp,
     nPolyPhase,
     polynomialAmplitude,
     polynomialPhase] = readBPOLY(bpolytable)

    # display results
#     caracal.log.info(f"The degphase ({np.unique(nPolyPhase)}) and degamp ({np.unique(nPolyAmp)}) parameters indicate the polynomial degree desired for the amplitude and phase solutions.")
    bcal_sol = np.full([len(antennasBP), len(freq_range), 2], 0.+1j*0 ,dtype=np.complex128)
    freq_rangeHz_ = np.array(freq_range)
    for index in antennasBP:
        # start and end frequency (hz) defining valid frequency domain
        scaleFactor_ = scaleFactor[index]
        validDomain_ = [frequencyLimits[index,0], frequencyLimits[index,1]]
        # X pol coefficients
        AmpCoeffX_ = polynomialAmplitude[index, 0:nPolyAmp[index]]
        PhaseCoeffX_ = polynomialPhase[index, 0:nPolyPhase[index]]
        # Y pol coefficients
        AmpCoeffY_ = polynomialAmplitude[index, nPolyAmp[index]:2*nPolyAmp[index]]
        PhaseCoeffY_ = polynomialPhase[index, nPolyPhase[index]:2*nPolyPhase[index]]
        # calculate CHEBYSHEV poly values
        bl=fitting.chebyshev(int(nPolyAmp[index])-1,
                             params=AmpCoeffX_,
                             xmin=frequencyLimits[index,0],
                             xmax=frequencyLimits[index,1])
        amplitudeSolutionX = bl.f(freq_range)
        amplitudeSolutionX = np.real(scaleFactor_) \
                           * amplitudeSolutionX \
                           + 1 - np.mean(amplitudeSolutionX[np.invert(flag_mask)])
        bl=fitting.chebyshev(int(nPolyAmp[index])-1,
                             params=AmpCoeffY_,
                             xmin=frequencyLimits[index,0],
                             xmax=frequencyLimits[index,1])
        amplitudeSolutionY = bl.f(freq_range)
        amplitudeSolutionY = np.real(scaleFactor_) \
                           * amplitudeSolutionY \
                           + 1 - np.mean(amplitudeSolutionY[np.invert(flag_mask)])
        bl=fitting.chebyshev(int(nPolyPhase[index])-1,
                             params=PhaseCoeffX_,
                             xmin=frequencyLimits[index,0],
                             xmax=frequencyLimits[index,1])
        phaseSolutionX = bl.f(freq_range)
        phaseSolutionX = np.real(scaleFactor_) \
                       * phaseSolutionX \
                       - np.mean(phaseSolutionX[np.invert(flag_mask)])  # rad
        bl=fitting.chebyshev(int(nPolyPhase[index])-1,
                             params=PhaseCoeffY_,
                             xmin=frequencyLimits[index,0],
                             xmax=frequencyLimits[index,1])
        phaseSolutionY = bl.f(freq_range)
        phaseSolutionY = np.real(scaleFactor_) \
                       * phaseSolutionY \
                       - np.mean(phaseSolutionY[np.invert(flag_mask)])  # rad

        antBX = np.array(amplitudeSolutionX*(np.cos(phaseSolutionX) \
              + 1j*np.sin(phaseSolutionX)), dtype=np.complex128)
        antBY = np.array(amplitudeSolutionY*(np.cos(phaseSolutionY) + \
              1j*np.sin(phaseSolutionY)), dtype=np.complex128)
        antB = np.column_stack([antBX.squeeze(), antBY.squeeze()])
        bcal_sol[index, :, :] = antB
    return bcal_sol


def map_main_tbl(
                 msfile,
                 bpolytable,
                 freq_range,
                 antennas,
                 flag_mask,
                 nrows=1,
                 ):
    """
    Populate Bcal main table column data

    @param msfile string     : Reference CASA MS name
    @param bpolytable string : CASA BPOLY table name
    @param freq_range Array : Observation Frequency range [Hz]
    @param antennas ndarray    : List of antenna names in array
    @param flag_mask ndarray   : Boolean array of generally flagged channels
    @return nrows int        : [Optional] Nr of rows in Bcal table

    @return mail_tbl_dict dict : Bcal table main table 
    """
    main_tbl_dict = {}
    # nrows [linux ts]
    main_tbl_dict["TIME"] = readcolumn(bpolytable, bp_field="TIME")
    # nrows [int]
    main_tbl_dict["FIELD_ID"] = readcolumn(bpolytable, bp_field="FIELD_ID")
    # nrows [int] ant index
    main_tbl_dict["ANTENNA1"] = readcolumn(bpolytable, bp_field="ANTENNA1")
    # nrows [int] ant index
    main_tbl_dict["ANTENNA2"] = readcolumn(bpolytable, bp_field="REF_ANT")
    # nrows [int]
    main_tbl_dict["INTERVAL"] = readcolumn(bpolytable, bp_field="INTERVAL")
    # nrows [int]
    main_tbl_dict["SCAN_NUMBER"] = readcolumn(bpolytable, bp_field="SCAN_NUMBER")
    # nrows [int]
    main_tbl_dict["OBSERVATION_ID"] = readcolumn(bpolytable, bp_field="OBSERVATION_ID")

    # nrows [varcol]
    spw_id = readcolumn(bpolytable+"::CAL_DESC", bp_field="SPECTRAL_WINDOW_ID")
    if isinstance(spw_id, dict):
        spw_id_varcol = makecolumndata(spw_id["r1"][0][0],
                                       arrsize=[nrows],
                                       dtype=spw_id["r1"][0][0].dtype,
                                       varcol=True)
        main_tbl_dict["SPECTRAL_WINDOW_ID"] = spw_id_varcol

    # 2 x nchans x nrows [complex] 
    main_tbl_dict["CPARAM"] =  bpolyfit(bpolytable, freq_range, antennas, flag_mask)

    return main_tbl_dict


def Bpoly2B(bpolytable, btable, template_btable):
    """
    Build a CASA B table from a CASA BPOLY table

    @param bpolytable string    : CARACal BPOLY table name
    @param btable string        : Output Bcal table name
    @param template_bcal string : Empty Bcal template table name

    @return btable string : CASA B table name
    """
#     caracal.log.info('Building a CASA B table from BPOLY table')
    # get various bits of metadata needed to extract and reconstruct information
    [msfile,
     nrows,
     Xpol_idx,
     Ypol_idx,
     freq_range,
     antennas,
     flag_mask] = metaref(bpolytable)

    # mapping of main table
    main_tbl_dict = map_main_tbl(
                                 msfile,
                                 bpolytable,
                                 freq_range,
                                 antennas,
                                 flag_mask,
                                 nrows=nrows,
                                 )

    # create empty BCAL table
    create_empty_B(btable, template_btable, nrows=nrows)

    # build Btable from BPOLY table
    writecolumndata(btable, main_tbl_dict)

    # copy relevant MS subtables over
    addsubtables(msfile, btable)

    return btable


if __name__ == '__main__':
    if (len(sys.argv)) < 3:
        print("Usage: {} <input_table_name.P> <output_table_name.B>".format(sys.argv[0]))
        sys.exit(0)

    bpolytable = sys.argv[1]
    btable = sys.argv[2]
    template_btable = "BCAL_TEMPLATE.B"

#     dirname = os.path.dirname(bpolytable)
#     tblname = os.path.basename(bpolytable)
#     print(dirname)
#     print(tblname)
#     basename, baseext = os.path.splitext(tblname)
#     print(basename, baseext)
#     btable="{}.B{}".format(basename, baseext[-1])
#     print(btable)

    Bpoly2B(bpolytable, btable, template_btable)

# -fin-
