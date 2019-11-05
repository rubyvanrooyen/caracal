import os, glob
import sys
import meerkathi
import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy import wcs

NAME = "Mosaic images"

def worker(pipeline, recipe, config):

    wname = pipeline.CURRENT_WORKER

    ##########################################
    # Defining functions for the worker
    ##########################################

    def identify_last_selfcal_image(directory_to_check, prefix, field, mfsprefix):
        
        # Doing this because convergence may have been reached before the user-specified number of iterations
        matching_files = glob.glob(directory_to_check+'/{0:s}_{1:s}_*{2:s}-image.fits'.format(prefix, field, mfsprefix)) # '*' to pick up the number
        max_num = 0  # Initialisation
        
        for filename in matching_files:
            split_filename = filename.split('_')
            number = split_filename[-1].split('-')[0]
            num = int(number)
            if num > max_num:
                max_num = num
        
        filename_of_last_selfcal_image = '{0:s}_{1:s}_{2:s}{3:s}-image.fits'.format(prefix, field, str(max_num),  mfsprefix)
        return filename_of_last_selfcal_image


    def build_beam(obs_freq,centre,cell,imsize,out_beam):  # Copied from masking_worker.py and edited

        #if copy_head == True:
        #    hdrfile = fits.open(headfile)
	#    hdr = hdrfile[0].header
	#elif copy_head == False:

        w = wcs.WCS(naxis=2)

        centre = coord.SkyCoord(centre[0], centre[1], unit=(u.deg, u.deg), frame='icrs') # Using u.deg for both due to using 'CRVAL1' and 'CRVAL2' to set the centre
        #cell /= 3600.0  # Am assuming that cell was passed to the function in units of arcsec, so this is converting it to units of deg.
        # Commenting the above out as 'CDELT2' from the corresponding image will be passed to the function, and this is already in deg.

        w.wcs.crpix = [ (imsize/2) + 1, (imsize/2) + 1 ] # The '+ 1's are needed to avoid a shape mismatch later on
        w.wcs.cdelt = np.array([-cell, cell])
        w.wcs.crval = [centre.ra.deg, centre.dec.deg]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

        hdr = w.to_header()
        hdr['SIMPLE']  = 'T'
        hdr['BITPIX']  = -32
        hdr['NAXIS']   = 2
        hdr.set('NAXIS1',  imsize, after='NAXIS')
        hdr.set('NAXIS2',  imsize, after='NAXIS1')

        if 'CUNIT1' in hdr:
            del hdr['CUNIT1']
        if 'CUNIT2' in hdr:
            del hdr['CUNIT2']

        dish_diameter = config['dish_diameter'] # Units of m. The default assumes that MeerKAT data is being processed
        pb_fwhm_radians = 1.02*( 2.99792458E8/obs_freq )/dish_diameter
        pb_fwhm = 180.0*pb_fwhm_radians/np.pi   # Now in units of deg
        pb_fwhm_pix = pb_fwhm/hdr['CDELT2']  
        x, y = np.meshgrid(np.linspace(-hdr['NAXIS2']/2.0,hdr['NAXIS2']/2.0,hdr['NAXIS2']),
				    np.linspace(-hdr['NAXIS1']/2.0,hdr['NAXIS1']/2.0,hdr['NAXIS1']))
        d = np.sqrt(x*x+y*y)
        sigma, mu = pb_fwhm_pix/2.35482, 0.0  # sigma = FWHM/sqrt(8ln2)
        gaussian = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

        fits.writeto(out_beam,gaussian,hdr,overwrite=True)


    ##########################################
    # Main part of the worker
    ##########################################

    # Prioritise parameters specified in the config file, under the 'mosaic' worker   ### How do we get them from the schema instead?
    specified_mosaictype = config['mosaic_type'] # i.e. 'continuum' or 'spectral'
    use_MFS_images = config['use_MFS_images']
    #specified_prefix = config['name'] ### Come back to. I'd said in the schema that this does not need to be specified...
    specified_images = config['target_images'] ### Test at a later point

    # Parameters that depend on the mosaictype
    if specified_mosaictype == 'continuum':
        label = 'corr' #pipeline.config['self_cal'].get('label', 'corr') ### Did not work as expected. Come back this 
        pb_origin = 'generated by the mosaic_worker'  ### Will change this to observation_config_worker once the necessary work is done
    elif specified_mosaictype == 'spectral':
        label = 'corr' #pipeline.config['image_HI'].get('label', 'corr') ### As above
        pb_origin = 'generated by the image_HI_worker'
    else:
        label = 'corr' #pipeline.config['self_cal'].get('label', 'corr') ### As above
        meerkathi.log.info("Mosaic_type was not set to either 'continuum' or 'spectral' in the config file, so proceeding with default mode ('continuum').")
        specified_mosaictype = 'continuum'

    # To ease finding the appropriate files, and to keep this worker self-contained
    if use_MFS_images == 'true':
        mfsprefix = '-MFS'
    else:
        mfsprefix = ''
    
    #pipeline.prefixes = ['{0:s}-{1:s}-{2:s}'.format(pipeline.prefix,did,config['label']) for did in pipeline.dataid]
    # In case there are different pipeline prefixes
    #for i in range(len(pipeline.prefixes)): ### I may need to put this loop back in later

    prefix = pipeline.prefix

    if specified_images is None: ### Check that this is the correct way to check that nothing is passed via the config file
         
        print('For this test, I do not expect to see this message, as image names are passed via the config file')

        #Needed for working out the field names for the targets 
        all_targets, all_msfile, ms_dict = utils.target_to_msfiles(pipeline.target,pipeline.msnames,label) 

        # Due to the way the output is now sorted, need to know the total number of targets
        n_targets = len(all_targets)  ### Assuming that all_targets is a list or an array
        meerkathi.log.info('The number of targets to be mosaicked is {0:d}'.format(n_targets))
        subdirectory_number = 0

        # Empty list to add filenames to
        pathnames = []
        specified_images = []

        # Expecting the same prefix and mfsprefix to apply for all fields to be mosaicked together
        for target in all_targets:

            field = utils.filter_name(target)
            subdirectory_number += 1 

            # Use the mosaictype to infer the filenames of the images
            if specified_mosaictype == 'continuum':  # Add name of 2D image output by selfcal

                path_to_image = pipeline.output + '/continuum/image_' + str(subdirectory_number)
                pathnames = pathnames.append(path_to_image)
                image_name = identify_last_selfcal_image(path_to_image, prefix, field, mfsprefix)
                specified_images = specified_images.append(image_name) # Note that the path is not included in image_name

            else:  # i.e. mosaictype = 'spectral', so add name of cube output by imageHI

                path_to_cube = pipeline.output + '/cubes/cube_' + str(subdirectory_number)
                pathnames = pathnames.append(path_to_cube)
                image_name = '{0:s}_{1:s}_HI{2:s}-image.fits'.format(prefix, field, mfsprefix)
                if mfsprefix == '':
                    image_name = image_name.replace('-image','.image') # Following the naming in image_HI_worker   
                specified_images = specified_images.append(image_name) # Note that the path is not included in image_name

    else: ### i.e. if the user has specified images, I start by assuming that they are all in the main 'output' directory

        #pathnames = [ pipeline.output ] * len(specified_images) 

          ### Now testing the scenario where the common directory is either output/continuum or output/cubes
        if specified_mosaictype == 'continuum':
            pathnames = [ pipeline.output+'/continuum' ] * len(specified_images)
        else:
            pathnames = [ pipeline.output+'/cubes' ] * len(specified_images)

    
    meerkathi.log.info('Images to be mosaicked are:')
    meerkathi.log.info(specified_images)
    print('Corresponding pathnames = ', pathnames)


    # Although montage_mosaic checks whether pb.fits files are present, we need to do this earlier in the worker,
    # so that we can create simple Gaussian primary beams if need be 
    for image_name in specified_images: 
        ### WARNING: pathnames is currently not defined if user has specified the images to use 
        
        pb_name = image_name.replace('image.fits', 'pb.fits')
        
        # Need the corresponding pathname for the image being considered
        index_to_use = specified_images.index(image_name)
        pathname = pathnames[index_to_use]

        if os.path.exists(pathname + '/' + pb_name):
            meerkathi.log.info('{0:s}/{1:s} is already in place, and will be used by montage_mosaic.'.format(pathname,pb_name))
            
        else:
                
            if specified_mosaictype == 'spectral':
                meerkathi.log.error('{0:s}/{1:s} does not exist. Please make sure that it is in place before proceeding. EXITING.'.format(pathname,pb_name))
                sys.exit(1)
                
            else: # i.e. mosaictype = 'continuum'
                
                meerkathi.log.info('{0:s}/{1:s} does not exist, so going to create a rudimentary pb.fits file instead.'.format(pathname,pb_name))

                # Create rudimentary primary-beam, which is assumed to be a Gaussian with FWMH = 1.02*lambda/D
                image_hdu = fits.open(pathname + '/' + image_name) ### Hmm, didn't have this prefixed with pipeline.output before 
                image_header = image_hdu[0].header
                image_centre = [ image_header['CRVAL1'], image_header['CRVAL2'] ] # i.e. [ RA, Dec ]. Assuming that these are in units of deg.
                image_cell = image_header['CDELT2'] ### Need to ensure that this value is a float?  # Again assuming that these are in units of deg.
                image_imsize = image_header['NAXIS1']  ### Need to ensure that this value is an integer?

                recipe.add(build_beam, 'Build Gaussian, 2D primary-beam',
		    {
		        'obs_freq' : config['ref_frequency'], # Units of Hz. The default assumes that MeerKAT data is being processed
		        'centre'   : image_centre,
		        'cell'     : image_cell,
		        'imsize'   : image_imsize,
		        'out_beam' : pathname + '/' + pb_name,
		    },
		    input=pipeline.input,
                    output=pathname, # Was pipeline=pipeline.output before the restructure of the output directory
                    label='build_beam:: Generating {0:s}'.format( pathname + '/' + pb_name ))

                pb_origin = 'generated by the mosaic_worker'   
                
                # Confirming freq and dish_diameter values being used for the primary beam
                meerkathi.log.info('Observing frequency = {0:f} Hz, dish diameter = {1:f} m'.format( config['ref_frequency'], config['dish_diameter'] )) 

    meerkathi.log.info('Checking for *pb.fits files now complete.')

    # Need to tell MeerKATHI where to look in the output folder for the images to be mosaicked
    if specified_mosaictype == 'continuum':
        input_directory = pipeline.continuum
    else:
        input_directory = pipeline.cubes

    # List of images in place, and have ensured that there are corresponding pb.fits files,
    # so now ready to add montage_mosaic to the meerkathi recipe
    if pipeline.enable_task(config, 'domontage'):
        recipe.add('cab/montage_mosaic', 'montage_mosaic',
            {
                "mosaic-type"    : specified_mosaictype,
                "domontage"      : True,
                "cutoff"         : config.get('cutoff'),
                "name"           : prefix,
                "target-images"  : specified_images,
            },
            input=input_directory,
            output=pipeline.mosaics,
            label='montage_mosaic:: Re-gridding {0:s} images before mosaicking them. For this mode, the mosaic_worker is using *pb.fits files {1:s}.'.format(specified_mosaictype, pb_origin))
        
    else:  # Written out for clarity as to what difference the 'domontage' setting makes 
        recipe.add('cab/montage_mosaic', 'montage_mosaic',
            {
                "mosaic-type"    : specified_mosaictype,
                "domontage"      : False,
                "cutoff"         : config.get('cutoff'),
                "name"           : prefix,
                "target-images"  : specified_images,
            },
            input=input_directory,
            output=pipeline.mosaics,
            label='montage_mosaic:: Re-gridding of images and beams is assumed to be already done, so straight to mosaicking {0:s} images. For this mode, the mosaic_worker is using *pb.fits files {1:s}.'.format(specified_mosaictype, pb_origin))
     
    ### Leaving the following as a reminder of syntax    
    #if pipeline.enable_task(config, 'add_spectral_weights'):
        #step = 'estimate_weights_{:d}'.format(i)
        #recipe.add('cab/msutils', step,
        #    {
        #      "msname"          : msname,
        #      "command"         : 'estimate_weights',
        #      "stats_data"      : config['add_spectral_weights'].get('stats_data', 'use_package_meerkat_spec'),
        #      "weight_columns"  : config['add_spectral_weights'].get('weight_columns', ['WEIGHT', 'WEIGHT_SPECTRUM']),
        #      "noise_columns"   : config['add_spectral_weights'].get('noise_columns', ['SIGMA', 'SIGMA_SPECTRUM']),
        #      "write_to_ms"     : config['add_spectral_weights'].get('write_to_ms', True),
        #      "plot_stats"      : prefix + '-noise_weights.png',
        #    },
        #    input=pipeline.input,
        #    output=pipeline.output,
        #    label='{0:s}:: Adding Spectral weights using MeerKAT noise specs ms={1:s}'.format(step, msname))


