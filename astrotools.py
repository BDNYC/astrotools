'''
The module **astrotools** is a set of functions for astrophysical analysis developed by Kelle Cruz's team at Hunter College and the American Museum of Natural History in New York City. It consists of an amalgamation of functions tailored primordialy to handle fits file spectral data.

:Authors:
	Dan Feldman, Alejandro Nunez, Damian Sowinski
:Version:
    1.0 of 2012/03/16
:Repository:
    https://github.com/BDNYC/astrotools (for access, contact jfilippazzo@gmail.com)

'''

#+++++++++++++++++++++++++++++ RELEVANT MODULES +++++++++++++++++++++++++++++++

import asciidata
import matplotlib
import numpy
import os
import pyfits
import pylab
import matplotlib.pyplot as plt
import scipy
import types
from scipy import interpolate
import pdb

#+++++++++++++++++++++++++++++++++ PATHS ++++++++++++++++++++++++++++++++++++++

# Folder where spectra are located (.fits files)
spectralDirectory = '/Users/memnhok/Data/Optical Data/Good Objects/'

# Folder where filter files are located (.dat and .txt files)
filterDirectory = '/Users/memnhok/Data/Filter Data/LSST baseline/Total/'

# Folder where Vega spectrum is located (needed to normalize the magnitudes)
normalizationPath = '/Users/memnhok/Data/Norm Spectra/Vega.dat'

# Folder to store magnitude list output
creationDirectory = '/Users/memnhok/Python/Magnitude files/'

# File with magnitude list (used by the maganal class)
magsfile = '/Users/memnhok/Python/Magnitude files/mags_RIZYJHK4.txt'

#+++++++++++++++++++++++++++ GLOBAL FUNCTIONS +++++++++++++++++++++++++++++++++
def avgFlux( startW, endW, SpecData, sum=True, median=False, verbose=True):
    '''
    (by Damian & Dan)
    Calculate the average flux of ... something.
    
    PARAMETERS
    
    OUTPUT
    
    '''
    
    Wavelength_big = SpecData[0]
    Flux_big = SpecData[1]
    
    if not(isinstance(Wavelength_big, numpy.ndarray)):
        Wavelength_big = numpy.array( Wavelength_big)
    if not(isinstance(Flux_big, numpy.ndarray)):
        Flux_big = numpy.array( Flux_big)
    # See if the wavelength range falls inside the wavelength array
    if numpy.min(Wavelength_big) > startW or numpy.max(Wavelength_big) < endW:
        if verbose == True:
            print "avgFlux: wavelength interval out of range"
        return
    # See that wavelength range does not fall between data points in 
    # wavelength array
    set1 = set(list(numpy.where( Wavelength_big >= startW )[0]))
    newEnd = endW + .0022
    set2 = set(list(numpy.where(Wavelength_big <= endW + .0022)[0]))
    temp = numpy.array(list(set1.intersection(set2)))
    if len(temp) == 0:
        if verbose  == True:
            print "avgFlux: there is no data in the selected interval"
        return
    
    # Winds the pixel scale
    temp = numpy.where(Wavelength_big >= startW)[0]
    pix_scale = Wavelength_big[temp[0]+1] - Wavelength_big[temp[0]]
    
    # Find wavelengths in array that are in interval and make a new array
    set1 = set(list(numpy.where( Wavelength_big + pix_scale/2 >= startW )[0]))
    set2 = set(list(numpy.where(Wavelength_big - pix_scale / 2 <= endW)[0]))
    # For some reason, temp1 may be slightly out of order, so sort it:
    temp1 = numpy.array(list(set1.intersection(set2)))
    temp  = numpy.sort(temp1)
    Wavelength = Wavelength_big[temp]
    Flux = Flux_big[temp]
    num_pixels = len(Wavelength)
    first = 0
    last = num_pixels - 1
    if num_pixels >= 1:
    # Determine fractional pixel value on the edge of the region
        frac1 = ( Wavelength[first] + pix_scale / 2 - startW ) / pix_scale
        frac2 = ( endW - Wavelength[last] + pix_scale / 2 ) / pix_scale
        #sums the fluxes in the interval
        sumflux = 0
        for n in numpy.arange(first, num_pixels):
            if n == first:
                pixflux = frac1 * Flux[n]
            elif n == last:
                pixflux = frac2 * Flux[n]
            else:
                pixflux = Flux[n]
            sumflux = sumflux + pixflux
        realpix = num_pixels - 2 + frac1 + frac2
        avgflux = sumflux / realpix
    
    else:
        frac = (endW - startW) / pix_scale
        avgflux = frac * Flux[0]
    
    if median == True and num_pixels > 5:
        if verbose == True:
            print "median worked"
        old = avgflux
        avgflux = numpy.median(Flux)
    
        if 100 * numpy.abs( avgflux - old ) / old > 3:
            print "avgFlux: WARNING: difference between average and median ' \
                  & 'is greater than 3%"
            print "avgFlux: median = " + str(avgflux) + ",\t"+ " average = " \
                  + str(old)
            print "avgFlux: difference % = " + \
                  str(100 * numpy.abs( avgflux - old ) / old)
        else:
            if verbose == True:
                print "median worked"
    
    return avgflux


def degtotime(degInput):
	'''
	*by Damian*
	
	Convert degrees to arctime. The output is a string of the time up to seconds.
	
	*degInput*
	    float number with Right Ascension format.
	'''
	
	sign = numpy.sign(degInput)
	degInput = numpy.abs(float(degInput))
	while degInput > 360:
		degInput = degInput - 360
		print degInput
	
	hours = int(degInput / 15)
	degInput = degInput / 15 - hours
	minutes  = int(degInput * 60)
	degInput = degInput * 60 - minutes
	seconds  = degInput * 60 
	
	time = str(sign*hours) + ":" + str(minutes) + ":" + '%.2f'%seconds
	
	return time


def integrate(xyData):
    '''
    *by Damian*
    
    Integrate x and y data treating it as a scatter plot with the trapezoid rule. the output is a float number.
    
    *xyData*
        2-D array with x-data in position 0 and y-data in position 1.
    '''
    
    try:
        if len(xyData) != 2:
            print 'Cannot integrate, object does not have necessary parameters.'
            return 
        else:
            xData = xyData[1]
            yData = xyData[0]
            integral = 0.
            try:
                length = len(xData)
                for n in range (0, length - 1):
                    integral = integral + (xData[n + 1] - xData[n]) * \
                               (yData[n + 1] + yData[n]) * 0.5
            except ValueError:
                print 'Data type cannot be integrated'
                return
    except TypeError:
        print 'Cannot integrate.'
        return
    
    return integral


def mean_comb(spectra):
    '''
    *by Alejandro Nunez*
    
    Combine spectra using a weighted mean. **Uncertainties are required** for this function. The mask wavelength array will be that of the first spectrum in the *spectra* list. The output is a python list with mask wavelength in position 0, combined flux in position 1, and combined uncertainties in position 2.
    
    This function mimics mc_meancomb (by Mike Cushing), with some restrictions.
    
    *spectra*
        Python list of spectra, where each spectrum is a python list as well, having wavelength in position 0, flux in position 1 and uncertainties in position 2. **Important**: flux array cannot have nan values.
    '''
    
    # 1. Interpolate all spectra using mask
    # Use x-axis (i.e. wl) values of first spectrum as mask for all others
    wl_mask   = spectra[0][0]
    numPoints = len(wl_mask)
    numSpec = len(spectra)
    
    # 3D-array that will hold interpolated spectra
    # (it omits wl axis, since all spectra have the same one)
    ip_spectra = numpy.zeros((numPoints, 2, numSpec)) * numpy.nan
    
    for spIdx, spec in enumerate(spectra):
        wls    = spec[0]
        fluxes = spec[1]
        errors = spec[2]
        
        if spIdx == 0:
            # No need to interpolate first spectrum
            fluxes_new = fluxes
            errors_new = errors
        else:
            ip_func_flux = scipy.interpolate.interp1d(wls, fluxes, bounds_error=False)
            ip_func_err  = scipy.interpolate.interp1d(wls, errors, bounds_error=False)
            fluxes_new = ip_func_flux(wl_mask.tolist())
            errors_new = ip_func_err(wl_mask.tolist())
        
        ip_spectra[:,0,spIdx] = fluxes_new
        ip_spectra[:,1,spIdx] = errors_new
    
    # 2. Calculate weighted mean of flux values
    mvar = 1. / numpy.nansum(1. / ip_spectra[:,1,:], axis=1)
    mean = numpy.nansum(ip_spectra[:,0,:] / ip_spectra[:,1,:], axis=1) * mvar
    
    return [wl_mask, mean, mvar]


def read_spec(specFiles, aToMicron=False, negToZero=False, normal=False, errors=False, plot=False, warn=False):
    '''
    *by Alejandro Nunez*
    
    Read spectral data from fits files. It returns a python list with wavelength in position 0, flux in position 1 and error values (if available and requested) in position 2. More than one fits file name can be provided simultaneously.
    
    *specFiles*
        fits file name (with full path) as string; it can also be a python list of file names.
    *aToMicron*
        If wavelength in fits file is in Angstrom, convert wavelength values into micrometers.
    *negToZero*
        Set any negative flux values equal to zero.
    *normal*
        Normalize the flux values using the simple average of all flux data points.
    *errors*
        Return error values for the flux data; return nans if unavailable.
    *plot*
        Plot the spectral data, including error bars when available
    *warn*
        Show warning messages
    '''
    import astrotools as at
    
    # 1. Convert specFiles into a list type if it is a string
    if isinstance(specFiles,types.StringTypes):
        specFiles = [specFiles,]
    
    try:
        specFiles[0]
    except TypeError:
        print 'File name is an unrecognizable python type.'
        return
    
    # 2. Initialize array to store spectral
    specData = [None] * len(specFiles)
    
    # 3. Loop through each file name:
    for spFileIdx,spFile in enumerate(specFiles):
        
        # 3.1. Get data from fits file
        try:
            fitsData, fitsHeader = pyfits.getdata(spFile, header=True)
        except IOError:
            print 'FITS FILE: ' + spFile + ' was not found.'
            continue
        
        # 3.2. Check if data in fits file is linear (if not, don't use data)
        KEY_TYPE = ['CTYPE1']
        setType  = set(KEY_TYPE).intersection(set(fitsHeader.keys()))
        if len(setType) == 0 and warn:
            print 'Flux data in ' + spFile + ' assumed to be linear.'
        if len(setType) != 0:
            valType = fitsHeader[setType.pop()]
            if valType.strip().upper() != 'LINEAR':
                print 'Flux data in file ' + spFile \
                      + ' is NOT linear and will not be used.'
                continue
        
        # 3.3. Get flux, error data, and get wl data when available
        #      (returns wl in pos. 0, flux in pos. 1, error values in pos. 2)
        specData[spFileIdx] = get_spec(fitsData, fitsHeader, spFile, errors)
        
        if specData[spFileIdx] is None:
            continue
        
        # 3.4. Generate wl axis when needed
        if specData[spFileIdx][0] is None:
            specData[spFileIdx][0] = create_waxis(fitsHeader, \
                                     len(specData[spFileIdx][1]), spFile)
        
        # If no wl axis generated, then clear out all retrieved data for object
        if specData[spFileIdx][0] is None:
            specData[spFileIdx] = None
            continue
        
        # 3.5. Convert units in wl-axis from Angstrom into microns if desired
        if aToMicron:
            if specData[spFileIdx][0][-1] > 8000:
                specData[spFileIdx][0] = specData[spFileIdx][0] / 10000
       
       # 3.6. Set zero flux values as nans
        zeros = numpy.where(specData[spFileIdx][1] == 0)
        if len(zeros[0]) > 0:
            specData[spFileIdx][1][zeros] = numpy.nan
        
        # 3.7. Set negative flux values equal to zero if desired
        if negToZero:
            negIdx = numpy.where(specData[spFileIdx][1] < 0)
            if len(negIdx[0]) > 0:
                specData[spFileIdx][1][negIdx] = 0
                if warn:
                    print 'FLUX DATA: %i negative data points found in %s.' \
                            % (len(negIdx[0]), spFile)
        
        # 3.8. Normalize flux data, if requested
        if normal:
            specData[spFileIdx] = normalize(specData[spFileIdx])
    
    # 4. Plot the flux data if desired
    if plot:
        plot_spec(specData, normal)
    
    # 5. Clear up memory
    fitsHeader = ''
    fitsData   = ''
    
    return specData


def sel_band(specData, limits, bandName='NA', objID='NA', normalizer=False):
    '''
    *by Alejandro Nunez*
    
    There are two mutually exclusive functions that *sel_band* accomplishes. If *normalizer=False*, it **selects** a region from a spectrum specified by *limits*. If *normalizer=True*, it **normalizes** a spectrum using the region of the spectrum specified by *limits*.
    
    *specData*
        Spectrum as a Python list with wavelength in position 0, flux in position 1, and optional error values in position 2. More than one spectrum can be provided simultaneously, in which case *specData* shall be a list of lists.
    *limits*
        Python list with lower limit in position 0 and upper limit in position 1.
    *bandName*
        String with name of the spectrum band. For error printing purposes only.
    *objID*
        String with ID for spectrum; if dealing with several spectra, *objID* shall be a list of strings. For error printing purposes only.
    *normalizer*
       If *False*, select a region from a spectrum using *limits*; if *True*, normalize a spectrum using the region of the spectrum specified by *limits*.
    '''
    
    # Convert specData to list or spectra if it consists only of one
    if len(specData) <= 3 and len(specData[0]) > 10:
        specData = [specData]
    
    # Initialize objects
    finalData = [None] * len(specData)
    
    # Check that given limits are reasonable
    if limits[0] >= limits[1]:
        print 'sel_band: the Min and Max values specified for the ' \
              + bandName + ' band are not reasonable.'
        return finalData
    
    # Re-define normalizing band (specified in limits) in the case when the
    # the limits fall outside of the spectrum limits itself
    if normalizer:
        smallest = limits[0]
        largest  = limits[1]
        for spData in specData:
            if spData is None:
                continue
            
            tmpNans = numpy.where(numpy.isfinite(spData[1]))
            if len(tmpNans[0]) != 0:
                if spData[0][tmpNans[0][0]] > smallest:
                    smallest = spData[0][tmpNans[0][0]]
                if spData[0][tmpNans[0][-1]] < largest:
                    largest = spData[0][tmpNans[0][-1]]
        
        limits[0] = smallest
        limits[1] = largest
    
    # Loop through each spectral data set
    for spIdx, spData in enumerate(specData):
        
        # 1) Skip if data is missing
        if spData is None:
            continue
        
        # 2) Determine if spectra come with error values
        if len(spData) == 3:
            errors = True
        else:
            errors = False
        
        # 3) Determine minimum wavelength value for band
        smallIdx = numpy.where(spData[0] < limits[0])
        
        # If lower limit < all values in spectrum wavelength points, then
        # make band's minimum value = first data point in spectrum
        if len(smallIdx[0]) == 0:
            minIdx = 0
        
        # If lower limit > all values in spectrum wavelength points, then
        # no band can be selected
        elif len(smallIdx[0]) == len(spData[0]):
            print 'sel_band: the wavelength data for the ' + bandName + \
               ' band for object %s is outside the given limits.' %objID[spIdx]
            continue
        else:
            minIdx = smallIdx[0][-1] + 1
        
        # 4) Determine maximum wavelength value for band
        largeIdx = numpy.where(spData[0] > limits[1])
        
        # If upper limit > all values in spectrum wavelength points, then
        # make band's maximum value = last data point in spectrum
        if len(largeIdx[0]) == 0:
            maxIdx = len(spData[0])    
        
        # If upper limit < all values in spectrum wavelength points, then
        # no band can be selected
        elif len(largeIdx[0]) == len(spData[0]):
            print 'sel_band: the wavelength data for the ' + bandName + \
               ' band for object %s is outside the given limits.' %objID[spIdx]
            continue
        else:
            maxIdx = largeIdx[0][0]
        
        # 5) Check for consistency in the computed band limits
        if maxIdx - minIdx < 2:
            print 'sel_band: The Min and Max values specified for the ' \
                + bandName + ' band for object %s yield no band.' %objID[spIdx]
            continue
            
        # 6) Select flux band from spectrum
        fluxSelect = spData[1][minIdx:maxIdx]
        
        # 7) Select error value band from spectrum
        if errors is True:
            errorSelect = spData[2][minIdx:maxIdx]
        
        # 8) Normalize spectrum if requested (using arithmetic mean)
        if normalizer:
            notNans = numpy.where(numpy.isfinite(fluxSelect))
            avgFlux = numpy.mean(fluxSelect[notNans])
            finalFlux = spData[1] / avgFlux
            
            finalData[spIdx] = [spData[0], finalFlux]
            
            if errors is True:
                notNans  = numpy.where(numpy.isfinite(errorSelect))
                avgError = numpy.mean(errorSelect[notNans])
                finalErrors = spData[2] / avgError
                
                finalData[spIdx] = [spData[0], finalFlux, finalErrors]
        else:
            if errors is True:
                finalData[spIdx] = [spData[0][minIdx:maxIdx], fluxSelect, \
                                    errorSelect]
            else:
                finalData[spIdx] = [spData[0][minIdx:maxIdx], fluxSelect]
    
    return finalData


def smooth_spec(specData, specFiles=None, goodRes=200, winWidth=10):
    '''
    *by Alejandro Nunez*
    
    Smooth flux data to resolution specified. The original spectrum resolution is determined more accurately by reading the metadata of the spectrum, and so this function prefers access to the fits file from where the spectrum was obtained, but it is not necessary.
    
    *specData*
        Spectrum as a Python list with wavelength in position 0, flux in position 1, and optional error values in position 2. More than one spectrum can be provided simultaneously, in which case *specData* shall be a list of lists.
    *specFiles*
        String with name of the fits file (with full path) from where the spectrum was obtained; if dealing with several spectra, *specFiles* shall be a list of strings.
    *goodRes*
        Float with Signal-to-Noise resolution desired.
    *winWidth*
        Float with width of smoothing window; used when spectrum resolution is unknown.
    '''
    
    import scipy.ndimage
    
    # Convert into python list type when only one set of spectrum & fits-file
    if isinstance(specFiles,types.StringTypes):
        specFiles = [specFiles,]
        specData  = [specData,]
    
    for specIdx,spec in enumerate(specData):
        if spec is not None:
            if specFiles[specIdx] is not None:
                # Get RES data from fits file header
                fitsData = pyfits.open(specFiles[specIdx])
                try:
                    origRes = int(fitsData[0].header['res'])
                except KeyError:
                    origRes = 0
                    fitsData.close()
            else:
                origRes = 0
            
            # Determine width of smoothing window
            width = 0 # Width of smoothing window
            if origRes > goodRes:
                width = origRes / goodRes
            elif origRes == 0:
                width = winWidth
            
            # Reduce spectrum resolution if greater than goodRes
            if width > 0:
                notNans = numpy.where(numpy.isfinite(spec[1]))
                spec[1][notNans] = scipy.ndimage.filters.uniform_filter( \
                                   spec[1][notNans], size=width)
    
    return specData


#+++++++++++++++++++++++++ SECONDARY FUNCTIONS ++++++++++++++++++++++++++++++++

def plot_spec(specData, normal):
# *Function used by read_spec only*
# (by Alejo)
# Plots spectral data together.
    
    # Check that there is data to plot
    allNone = True
    for spData in specData:
        if spData is not None:
            allNone = False
            break
    if allNone:
        return
    
    # Initialize figure
    plt.figure(1)
    plt.clf()
    
    # Set plot titles
    TITLE   = 'SPECTRAL DATA'
    X_LABEL = 'Wavelength'
    Y_LABEL = 'Flux'
    if normal:
        Y_LABEL = Y_LABEL + ' (Normalized)'
    
    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    
    # Plot data
    for spData in specData:
        if spData is not None:
            if len(spData) == 3:
                plt.errorbar(spData[0], spData[1], spData[2], capsize=2, hold=True)
            else:
                plt.plot(spData[0], spData[1], drawstyle='steps-mid', hold=True)
    

def normalize(specData):
# *Function used by read_spec only*
# (by Alejo)
# Normalizes the flux (and sigma if present) data using the mean flux of the
# whole set.
    
    specFlux = specData[1]
    
    if len(specData) == 3:
        specSigma = specData[2]
    
    nonNanFlux = specFlux[numpy.isfinite(specFlux)]
    avgFlux = numpy.mean(nonNanFlux)
    
    specData[1] = specFlux / avgFlux
    if len(specData) == 3:
        specData[2] = specSigma / avgFlux
    
    return specData
    

def create_waxis(fitsHeader, lenData, fileName):
# *Function used by read_spec only*
# (by Alejo)
# Generates a wavelength (wl) axis using header data from .fits file.
    
    # Define Key names in
    KEY_MIN  = ['CRVAL1']            # Min wl
    KEY_DELT = ['CDELT1','CD1_1']    # Delta of wl
    KEY_OFF  = ['LTV1']              # Offset in wl to subsection start
    
    # Find Key names for minimum wl, delta, and wl offset in .fits file header
    setMin  = set(KEY_MIN).intersection(set(fitsHeader.keys()))
    setDelt = set(KEY_DELT).intersection(set(fitsHeader.keys()))
    setOff  = set(KEY_OFF).intersection(set(fitsHeader.keys()))
    
    # Get the values for minimum wl, delta, and wl offset, and generate axis
    if len(setMin) >= 1 and len (setDelt) >= 1:
        keyName = setMin.pop()
        valMin  = fitsHeader[keyName]
        
        keyName = setDelt.pop()
        valDelt = fitsHeader[keyName]
        
        if len(setOff) == 0:
            valOff = 0
        else:
            keyName = setOff.pop()
            valOff  = fitsHeader[keyName]
        
        # generate wl axis
        wAxis = (numpy.arange(lenData) * valDelt) + valMin - (valOff * valDelt)
        
    else:
        wAxis = None
        print 'create_waxis: Could not re-create a wavelength axis for ' \
              + fileName + '.'
    
    return wAxis
    

def get_spec(fitsData, fitsHeader, fileName, errorVals):
# *Function used by read_spec only*
# (by Alejo)
# Interprets spectral data from .fits file.
# Returns wavelength (wl) data in pos. 0, flux data in pos. 1, 
# and if requested, sigma (error) values in pos. 2.
    
    if errorVals:
        validData = [None] * 3
    else:
        validData = [None] * 2
    
    # Identify number of data sets in .fits file
    dimNum = len(fitsData)
    
    # Identify data sets in .fits file
    fluxIdx  = None
    waveIdx  = None
    sigmaIdx = None
    
    if dimNum == 1:
        fluxIdx = 0
    elif dimNum == 2:
        waveIdx = 0
        fluxIdx = 1
    elif dimNum == 3:
        waveIdx  = 0
        fluxIdx  = 1
        sigmaIdx = 2
    elif dimNum == 4:
    # Data sets in fits file are: 0-flux clean, 1-flux raw, 2-background,
    #                             3-sigma clean
        fluxIdx  = 0
        sigmaIdx = 3
    elif dimNum > 10:
    # Implies that only one data set in fits file: flux
        fluxIdx = -1
        if numpy.isscalar(fitsData[0]):
            fluxIdx = -1
        elif len(fitsData[0]) == 2:
        # Data comes in a xxxx by 2 matrix (ascii origin)
            tmpWave = []
            tmpFlux = []
            for pair in fitsData:
                tmpWave.append(pair[0])
                tmpFlux.append(pair[1])
            fitsData = [tmpWave,tmpFlux]
            fitsData = numpy.array(fitsData)
            
            waveIdx = 0
            fluxIdx = 1
        else:
        # Indicates that data is structured in an unrecognized way
            fluxIdx = None
    else:
        fluxIdx = None
        
    # Fetch wave data set from .fits file
    if fluxIdx is None:
    # No interpretation known for .fits file data sets
        validData = None
        print 'get_spec: Unable to interpret data sets in ' + fileName + '.'
        return validData
    else:
        if waveIdx is not None:
            if len(fitsData[waveIdx]) == 1: # Data set may be a 1-item list
                validData[0] = fitsData[waveIdx][0]
            else:
                validData[0] = fitsData[waveIdx]
    
    # Fetch flux data set from .fits file
    if fluxIdx == -1:
        validData[1] = fitsData
    else:
        if len(fitsData[fluxIdx]) == 1:
            validData[1] = fitsData[fluxIdx][0]
        else:
            validData[1] = fitsData[fluxIdx]
    
    # Fetch sigma data set from .fits file, if requested
    if errorVals:
        if sigmaIdx is None:
            validData[2] = numpy.array([numpy.nan] * len(validData[1]))
        else:
            if len(fitsData[sigmaIdx]) == 1:
                validData[2] = fitsData[sigmaIdx][0]
            else:
                validData[2] = fitsData[sigmaIdx]
        
        # If all sigma values have the same value, then replace them with nans
        if validData[2][10] == validData[2][11] == validData[2][12]:
            validData[2] = numpy.array([numpy.nan] * len(validData[1]))
    
    return validData
    

def getwaveunit(minWave):
# *Function used by readfilt only*
# (by Damian)
# Recognizes the unit being used for the wavelength data in a filter.
	
	if float(minWave) < 3000:
		return 10
	else:
		return 1

def createspectra(spectralPath):
# *Function used by photo_calc.__init__ only*
# (by Damian)
# Endows object with all the spectral data.
	
	objectNames = os.listdir(spectralPath)
	
	for Name in objectNames:
		if Name.startswith('.'):
			objectNames.pop(objectNames.index(Name))
	
	objectList = [None] * len(objectNames)
	
	for n in range (0,len(objectList)):
		objectList[n] = read_spec(spectralPath + objectNames[n])
	
	return objectList

def createfilters(filterPath):
# *Function used by photo_calc.__init__ only*
# (by Damian)
# Endows object with all the filter data.
	import astrotools as at
	
	filterNames = os.listdir(filterPath)
	filterList = [None] * len(filterNames)
	
	for n in range (0,len(filterNames)):
		
		if filterNames[n].startswith('.'):
			continue
		else:
			filterList[n] = at.readfilt(filterPath + filterNames[n])
			print filterNames[n]
	
	return filterList

def readfilt(filtFile, checkwave=False):
# (by Damian)
# Reads transmittance data from .txt and .dat file(s); it returns a list.
	
	import astrotools as at
	
	#This piece cleans up the file, removing any header info
	headerList = []
	tempList = open(filtFile).readlines()
	
	for n in range (0,len(tempList)):
		if tempList[n][0] == '#':
			headerList.insert(0,n)
	
	for n in range (0,len(headerList)):
		tempList.pop(headerList[n])
	
	#This part will extract the data from the cleaned up file
	filterData = [[],[]]
	
	if checkwave == False:
		waveCorrection = at.getwaveunit( tempList[0].split()[0] )
	else:
		waveCorrection = 1
	
	for n in range (0, len(tempList)):
		tempLine = tempList[n].split()
		try:
			filterData[1].append( waveCorrection * float(tempLine[0]) )
			filterData[0].append( float(tempLine[1]) )
		except ValueError:
			print filtFile
		
	return filterData

def filtNormSpec(filterFile, normFile=None):
# (by Damian)
# Uses the spectrum of Vega (or other star) & convolves it with a filter band.
	import astrotools as at
	
	if isinstance(filterFile, str):
		try:
			filterFile = at.readfilt(filterFile)
		except IOError:
			errfile(filterFile)
	
	if normFile == None:
		normFile = normalizationPath
		normData = at.readfilt(normFile,True)
	elif isinstance(normFile,str):	
		fileType = normFile.split('.')[-1]
		if fileType == 'dat' or fileType == 'txt':
			normData = at.readfilt(vegaFile)
		elif filetype == 'fits':
			normData = readspec(vegaFile)
		else:
			print 'Cannot recognize Normalization spectrum'
			return
	elif len(normFile) == 2:
		normData = normFile
	else:
		print 'Spectrum cannot be processed. Try another one.'
	
	while float(normData[1][-1]) > float(filterFile[1][-1]):
		normData[1].pop()
		normData[0].pop()
	
	while float(normData[1][0]) < float(filterFile[1][0]):
		normData[1].pop(0)
		normData[0].pop(0)
		
	# Filter is now interpolated with linear functions and convolved 
	# with the normalization spectrum
	filterSpline = scipy.interpolate.interpolate.interp1d(filterFile[1], \
	                                                      filterFile[0],1)
	filteredSpec = normData[0]*filterSpline(normData[1])
	
	data = [[],[]]
	data[1] = normData[1]
	data[0] = filteredSpec
	
	return data

def getnormconst(normSpectrum=None):
# (by Damian)
# Finds the normalization constant used in calculating magnitudes.
	import astrotools as at
	if normSpectrum == None:
		normFile = normalizationPath
	else:
		normFile = normSpectrum
		
	normData = at.readfilt(normFile,True)
	
	Fbol = 2983 * 10**8
	
	normConst = 2.5 * numpy.log10(integrate(normData)/Fbol)
	
	return normConst
	

def createnames(path):
# (by Damian)
# Creates a list of all the names, right ascensions, and declinations of 
# the spectra objects.
	
	import astrotools as at
	
	files = os.listdir(path)
	
	for name in files:
		if name.startswith('.'):
			files.pop(files.index(name))
			
	length = len(files)
	data = []
	n = 0
	
	while n < length:
		hdr = pyfits.getheader(path + files[n])
		data.append([files[n].split('.')[0]])
		try:
			data[n].append(hdr['RA'])
			if isinstance(data[n][-1],float):
				data[n][-1] = at.degtotime(data[n][-1])
		except KeyError:
			data[n].append('None')
		try:
			data[n].append(hdr['DEC'])
			if isinstance(data[n][-1],float):
				data[n][-1] = at.degtotime(data[n][-1])
		except KeyError:
			data[n].append('None')
		n = n + 1
	
	return data


#++++++++++++++++++++++++++++++++ CLASSES +++++++++++++++++++++++++++++++++++++

class photo_calc:
    # '''
    # *by Damian*
    # 
    # Stores the information from all of the spectral and filter files.
    # 
    # ATTRIBUTES
    #     .spectra[ object ][ 0-spectrum , 1-wavelength ][ value ]
    #     .filters[ filter ][ 0-transmission , 1-wavelength ][ value ]
    #     .specNames[]
    #     .normConst[]
    # 
    #     PARAMETERS
    #         spectrallPath: path of folder leading to all the spectral files.
    #     filterPath:    path of folder leading to all the filter files.
    #     normPath:      path of folder leading to the normalization file.
    # '''
		
	def __init__(self, spectralPath = None, filterPath = None, normPath = None):
		
		import astrotools as at
		
		if spectralPath == None:
			self.spectralPath = spectralDirectory
		else:
			 self.spectralPath = spectralPath
		
		if filterPath == None:
			self.filterPath = filterDirectory
		else:
			self.filterPath = filterPath
		
		if normPath == None:
			self.normPath = normalizationPath
		else:
			self.normPath = normPath	
		
		self.spectra = at.createspectra( self.spectralPath )
		self.filters = at.createfilters( self.filterPath )
		self.specNames = at.createnames( self.spectralPath )
		self.normConst = at.getnormconst( self.normPath )
	
	def addspectrum(self,spectralFile):
#		'''adds a spectrum to the spectral data'''
		
		self.spectra.append(readspect(spectralFile))
		
		temp = []
		
		hdr = pyfits.getheader(spectralFile)
		temp.append([spectralFile.split('/')[-1].split('.')[0]])
		try:
			temp.append(hdr['RA'])
		except KeyError:
			temp.append('None')
		try:
			temp.append(hdr['DEC'])
		except KeyError:
			temp.append('None')
		
		self.specNames.append(temp)
	
	def addfilter(self,filterPath):
#		'''Adds a filter to the filter data'''
		import astrotools as at
		
		self.filters.append(at.readfilt(filterFile))
	
	def delspectrum(self,specIndex):
#		'''Deletes a spectrum from the spectral data'''
		
		self.spectra.pop(specIndex)
		self.specNames.pop(specIndex)
	
	def delfilter(self,filtIndex):
#		'''Deletes a filter from the filter data'''
		
		self.filters.pop(filtIndex)
	
	def filterconvolve(self,specIndex,filtIndex,):
#		'''convolves a filter with a spectrum'''
		
		try:
			# Finds the main filter range, and then the corresponding indices
			# and wavelengths are saved
			indexRange = numpy.where( self.filters[filtIndex][0] > \
			             numpy.array(self.filters[filtIndex][0]).max()/100)
			indexRange = indexRange[0]
			minIndex = indexRange.min()
			rangeMin = self.filters[filtIndex][1][minIndex]
			maxIndex = indexRange.max()
			rangeMax = self.filters[filtIndex][1][maxIndex]
			
			# Check if the filter is inside the spectrum range
			if self.spectra[specIndex][1][0] > rangeMin or \
			            self.spectra[specIndex][1][-1] < rangeMax:
				print 'Filter range outside of spectrum data.'
				return
			
			# The overlapping range of data points in the spectrum is now found
			minIndexS = 0
			
			while self.spectra[specIndex][1][minIndexS] < rangeMin:
				minIndexS = minIndexS + 1
			
			maxIndexS = len(self.spectra[specIndex][1]) - 1
			
			while self.spectra[specIndex][1][maxIndexS] > rangeMax:
				maxIndexS = maxIndexS - 1
			
			# Filter is now interpolated with linear functions and convolved
			# with the spectrum
			filterVals = self.filters[filtIndex][0][minIndex:maxIndex + 1]
			filterSpline = scipy.interpolate.interpolate.interp1d( \
			                 self.filters[filtIndex][1][minIndex:maxIndex \
			                                            + 1],filterVals,1)
			filteredSpec = self.spectra[specIndex][0][minIndexS:maxIndexS] \
			               * filterSpline(self.spectra[specIndex][1] \
			                              [minIndexS:maxIndexS])
			
			filtData = [[],[]]
			filtData[0] = filteredSpec
			filtData[1] = self.spectra[specIndex][1][minIndexS:maxIndexS]
			
			return filtData 
			
		except IndexError:
			print'Object data does not exist for that index.'
			return
	
	def getmag(self,specIndex,filtIndex):
#		'''Gets filter magnitude of a spectrum, using normalization spectrum'''
		import astrotools as at
		
		try:
			normalization = integrate(at.filtNormSpec(self.filters[filtIndex]))
			filtFlux = integrate(self.filterconvolve(specIndex,filtIndex))
		
			# Constant is needed only if spectrum used to normalize has been 
			# flux calibrated
			magnitude = -2.5 * numpy.log10(filtFlux/normalization) \
			            + self.normConst
			return magnitude
			
		except TypeError:
			print 'Cannot calculate magnitude.'
			return 'None'
	
	


class list_photo_calc:
    # '''
    # (by Damian)
    #     
    #     PARAMETERS
    # spectralFile: tab delimited text file containing the names of the objects
    #               to be analyzed. The file must have the following columns
    #               (in this order): Filename  Ref  DesignationText  SpType.
    #               Filename is the name of the fits file
    #               Ref is a reference name
    #               SpType is the spectral type
    # filterFile:   tab delimited text file containing all the filters
    #               The file must have the following columns(in this order): 
    #               Filename  Name
    # 
    # ATTRIBUTES
    # specHeaders: 1-d array comtaining all the spectral headers
    # spectra:     1-d array containing all the spectral arrays
    # specFiles:   1-d array containing all the spectral files
    # specRef:     1-d array containing all the spectral reference numbers
    # specPos:     1-d array containing all the RA:Dec information
    # specTypes:   1-d array containing all the spectral classifications
    # filters:     1-d array containing all the filter arrays
    # filtNames:   1-d array containing all the filter names
    # mags:        2-d array containing the band magnitudes for each spectrum
    # '''
	
	def filterconvolve(self,specIndex,filtIndex,):
		
		try:
			# Find the main filter range, and then the corresponding 
			# indices and wavelengths are saved
			indexRange = numpy.where( self.filters[filtIndex][0] > \
			             numpy.array(self.filters[filtIndex][0]).max()/100)
			indexRange = indexRange[0]
			minIndex = indexRange.min()
			rangeMin = self.filters[filtIndex][1][minIndex]
			maxIndex = indexRange.max()
			rangeMax = self.filters[filtIndex][1][maxIndex]
			
			# Check if the filter is inside the spectrum range
			if self.spectra[specIndex][1][0] > rangeMin or \
			            self.spectra[specIndex][1][-1] < rangeMax:
				print self.filtNames[filtIndex] + ' outside of ' + \
				      self.specFiles[specIndex] + ' spectrum data'
				return
			
			# Overlapping range of data points in the spectrum is now found
			minIndexS = 0
			
			while self.spectra[specIndex][1][minIndexS] < rangeMin:
				minIndexS = minIndexS + 1
			
			maxIndexS = len(self.spectra[specIndex][1]) - 1
			
			while self.spectra[specIndex][1][maxIndexS] > rangeMax:
				maxIndexS = maxIndexS - 1
			
			# Filter is now interpolated with linear functions and convolved
			# with the spectrum
			filterVals = self.filters[filtIndex][0][minIndex:maxIndex + 1]
			filterSpline = scipy.interpolate.interpolate.interp1d( \
			               self.filters[filtIndex][1][minIndex:maxIndex + 1], \
			               filterVals,1)
			filteredSpec = self.spectra[specIndex][0][minIndexS:maxIndexS] \
			               * filterSpline(self.spectra[specIndex][1] \
			                              [minIndexS:maxIndexS])
			
			filtData = [[],[]]
			filtData[0] = filteredSpec
			filtData[1] = self.spectra[specIndex][1][minIndexS:maxIndexS]
			
			return filtData 
			
		except IndexError:
			print 'Object data does not exist for that index.'
			return
	
	def getmag(self,specIndex,filtIndex):
		import astrotools as at
		
		try:
			normalization = integrate(at.filtNormSpec(self.filters[filtIndex]))
			filtFlux = integrate(self.filterconvolve(specIndex,filtIndex))
		
			# Constant is needed only if spectrum used to normalize has been
			# flux calibrated
			magnitude = -2.5 * numpy.log10(filtFlux/normalization) + \
			            at.getnormconst()
			return magnitude
			
		except TypeError:
			print 'Cannot calculate magnitude'
			return 'None'
	
	def __init__(self, spectraFile, filterFile = None):
        
		import astrotools as at
        
		godFile = open(spectraFile).readlines()
		filters = open(filterFile).readlines()
		    
		specheaders = godFile[0].split('\t')
		filtheaders = filters[0].split('\t')
		specheads = len(specheaders)
		filtheads = len(filtheaders)
		spectraNum = len(godFile)
		filtersNum = len(filters)
		
		specFileIndex = specheaders.index('Filename')
		filtFileIndex = filtheaders.index('Filename')
		
		specTypeIndex = specheaders.index('SpType')
		filtNameIndex = filtheaders.index('Name')
		
		refIndex = specheaders.index('Ref')
		posIndex = specheaders.index('DesignationText')
		
		self.specHeaders = specheaders
		self.spectra = []
		self.specFiles = []
		self.filters = []
		self.filtNames = []
		self.specTypes = []
		self.mags = []
		self.specRef = []
		self.specPos = []
		
		for n in range (1,spectraNum):
			
			temp = godFile[n].split('\t')[specFileIndex]
			tempFile = readspec(spectralDirectory + temp)
			temp2 = godFile[n].split('\t')[specTypeIndex].split('\n')[0]
			temp3 = godFile[n].split('\t')[refIndex]
			temp4 = godFile[n].split('\t')[posIndex]
			
			if tempFile != [None]:
				self.spectra.append(tempFile)
				self.specFiles.append(temp)
				self.specTypes.append(temp2)
				self.specRef.append(temp3.split('.')[0])
				self.specPos.append(temp4)
		
		spectraNum = len(self.spectra)
		
		for n in range (1,filtersNum):
			
			temp = filters[n].split('\t')[filtFileIndex]
			tempFile = at.readfilt(filterDirectory + temp)
			temp2 = filters[n].split('\t')[filtNameIndex]
			
			if tempFile != [None]:
				self.filters.append(tempFile)
				self.filtNames.append(temp2)
		
		filtersNum = len(self.filters)
		
		for n in range (0,filtersNum):
			templist = []
			for m in range (0,spectraNum):
				temp = self.getmag(m,n)
				if temp != 'None':
					templist.append('%.2f'%temp)
				else:
					templist.append(numpy.nan)
			self.mags.append(templist)
	
	def printtofile(self,filename='test.txt'):
		
		tempFile = file(creationDirectory + filename,'r+')
		
		title = 'Ref' + '\t' + 'FileName'+ '\t' + 'RA:Dec'+ '\t' + \
		        'SpecClass' + '\t'
		
		for n in range (0,len(self.filtNames)):
			title = title + self.filtNames[n] + '\t'
			
		title = title + '\n'
		tempFile.write(title)
		
		for n in range(0,len(self.spectra)):
			line = self.specRef[n] + '\t' + self.specFiles[n] + '\t' + \
			       self.specPos[n] + '\t' + self.specTypes[n] + '\t'
			for m in range (0,len(self.filtNames)):
				line = line + str(self.mags[m][n]) + '\t'
			line = line + '\n'
			tempFile.write(line)
		
		print 'Magnitude list saved as ' + filename
	


class mag_analysis:
    # '''
    # ATTRIBUTES
    # .bandNames
    # .refNum 
    # .specType
    # .bandMags
    # 
    # METHODS
    # .gettype
    # .magavg 
    # .magdev
    # .getmags
    # .getcolor
    # .getcolors
    # .seespecmags
    # .seecolorcolor
    # '''
		
	def __init__(self):
		
		temp = open(magsfile).readlines()
		
		objects = len(temp) - 1
		bands = len(temp[0].split('\t')) - 2
		
		self.bandNames = []
		
		for n in range (2,2 + bands):
			bandName = temp[0].split('\t')[n]
			self.bandNames.append(bandName)
		
		self.bandNames[-1] = self.bandNames[-1].split('\n')[0]
		
		self.refNum = []
		
		for n in range (1,1 + objects):
			self.refNum.append(temp[n].split('\t')[0])
		
		self.specType = []
		
		for n in range (1,1 + objects):
			self.specType.append(temp[n].split('\t')[1])
		
		self.bandMags = []
		
		for m in range (1,1 + objects):
			
			temp2 = []
			
			for n in range (2,2 + bands):
				bandMag = temp[m].split('\t')[n]
				
				if bandMag != '':
					temp2.append(bandMag)
				else:
					temp2.append('NoMag')
			
			temp2[-1] = temp2[-1].split('\n')[0]
			
			self.bandMags.append(temp2)
	
	def gettype(self,specType):
        # '''Creates an array with all the band magnitudes of the objects 
        # in the given spectral class
        # data[1:Magnitude 2:Reference# 3:BandMagnitudes]'''
		
		specType = int(float(specType))
		
		length = len(self.bandMags)
		
		temp = []
		
		for n in range (0,length):
			if self.specType[n] != '':
				if int(float(self.specType[n])) == specType:
					temp2 = []
					temp2.append(self.specType[n])
					temp2.append(self.refNum[n])
					temp2.append(self.bandMags[n])
					temp.append(temp2)
		
		if len(temp) == 0:
			print 'No spectra in the magnitude range ' + str(specType) + \
			      '-' + str('%.2f'%(specType + .99))
			return temp
		else:
			return temp
	
	def magavg(self,specType, bandNum):
        # '''Calculates the average magnitude in the given band for the 
        # given spectral class'''
		
		temp = self.gettype(specType)
		
		mags = []
		
		if len(temp) != 0:
			for n in range (0,len(temp)):
				tmp = temp[n][2][bandNum]
				if tmp != 'NoMag':
					mags.append('%.2f'%tmp)
			
			return numpy.average(mags)
		else:
			return numpy.nan
	
	def magdev(self,specType, bandNum):
        # '''Calculates the standard deviation of the magnitudes for a given 
        # band in the given spectral class'''
		
		temp = self.gettype(specType)
		
		mags = []
		
		if len(temp) != 0:
			for n in range (0,len(temp)):
				tmp = temp[n][2][bandNum]
				if tmp != 'NoMag':
					mags.append('%.2f'%tmp)
			
			return numpy.std(mags)
		else:
			return 1
	
	def getmags(self, specTypei, specTypef, bandNum):
        # '''Returns an array with magnitude data for each spectral type between
        # the given two for the given band: 
        # data[0:SpectralClass 1:AverageMagnitudes 2:MagStandardDeviations]''' 
		
		start = int(specTypei)
		end = int(specTypef)
		
		tempmag = []
		tempavg = []
		tempstd = []
		
		for n in range (start, end + 1):
			tempmag.append(n)
			tempavg.append(self.magavg(n,bandNum))
			tempstd.append(self.magdev(n,bandNum))
		
		return [tempmag,tempavg,tempstd]
	
	def getcolor(self,specType,bandNum1,bandNum2):
        # '''Returns the color indices between the two given bands for each 
        # object in the spectral class'''
		
		temp = self.gettype(specType)
		
		colorName = self.bandNames[bandNum1] + '-' + self.bandNames[bandNum2]
		
		colors = [colorName,[]]
		
		if len(temp) != 0:
			for n in range (0, len(temp)):
				tmp1 = temp[n][2][bandNum1]
				tmp2 = temp[n][2][bandNum2]
				if tmp1 != 'NoMag' and tmp2 != 'NoMag':
					color = float(tmp1) - float(tmp2)
					colors[1].append('%.2f'%color)
				else:
					colors[1].append(numpy.nan)
			return colors
		else:
			return [colorName,numpy.nan]
			
	
	def colavg(self, specType, bandNum1, bandNum2):
        # '''Calculates the average color index between two bands in the 
        # given spectral class'''
		
		temp = self.getcolor(specType, bandNum1, bandNum2)
		
		colors = []
		
		if str(temp[1]) != 'nan':
			for n in range (0, len(temp[1])):
				tmp = str(temp[1][n])
				if str(tmp) != 'nan':
					colors.append(float(tmp))
			if len(colors) != 0:
				return [temp[0],'%.2f'%numpy.average(colors)]
			else:
				return [temp[0],'nan']
		else:
			return [temp[0],'nan']
	
	def coldev(self, specType, bandNum1, bandNum2):
        # '''Calculates the standard deviation of the color indices between
        # two bands for the given spectral class'''
		
		temp = self.getcolor(specType, bandNum1, bandNum2)
		
		colors = []
		
		if str(temp[1]) != 'nan':
			for n in range (0, len(temp[1])):
				tmp = str(temp[1][n])
				if tmp != 'nan':
					colors.append(float(tmp))
			if len(colors) > 1:
				return [temp[0],'%.2f'%numpy.std(colors)]
			else:
				return [temp[0],'%.2f'%.25]
		else:
			return [temp[0],numpy.nan]
	
	def getcolors(self, specTypei, specTypef, bandNum1, bandNum2):
        # '''Returns an array:
        #   data[0:ColorIndexName 1:SpectralClass 2:ColorAverages 
        #        3:ColorStandardDeviations]'''
			
		start = int(specTypei)
		end = int(specTypef)
		
		colorName = self.bandNames[bandNum1] + ' - ' + self.bandNames[bandNum2]
		
		colors = [colorName]
		tmp0,tmp1,tmp2 = [],[],[]
		
		for n in range (start, end + 1):
			tmp0.append(n)
			tmp1.append(float(self.colavg(n,bandNum1,bandNum2)[1]))
			tmp2.append(float(self.coldev(n,bandNum1,bandNum2)[1]))
		
		colors.append(tmp0)
		colors.append(tmp1)
		colors.append(tmp2)
			
		return colors
	
	def seespecmags(self, columns, rows, filters=None, startClass=0, endClass=18, scatter=True, pretty=False):
		
		if filters == None:
			filters = list(numpy.arange(len(self.bandNames)))
		
		if not(isinstance(filters,list)):
			print '"filters" must be a list type object '
			print 'Try that one again, genius...'
			return
		
		# Determine number of filter combinations that will be graphed
		mag1,mag2 = [],[]
		numberBands = len(filters)
		for n in range (0,numberBands):
			for m in range (n+1,numberBands):
				mag1.append(int(filters[n]))
				mag2.append(int(filters[m]))
		print mag1,mag2
		
		print 'There are ' + str(len(mag1)) + ' combinations of colors, and ' \
				+ str(rows*columns) + ' will be displayed.'
		
		# Set the colors that will be used for the graph
		if pretty == True:
			colors = 	['#CC0099','#993399','#9933CC','#6600FF','#3300CC',\
					 	'#0066FF','#0099CC','#00CCCC','#00CC99','#009900',\
					 	'#99CC33','#99FF00','#CCFF00','#CCCC00','#FFCC00',
					 	'#FF9900','#FF6633','#FF3333','#CC0033','#CC0000']
		else:
			colors = ['#000000']
		
		plotsNum = rows * columns
		
		for n in range (1, plotsNum + 1):
			
			data = self.getcolors(startClass,endClass,mag1[n-1],mag2[n-1])
			
			# Axisbg command sets the default background of the plots as 
			# white, or black if 'pretty' is True
			if pretty == True:
				pylab.subplot(rows,columns,n,axisbg='#000000')
			else:
				pylab.subplot(rows,columns,n)
			
			# Creates a title on the top of the page
			pylab.suptitle("Spectral Class vs. Color Indices", size=13, \
			               weight='bold' )
			
			# Creates the scatter plot for each spectral class
			if scatter == True:
				for m in range (startClass,endClass + 1):
					k = m - startClass
					if k > len(colors) - 1:
						k = len(colors) - 1
					scatterdata = self.getcolor(m,mag1[n-1],mag2[n-1])
					if str(scatterdata[1]) != 'nan':
						length = len(scatterdata[1])
						for j in range (0,length):
							pylab.plot(m,scatterdata[1][j],',',color=colors[k])
			
			# Plots the averages and the error bars on them	
			for m in range (0,endClass-startClass):
				k = m
				if k > len(colors) - 1:
					k = len(colors) - 1
				pylab.errorbar(data[1][m],data[2][m],data[3][m],fmt='or', \
				               color=colors[k], ecolor=colors[k], \
				               capsize=5)
			
			# Creates labels on all the y-axis, and only on the lower x axes
			pylab.ylabel(data[0], size=10)
			if n > plotsNum - columns:
				pylab.xlabel('Spectral Class', size = 10)
			
	
	def seecolorcolor(self, xBand1, xBand2, yBand1, yBand2, startClass=0, endClass=19, mean=True, pretty=False):
			
		# Set the default color as black. sets other colors if pretty is True
		if pretty == True:
			colors = 	['#CC0099','#993399','#9933CC','#6600FF','#3300CC',\
						 '#0066FF','#0099CC','#00CCCC','#00CC99','#009900',\
						 '#99CC33','#99FF00','#CCFF00','#CCCC00','#FFCC00',\
						 '#FF9900','#FF6633','#FF3333','#CC0033','#CC0000']
		else:
			colors = ['#000000']
			
		# Set the background color of the plot, black if pretty is True
		if pretty == True:
			pylab.subplot(111, axisbg='#000000')
		else:
			pylab.subplot(111)
		
		# Create a scatterplot of all the color vs color information
		for n in range (startClass, endClass + 1):
			
			m = n - startClass
			if m > len(colors) - 1:
				m = len(colors) - 1
			temp = self.getcolor(n, xBand1, xBand2)
			x = temp[1]
			if x != 'nan':
				xname = temp[0]
				temp = self.getcolor(n, yBand1, yBand2)
				y = temp[1]
				yname = temp[0]
				pylab.plot(x,y,',',color=colors[m])
				pylab.xlabel(xname)
				pylab.ylabel(yname)	
		
		# Graph the mean and stdev if mean is True	
		if mean == True:
			for n in range (startClass, endClass+1):
				
				m = n - startClass
				if m > len(colors) - 1:
					m = len(colors) - 1
				xavg = float(self.colavg(n, xBand1, xBand2)[1])
				if str(xavg) != 'nan':
					yavg = float(self.colavg(n, yBand1, yBand2)[1])
					xdev = float(self.coldev(n, xBand1, xBand2)[1])
					ydev = float(self.coldev(n, yBand1, yBand2)[1])
					pylab.errorbar(xavg, yavg, ydev, xdev, fmt='o', \
					              color=colors[m], ecolor=colors[m], capsize=5)
		
		# Create a title
		pylab.title('Color vs Color',size=13, weight='bold')
	
