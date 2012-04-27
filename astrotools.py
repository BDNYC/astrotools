'''
The module **astrotools** is a set of functions for astrophysical analysis developed by Kelle Cruz's team at Hunter College and the American Museum of Natural History in New York City. It consists of an amalgamation of functions tailored primordialy to handle fits file spectral data.

:Authors:
	Dan Feldman, Alejandro N |uacute| |ntilde| ez, Damian Sowinski

:Version:
    1.0 of 2012/03/16

:Repository:
    https://github.com/BDNYC/astrotools (for access, contact jfilippazzo@gmail.com)

'''

#+++++++++++++++++++++++++++++ RELEVANT MODULES +++++++++++++++++++++++++++++++
# Modules used by functions and classes
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

# +++++++++++++++++++++++++++ GLOBAL FUNCTIONS ++++++++++++++++++++++++++++++++
# Functions meant to be used by end users of astrotools
def avg_flux(startW, endW, SpecData, sum=True, median=False, verbose=True):
    '''
    (by Damian & Dan)
    
    Calculate the average flux of ... something.
    
    PARAMETERS...
    
    
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
            print "avg_flux: wavelength interval out of range"
        return
    # See that wavelength range does not fall between data points in 
    # wavelength array
    set1 = set(list(numpy.where( Wavelength_big >= startW )[0]))
    newEnd = endW + .0022
    set2 = set(list(numpy.where(Wavelength_big <= endW + .0022)[0]))
    temp = numpy.array(list(set1.intersection(set2)))
    if len(temp) == 0:
        if verbose  == True:
            print "avg_flux: there is no data in the selected interval"
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
            print "avg_flux: WARNING: difference between average and median ' \
                  & 'is greater than 3%"
            print "avg_flux: median = " + str(avgflux) + ",\t"+ " average = " \
                  + str(old)
            print "avg_flux: difference % = " + \
                  str(100 * numpy.abs( avgflux - old ) / old)
        else:
            if verbose == True:
                print "median worked"
    
    return avgflux


def integrate(xyData):
    '''
    (by Damian)
    
    Integrate x and y data treating it as a scatter plot with the trapezoid rule. the output is a float number.
    
    *xyData*
        2-D array with x-data in position 0 and y-data in position 1, can be a Python list or a numpy array.
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
    (by Alejandro N |uacute| |ntilde| ez)
    
    Combine spectra using a weighted mean. *Uncertainties are required* for this function. The mask wavelength array will be that of the first spectrum in the *spectra* list. The output is a python list with mask wavelength in position 0, combined flux in position 1, and combined uncertainties in position 2.
    
    This function mimics mc_meancomb (by Mike Cushing), with some restrictions.
    
    *spectra*
        Python list of spectra, where each spectrum is a python list as well, having wavelength in position 0, flux in position 1 and uncertainties in position 2. *Important*: flux array cannot have nan values.
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


def norm_spec(specData, limits, objID='NA'):
    '''
    (by Alejandro N |uacute| |ntilde| ez)
    
    Normalize a spectrum using a band (i.e. a portion) of the spectrum specified by *limits*.
    
    *specData*
        Spectrum as a Python list with wavelength in position 0, flux in position 1, and (optional) error values in position 2. More than one spectrum can be provided simultaneously, in which case *specData* shall be a list of lists.
    *limits*
        Python list with lower limit in position 0 and upper limit in position 1.
    *objID*
        String to be used as an identifier for spectrum; if dealing with several spectra, *objID* shall be a list of strings. For error printing purposes only.
    '''
    
    # Convert specData to list or spectra if it consists only of one
    if len(specData) <= 3 and len(specData[0]) > 10:
        specData = [specData]
    
    # Initialize objects
    finalData = [None] * len(specData)
    
    # Check that given limits are reasonable
    if limits[0] >= limits[1]:
        print 'norm_spec: the Min and Max values specified are not reasonable.'
        return None
    
    # Re-define normalizing band (specified in limits) in the case when the
    # the limits fall outside of the spectrum limits itself
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
            print 'norm_spec: the wavelength data for object %s is outside ' \
                  + 'the given limits.' %objID[spIdx]
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
            print 'norm_spec: the wavelength data for object %s is outside ' \
                  + 'the given limits.' %objID[spIdx]
            continue
        else:
            maxIdx = largeIdx[0][0]
        
        # 5) Check for consistency in the computed band limits
        if maxIdx - minIdx < 2:
            print 'norm_spec: The Min and Max values specified for object %s ' \
                  + 'yield no band.' %objID[spIdx]
            continue
            
        # 6) Select flux band from spectrum
        fluxSelect = spData[1][minIdx:maxIdx]
        
        # 7) Select error value band from spectrum
        if errors is True:
            errorSelect = spData[2][minIdx:maxIdx]
        
        # 8) Normalize spectrum using arithmetic mean
        notNans = numpy.where(numpy.isfinite(fluxSelect))
        avgFlux = numpy.mean(fluxSelect[notNans])
        finalFlux = spData[1] / avgFlux
        
        finalData[spIdx] = [spData[0], finalFlux]
        
        if errors is True:
            notNans  = numpy.where(numpy.isfinite(errorSelect))
            avgError = numpy.mean(errorSelect[notNans])
            finalErrors = spData[2] / avgError
            
            finalData[spIdx] = [spData[0], finalFlux, finalErrors]
    
    return finalData


def plot_spec(specData, ploterrors=False):
    '''
    (by Alejandro N |uacute| |ntilde| ez)
    
    Plot a spectrum. If more than one spectrum is provided simultaneously, it will plot all spectra on top of one another.
    
    This is a quick and dirty tool to visualize a set of spectra. It is not meant to be a paper-ready format. You can use it, however, as a starting point.
    
    *specData*
        Spectrum as a Python list with wavelength in position 0, flux in position 1, and (optional) error values in position 2. More than one spectrum can be provided simultaneously, in which case *specData* shall be a list of lists.
    
    *ploterrors*
        Boolean: Include flux error bars when available. This will work only if all spectra have error values.
    '''
    
    # Check that there is data to plot
    allNone = True
    for spData in specData:
        if spData is not None:
            allNone = False
            break
    if allNone:
        return
    
    # Initialize figure
    fig = plt.figure(1)
    fig.clf()
    
    # Set plot titles
    TITLE   = 'SPECTRAL DATA'
    X_LABEL = 'Wavelength'
    Y_LABEL = 'Flux'
    
    # Initialize plot within figure
    subPlot = fig.add_subplot(1,1,1)
    subPlot.set_title(TITLE)
    subPlot.set_xlabel(X_LABEL)
    subPlot.set_ylabel(Y_LABEL)
    
    # Check if all spectra have error values
    errorsOK = True
    for spData in specData:
        if len(spData) != 3:
            errorsOK = False
    
    # Plot spectra
    for spData in specData:
        if spData is not None:
            if errorsOK and ploterrors:
                subPlot.errorbar(spData[0], spData[1], spData[2], \
                          capsize=2)
            else:
                subPlot.plot(spData[0], spData[1], drawstyle='steps-mid')
    
    return fig


def read_spec(specFiles, aToMicron=False, negToZero=False, normal=False, errors=False, plot=False, warn=False):
    '''
    (by Alejandro N |uacute| |ntilde| ez)
    
    Read spectral data from fits files. It returns a python list with wavelength in position 0, flux in position 1 and error values (if available and requested) in position 2. More than one fits file name can be provided simultaneously.
    
    *specFiles*
        String with fits file name (with full path); it can also be a python list of file names.
    *aToMicron*
        Boolean: If wavelength in fits file is in Angstrom, convert wavelength values into micrometers.
    *negToZero*
        Boolean: Set any negative flux values equal to zero.
    *normal*
        Boolean: Normalize the flux values using the simple average of all flux data points.
    *errors*
        Boolean: Return error values for the flux data; return nans if unavailable.
    *plot*
        Boolean: Plot the spectral data, including error bars when available
    *warn*
        Boolean: Show warning messages
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
        specData[spFileIdx] = __get_spec(fitsData, fitsHeader, spFile, errors)
        
        if specData[spFileIdx] is None:
            continue
        
        # 3.4. Generate wl axis when needed
        if specData[spFileIdx][0] is None:
            specData[spFileIdx][0] = __create_waxis(fitsHeader, \
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
            specData[spFileIdx] = __normalize(specData[spFileIdx])
    
    # 4. Plot the spectra if desired
    if plot:
        plot_spec(specData, ploterrors=True)
    
    # 5. Clear up memory
    fitsHeader = ''
    fitsData   = ''
    
    return specData


def sel_band(specData, limits, objID='NA'):
    '''
    (by Alejandro N |uacute| |ntilde| ez)
    
    Select a band (i.e. a portion) from a spectrum specified by *limits*.
    
    *specData*
        Spectrum as a Python list with wavelength in position 0, flux in position 1, and (optional) error values in position 2. More than one spectrum can be provided simultaneously, in which case *specData* shall be a list of lists.
    *limits*
        Python list with lower limit in position 0 and upper limit in position 1.
    *objID*
        String with ID for spectrum; if dealing with several spectra, *objID* shall be a list of strings. For error printing purposes only.
    '''
    
    # Convert specData to list or spectra if it consists only of one
    if len(specData) <= 3 and len(specData[0]) > 10:
        specData = [specData]
    
    # Initialize objects
    finalData = [None] * len(specData)
    
    # Check that given limits are reasonable
    if limits[0] >= limits[1]:
        print 'sel_band: the Min and Max values specified are not reasonable.'
        return None
    
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
            print 'sel_band: the wavelength data for object %s is outside ' \
                  + 'the given limits.' %objID[spIdx]
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
            print 'sel_band: the wavelength data for object %s is outside  ' \
                  + 'the given limits.' %objID[spIdx]
            continue
        else:
            maxIdx = largeIdx[0][0]
        
        # 5) Check for consistency in the computed band limits
        if maxIdx - minIdx < 2:
            print 'sel_band: The Min and Max values specified for object %s ' \
                  + 'yield no band.' %objID[spIdx]
            continue
            
        # 6) Select flux band from spectrum
        fluxSelect = spData[1][minIdx:maxIdx]
        
        # 7) Select error value band from spectrum
        if errors is True:
            errorSelect = spData[2][minIdx:maxIdx]
            
            finalData[spIdx] = [spData[0][minIdx:maxIdx], fluxSelect, \
                                errorSelect]
        else:
            finalData[spIdx] = [spData[0][minIdx:maxIdx], fluxSelect]
    
    return finalData


def smooth_spec(specData, specFiles=None, goodRes=200, winWidth=10):
    '''
    (by Alejandro N |uacute| |ntilde| ez)
    
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


# +++++++++++++++++++++++++ SECONDARY FUNCTIONS +++++++++++++++++++++++++++++++
# Functions used by Global Functions; these are not meant to be used directly 
# by end users of astrotools
def __normalize(specData):
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
    


def __create_waxis(fitsHeader, lenData, fileName):
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
    


def __get_spec(fitsData, fitsHeader, fileName, errorVals):
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
    


# ++++++++++++++++++++++++++++++++ CLASSES ++++++++++++++++++++++++++++++++++++
# Add you class here
    