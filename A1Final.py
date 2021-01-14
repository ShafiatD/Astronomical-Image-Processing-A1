import numpy as np
import astropy.io.fits as fits
import astropy.table as tab
import astropy.io.ascii as asc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cycler import cycler
import copy as cp
import time as tm

# Setting maplotlib parameters
colour = plt.cm.viridis(np.linspace(0, 1, 8))
params = {'font.size' : 12,
          'font.family' : 'serif',
          'font.serif' : 'Times New Roman',
          'mathtext.fontset' : 'stix',
          'figure.dpi' : 300,
          'axes.grid' : False,
          'xtick.minor.visible' : True,
          'xtick.top' : True,
          'xtick.direction' : 'in',
          'ytick.minor.visible' : True,
          'ytick.right' : True,
          'ytick.direction' : 'in',
          'axes.prop_cycle' : cycler('color', colour[::1]),
          }

plt.rcParams.update(params)

def gaussian(x,mean,sig,a): # Defining a gaussian function
    return a * (1 / (sig*np.sqrt(2*np.pi))) * np.exp(-((x-mean)**2) / (2*sig**2))

def cutting(array):
    for i in range(len(array)):             # Iterate over ever element
        if i > 0 and array[i] > array[i-1]: # If current value is bigger than the previous value
            return i                        # Return the index of that point
    return len(array)                       

def brightness(arr, background, xmin = 0, xmax = -1, ymin = 0, ymax = -1):
    """
    Returns the brightness value of the image adjusted for background noise

    Parameters
    ----------
    arr : np.array
        Array of values of target image.
    background : int
        Value of background noise.
    xmin : int, optional
        The minimum x coordinate of the image. The default is 0.
    xmax : int, optional
        The maximum x coordinate of the image. The default is -1.
    ymin : int, optional
        The minimum y coordinate of the image. The default is 0.
    ymax : int, optional
        The maximum y coordinate of the image. The default is -1.

    Returns
    -------
    pixel_count : int
        The count of the number of pixels within the image.

    """
    if type(background) != int:
        raise TypeError('Background must be an integer')
    
    newdata = arr[ymin:ymax+1,xmin:xmax+1] - background
    
    for j in range(len(newdata)):   # making sure all pixels are > 0
        newdata[j] = [0 if i < 0 else i for i in newdata[j]]    # Subtracting Background
    
    pixel_count = np.sum(newdata)
    return pixel_count


def calculate_magnitude(pxc, pxc_e, hdr):
    """
    Calculates the calibrated flux of a source and outputs the apparent
    magnitude

    Parameters
    ----------
    pxc : int
        Value of brightness of the source using the total pixel count.
    pxc_e : float
        The error in the pixel count.
    hdr : astropy.io.fits.header.header
        Header of the fits file we are analysing.

    Returns
    -------
    m : float
        Apparent magnitude of a source.
    err : float
        Error in apparent magnitude.

    """
    MAGZPT = hdr['MAGZPT']           # Extracting the zero point calibration
    MAGZRR = hdr['MAGZRR']           # Extracting the error in the zero point
    m = MAGZPT - 2.5*np.log10(pxc)   # Calculating the magnitude
    err = (MAGZRR**2 + (-2.5/np.log(10))**2 * (pxc_e/pxc) ** 2) ** 0.5    # Calculating the error in the magnitude
    return m, err


def find(input_array, mu, sigma, n, hdr, splitting = False, filename = 'data.txt'):
    """
    Function that counts the number of sources within a .fits CCD image of the
    sky. Outputs the count of the number of sources in the image, the location
    of each source, the magnitude of each source, the error associated with the
    magnitude, and an esimation of the location of a star. An adaptive aperture
    is used to detect sources. The code also then saves the data as a table in
    the filename specified in the input.

    Parameters
    ----------
    input_array : list/array
        Input array extracted from the .fits file of the image.
    mu : float
        The mean background noise value of the image.
    sigma : float
        The error associated with the background noise.
    n : int
        The tolerance of the noise; how much mu + n*sigma the algorithm should
        cut-off the detection at.
    hdr : astropy.io.fits.header.header
        Header of the fits file we are analysing.
    splitting : bool, optional
        If True, then the algorithm will attempt to split images with multiple
        sources in them. The default is False.
    filename : str, optional
        Filename for the ascii data file. The default is 'data.txt'

    Returns
    -------
    count : int
        Number of sources detected within the image.
    locations : list
        Returns the location of each source within the image in the form:
        [xmin, xmax, ymin, ymax].
    magnitudes : list
        Returns the magnitude of each source (float). Calculated using mag()
    errors : list
        Returns the error on the magnitude of each source (float). Calculated
        using mag()
    starlocations : list
        Returns the location of each star within the image in the form:
        [xmin, xmax, ymin, ymax]. Stars are detected according to the hdr data
        with a tolerance of 0.8x the saturation limit of the CCD.

    """
    array = cp.deepcopy(input_array)# Create a copy of the original array
    complete = False                # Set a checking variable to exit the source searching loop
    
    # Setting initial variables
    count = 0                       # Setting integer container for source counts
    locations = []                  # Setting array conatiner for source edge locations
    xlocation = []                  # Setting array conatiner for source center x locations
    ylocation = []                  # Setting array conatiner for source center y locations
    magnitudes = []                 # Setting array container for source magnitudes
    errors = []                     # Setting array container for source errors
    starlocations = []              # Setting array container for star locations

    # Source searching loop
    while not complete:             # Loop until image has been masked and all sources detected
        
        currentmax = np.amax(array)    # Find the current maximum value in the array
        if currentmax < mu + n*sigma:  # Break loop if the current maximum is less than the noise
            break
        
        peaks = np.where(array == np.amax(array))    # Locate maximum values in the image
        for i in range(len(peaks[0])):               # Loop across all maximum values found
            if array[peaks[0][i]][peaks[1][i]] == 0: # If current maximum has already been blocked out, skip
                continue

            xmax = peaks[1][i]          # Set bounding maximum and minimum x and y values
            xmin = peaks[1][i]
            ymax = peaks[0][i]
            ymin = peaks[0][i]          
            complete_area = False       # Set checking variable for identifying area of the source
            
            while not complete_area:    # Loop until full area of object found
                xmax_check = 0          # Set checking variables for each bound to 0
                xmin_check = 0
                ymax_check = 0
                ymin_check = 0          
                
                for y in range(ymin-1, ymax+2):          # For all x to the right of current 'found' portion of object
                    if array[y, xmax+1] > mu + 3*sigma:  # If a 'readable' pixel is found
                        xmax += 1                        # Increment the maximum x bound 
                        xmax_check = 1                   # Bound has been increased, set xmax_check
                        break
                    
                for y in range(ymin-1, ymax+2):          # For all x to the left of current 'found' portion of object
                    if array[y, xmin-1] > mu + 3*sigma:  # If a 'readable' pixel is found
                        xmin -= 1                        # Decrement the minimum x bound
                        xmin_check = 1                   # Bound has been decreased, set xmin_check
                        break
                    
                for x in range(xmin-1, xmax+2):          # For all y above the current 'found' portion of object
                    if array[ymax+1, x] > mu + 3*sigma:  # If a 'readable' pixel is found
                        ymax += 1                        # Increment the maximum y bound
                        ymax_check = 1                   # Bound has been increased, set ymax_check
                        break
                    
                for x in range(xmin-1, xmax+2):          # For all y below the current 'found' portion of object
                    if array[ymin-1, x] > mu + 3*sigma:  # If a 'readable' pixel is found
                        ymin -= 1                        # Decrement the minimum y bound
                        ymin_check = 1                   # Bound has been decresed, set ymin_check
                        break
                
                # If no new bounds were found, then bounds of source have been finalised
                if xmax_check == 0 and xmin_check == 0 and ymax_check == 0 and ymin_check == 0:
                    complete_area = True                 # Set complete_area check variable
                    
                    
            if xmin == xmax or ymin == ymax:
                for y in range(ymin,ymax+1):             # Block off region where the object is
                    for x in range(xmin,xmax+1):
                        array[y][x]=0                    # Set those pixels to 0        
                continue
            
            if splitting:
                xpeak = peaks[1][i]         # Saving x,y location of peak pixel            
                ypeak = peaks[0][i]
            
                averagesr = []  # Creating a container for the column averages to the right of the peak
                averagesu = []  # Creating a container for the row averages above the peak
                averagesl = []  # Creating a container for the column averages to the left of the peak
                averagesd = []  # Creating a container for the row averages below the peak

                
                for j in range(xpeak, xmax - 2): # For every column to the right of the centre,
                    # Average over two column average pixel values
                    averagesr.append(np.average(array[ymin:ymax,peaks[1][i]:xmax][:,j - peaks[1][i]: j - peaks[1][i] + 2]))

                
                for k in range(ymin, ypeak - 2): # For every row above the centre,
                    # Average over two row average pixel values
                    averagesu.append(np.average(array[ymin:ymax,xmin:xmax][k - ymin: k - ymin + 2,:]))
                
                averagesu = averagesu[::-1]      # Reversing array for analysis
                
                for l in range(xmin, xpeak - 2): # For every column to the left of the centre,
                    # Average over two column average pixel values
                    averagesl.append(np.average(array[ymin:ymax,xmin:xmax][:,l - xmin: l - xmin + 2]))
                    
                averagesl = averagesl[::-1]      # Reversing array for analysis
                
                for m in range(ypeak, ymax - 2): # For every row below the centre,
                    # Average over two row average pixel values
                    averagesd.append(np.average(array[peaks[0][i]:ymax,xmin:xmax][m - peaks[0][i]: m - peaks[0][i] + 2,:]))
                
                right = xpeak + cutting(averagesr)    # Calculating new coordinate bounds
                up = ypeak - cutting(averagesu)
                left = xpeak - cutting(averagesl)
                down = ypeak + cutting(averagesd)
                
                xmin, xmax, ymin, ymax = left, right, up, down  # Settings new bounds

            for y in range(ymin, ymax+1):           # Block off region where the object is
                for x in range(xmin, xmax+1):
                    array[y][x] = 0                 # Set those pixels to 0       
                         
            if (xmax-xmin+1)*(ymax-ymin+1) <= 9:    # Do not count as a source if the area
                continue                            # is less than 9 pixels            
            
            if currentmax >= hdr['SATURATE']:                   # If the source is saturated depening on the hdr
                starlocations.append([xmin, xmax, ymin, ymax])  # Count as a star, and add to star locations
                continue
            
            count += 1                                                  # Increase object count by 1
            locations.append([xmin, xmax, ymin, ymax])                  # Append the objects location to the array
            xlocation.append(int((xmax+xmin)/2))
            ylocation.append(int((ymax+ymin)/2))
            bright = brightness(array, mu, xmin, xmax, ymin, ymax)      # Calculate brightness of object
            mags, errs = calculate_magnitude(bright, int(np.sqrt(bright)), hdr) # Calculate the magnitudes and errors
            magnitudes.append(mags)     # Append the object's magnitude to the array
            errors.append(errs)         # Append the object's magnitude error to the array
            
    index=np.arange(1,count+1)                          # create an array of the indexes of each object
    data_table=tab.Table([index,xlocation,ylocation,magnitudes,errors],
                         names=('index','x','y','magnitudes','error')) # create a table of the data
    asc.write(data_table,filename,fast_writer=False)    # save the table as a txt file
    return count, locations, magnitudes, errors, starlocations


#%% Importing fits file into Python
with fits.open('mosaic.fits') as hdul: #open fits file
    data = hdul[0].data     # Saving pixel data
    hdr = hdul[0].header    # Saving header data

#%% Fitting the noise with a Gaussian profile
bins = 30000                                # Create a 30,000 bin histogram of the image 
ndata = data.flatten()
plt.figure()
plt.grid(alpha = 0.4)
binsdata, binsedges, patches = plt.hist(ndata, bins, color = 'C4',
                                        edgecolor = 'black', linewidth = 0.8) # Plotting the histogram

bins_x = binsedges[1:]                      # Matching shapes of bins_x and binsdata
bins_x = bins_x - (bins_x[3]-bins_x[2])/2   # Create an array of the centres of the bins

guess1=[3400,12,8000000]                    # Initial guess, with noise mean ~ 3400
pop, pcov = curve_fit(gaussian,bins_x, binsdata, p0 = guess1)  # Fitting gaussian to histogram of data

xrange = np.linspace(3319,3519,1000)        # Plotting the fitted noise profile
plt.plot(xrange,gaussian(xrange, *pop), color = 'black')
plt.xlim(3319,3519)
plt.ylim(0,max(binsdata))
plt.xlabel('Pixel Value')
plt.show()
plt.savefig('Noise_Fit', bbox_inches='tight')

#%% Premasking Data
testdata = cp.deepcopy(data)          # Create a copy of the original array

print('Premasking')
maxy = len(testdata)
maxx = len(testdata[0])

# Locations of areas to block off
to_block = [[0,118,0,maxy],           # Left edge blocking
            [2469,maxx,0,maxy],       # Right edge blocking
            [118,2469,0,120],         # Bottom edge blocking
            [118,2469,4513,maxy],     # Top edge blocking
            [2363,2470,4192,4514],    # Bottom right corner edge blocking
            [2160,2363,4418,4514],    #   ^
            [1131,1720,2923,3490],    # Central Star blocking
            [1424,1454,0,maxy],       # Central Star 'Spine' Blocking
            [1364,1501,114,469],      # Central Star CCD Saturation Blocking
            [1286,1527,120,128],      #   ^
            [1098,1654,423,438],      #   ^
            [1014,1705,310,330],      #   ^
            [895,915,2219,2356],      # Additional 'large' star blocking
            [960,988,2702,2835],      #   ^
            [763,790,3201,3418],      #   ^
            [2125,2145,3706,3801],    #   ^
            [1527, 1537, 120, 137],   # CCD artefact blocking
            [2468, 2468, 3391, 3437], #   ^
            [1351, 1363, 128, 129],   #   ^
            [1537, 1537, 123, 123],   #   ^
            [1533, 1534, 137, 137],   #   ^
            [1027, 1042, 424, 450],   #   ^
            [1641, 1647, 333, 353],   #   ^
            [1361, 1363, 128, 129],   #   ^
            [1537, 1537, 123, 123],   #   ^
            [1533, 1534, 137, 137],   #   ^
            [876, 935, 3356, 3414],   # More bright sources above hdr['SATURATE']
            [658, 696, 1920, 1956],   #   ^
            [427, 468, 2281, 2323],   #   ^
            [533, 590, 4071, 4125],   #   ^
            [1754, 1792, 560, 596],   #   ^
            [170, 208, 3903, 3941],   #   ^
            [685, 724, 2245, 2281],   #   ^
            [2060, 2124, 1375, 1454], #   ^
            [1454, 1482, 4004, 4055], #   ^
            [949, 985, 1637, 1670],   #   ^
            [1292, 1345, 4379, 4418], #   ^
            [2103, 2159, 2278, 2337], #   ^
            [1348, 1385, 4312, 4349], #   ^
            [2433, 2468, 3382, 3445], #   ^
          ]

# Showing the blocked regions
fig, ax = plt.subplots(8,5, dpi = 300)
x = 0
y = -1
for i in range(len(to_block)):
    x = (i)%5
    if x == 0:
        y += 1
    xmin, xmax, ymin, ymax = [*to_block[i]]
    ax[y, x].imshow(testdata[ymin:ymax, xmin:xmax])
    ax[y, x].tick_params(
        axis = 'both',       # Changes apply to the x-axis
        which = 'both',      # Both major and minor ticks are affected
        bottom = False,      # Ticks along the bottom edge are off
        top = False,         # Ticks along the top edge are off
        right = False,
        left = False,
        labelbottom = False, # Labels along the bottom edge are off
        labelleft = False)   # Labels along the bottom edge are off

plt.show()
plt.savefig('Premasking', bbox_inches='tight')

for i in range(len(to_block)):        # Set all the pixels in the locations to 0
    for x in range (to_block[i][0],to_block[i][1]):
        for y in range (to_block[i][2],to_block[i][3]):
            testdata[y][x]=0
            
print('Premasked')     

#%% Calculate count, locations, magnitudes, errors, and  for 3,4,5,6,8,10,12,20,30 sigmas
print('Detecting sources')

time1_3 = tm.time()
count_3, loc_3, mag_3, err_3, star_3 = find(testdata, int(pop[0]), int(pop[1]), 3, hdr, filename = '3sigma_data.txt')
time2_3 = tm.time()
print('3 sigma finished: %.2fs' % (time2_3 - time1_3))

time1_4 = tm.time()
count_4, loc_4, mag_4, err_4, star_4 = find(testdata, int(pop[0]), int(pop[1]), 4, hdr, filename = '4sigma_data.txt')
time2_4 = tm.time()
print('4 sigma finished: %.2fs' % (time2_4 - time1_4))

time1_5 = tm.time()
count_5, loc_5, mag_5, err_5, star_5 = find(testdata, int(pop[0]), int(pop[1]), 5, hdr, filename = '5sigma_data.txt')
time2_5 = tm.time()
print('5 sigma finished: %.2fs' % ((time2_5 - time1_5)))

time1_6 = tm.time()
count_6, loc_6, mag_6, err_6, star_6 = find(testdata, int(pop[0]), int(pop[1]), 6, hdr, filename = '6sigma_data.txt') 
time2_6 = tm.time()
print('6 sigma finished: %.2fs' % ((time2_6 - time1_6)))

time1_10 = tm.time()
count_10, loc_10, mag_10, err_10, star_10 = find(testdata, int(pop[0]), int(pop[1]), 10, hdr, filename = '10sigma_data.txt')
time2_10 = tm.time()
print('10 sigma finished: %.2fs' % ((time2_10 - time1_10)))

time1_20 = tm.time()
count_20, loc_20, mag_20, err_20, star_20 = find(testdata, int(pop[0]), int(pop[1]), 20, hdr, filename = '20sigma_data.txt')
time2_20 = tm.time()
print('20 sigma finished: %.2fs' % ((time2_20 - time1_20)))

time1_30 = tm.time()
count_30, loc_30, mag_30, err_30, star_30 = find(testdata, int(pop[0]), int(pop[1]), 30, hdr, filename = '30sigma_data.txt')
time2_30 = tm.time()
print('30 sigma finished: %.2fs' % ((time2_30 - time1_30)))
        
#%% # Calculating N for all magnitudes for 3, 4, 5, 6, 10, 20, 30 sigma

print('Calculating N')

def counting(magnitudes):
    N = [0]
    sortedC = [0]
    minC = min(magnitudes)
    complete = False
    i = 0
    cnew = cp.deepcopy(np.array(magnitudes))
    
    while not complete:
        index = np.where(magnitudes == minC)
        sortedC.append(minC)
        N.append(len(index) + N[i])
        
        for x in range(len(index)):
            cnew[index[x]] = 1000
            
        minC = min(cnew)        
        if minC == 1000:
            return N, sortedC
            
        i += 1    
    
N_3, sortedmag_3 = counting(mag_3)      # Find the N count for each sigma
N_4, sortedmag_4 = counting(mag_4)
N_5, sortedmag_5 = counting(mag_5)
N_6, sortedmag_6 = counting(mag_6)
N_10, sortedmag_10 = counting(mag_10)
N_20, sortedmag_20 = counting(mag_20)
N_30, sortedmag_30 = counting(mag_30)

#%% # Fitting data for 5 sigma

print('Fitting')
def f(x,a,b):       # Defining function to fit the parameters
    return a*x+b

tofitN = np.log10(N_5[1:100])    # Selecting straight line portion of Log(N) graph
tofitC = sortedmag_5[1:100]
fit5, fiterror5 = curve_fit(f, tofitC, tofitN)
x1 = np.arange(min(sortedmag_5[1:]), sortedmag_5[100],0.01)
y1 = f(x1,*fit5)

#%% Plot log(N) graph with fitted line
fig, ax = plt.subplots()
plt.plot(sortedmag_3, np.log10(N_3), label = r'3 $\sigma$')
plt.plot(sortedmag_4, np.log10(N_4), label = r'4 $\sigma$')
plt.plot(sortedmag_5, np.log10(N_5), label = r'5 $\sigma$')
plt.plot(sortedmag_6, np.log10(N_6), label = r'6 $\sigma$')
plt.plot(sortedmag_10, np.log10(N_10), label = r'10 $\sigma$')
plt.plot(sortedmag_20, np.log10(N_20), label = r'20 $\sigma$')
plt.plot(sortedmag_30, np.log10(N_30), label = r'30 $\sigma$')

ax.axvline(sortedmag_5[1],0,0.19, color = 'black', linestyle = ':')
ax.axvline(sortedmag_5[100],0,0.79, color = 'black', linestyle = ':')
plt.plot(x1,y1,label=r'm = %.3f $\pm$ %.3f' % (fit5[0],np.sqrt(fiterror5[0][0])),
          color = 'red')
plt.legend(fontsize = 'small')
plt.xlabel(r'Magnitude')
plt.ylabel(r'$\log_{10}N$')
plt.grid()
plt.show()
plt.savefig('LogNm', bbox_inches='tight')
print('Gradient obtained =',fit5[0],'+-',np.sqrt(fiterror5[0][0]))

#%% Show first 25 objects found
# Plotting
fig, ax = plt.subplots(5,5, dpi = 300)
x = 0
y = -1
for i in range(len(loc_5[:25])):
    x = (i)%5
    if x == 0:
        y += 1
    xmin, xmax, ymin, ymax = [*loc_5[i+1]] 
    ax[y, x].imshow(testdata[ymin:ymax, xmin:xmax], cmap = 'viridis')
    ax[y, x].tick_params(
        axis = 'both',       
        which = 'both',      
        bottom = False,      
        top = False,         
        right = False,
        left = False,
        labelbottom = False, 
        labelleft = False)   
plt.show()
plt.savefig('LogNm25', bbox_inches='tight')

#%% Calculate count, locations, magnitudes, errors, and  for 3,4,5,6,8,10,12,20,30 sigmas
print('Detecting sources')

stime1_3 = tm.time()
scount_3, sloc_3, smag_3, serr_3, sstar_3 = find(testdata, int(pop[0]), int(pop[1]), 3, hdr, True, filename = '3sigma_splitting_data.txt')
stime2_3 = tm.time()
print('3 sigma finished: %.2fs' % (stime2_3 - stime1_3))

stime1_4 = tm.time()
scount_4, sloc_4, smag_4, serr_4, sstar_4 = find(testdata, int(pop[0]), int(pop[1]), 4, hdr, True, filename = '4sigma_splitting_data.txt')
stime2_4 = tm.time()
print('4 sigma finished: %.2fs' % (stime2_4 - stime1_4))

stime1_5 = tm.time()
scount_5, sloc_5, smag_5, serr_5, sstar_5 = find(testdata, int(pop[0]), int(pop[1]), 5, hdr, True, filename = '5sigma_splitting_data.txt')
stime2_5 = tm.time()
print('5 sigma finished: %.2fs' % ((stime2_5 - stime1_5)))

stime1_6 = tm.time()
scount_6, sloc_6, smag_6, serr_6, sstar_6 = find(testdata, int(pop[0]), int(pop[1]), 6, hdr, True, filename = '6sigma_splitting_data.txt') 
stime2_6 = tm.time()
print('6 sigma finished: %.2fs' % ((stime2_6 - stime1_6)))

stime1_10 = tm.time()
scount_10, sloc_10, smag_10, serr_10, sstar_10 = find(testdata, int(pop[0]), int(pop[1]), 10, hdr, True, filename = '10sigma_splitting_data.txt')
stime2_10 = tm.time()
print('10 sigma finished: %.2fs' % ((stime2_10 - stime1_10)))

stime1_20 = tm.time()
scount_20, sloc_20, smag_20, serr_20, sstar_20 = find(testdata, int(pop[0]), int(pop[1]), 20, hdr, True, filename = '20sigma_splitting_data.txt')
stime2_20 = tm.time()
print('20 sigma finished: %.2fs' % ((stime2_20 - stime1_20)))

stime1_30 = tm.time()
scount_30, sloc_30, smag_30, serr_30, sstar_30 = find(testdata, int(pop[0]), int(pop[1]), 30, hdr, True, filename = '30sigma_splitting_data.txt')
stime2_30 = tm.time()
print('30 sigma finished: %.2fs' % ((stime2_30 - stime1_30)))

#%% Justifying Splitting
# Looking at image 54 with 5 sigma is a perfect example of many multiples of sources within one aperture
xmin_j, xmax_j, ymin_j, ymax_j = [*loc_5[53]]
#Plotting
fig, ax = plt.subplots(2,2, gridspec_kw = {'width_ratios' : [3,1],
                                           'height_ratios' : [2,1]}, sharex=True)
ax[0, 0].imshow(testdata[ymin_j:ymax_j, xmin_j:xmax_j], origin = 'lower')
ax[0, 0].set_ylabel('Nth row')
ax[0, 0].grid(alpha = 0.2)
ax[0, 0].tick_params(
        axis = 'both',       
        which = 'both',      
        top = False,         
        right = False,
        bottom = False,
        left = False)
ax[1, 0].plot(np.average(testdata[ymin_j:ymax_j, xmin_j:xmax_j], axis = 0))
ax[1, 0].set_xlabel('Nth column')
ax[1, 0].set_ylabel('Average pixel column value')
ax[1, 0].grid()
ax[1, 1].remove()
ax[0, 1].remove()
plt.savefig('Splitting_Justification', bbox_inches='tight')

#%% # Calculating N for all magnitudes for 3, 4, 5, 6, 10, 20, 30 sigma when splitting applied

print('Calculating N for splitting')
    
sN_3, ssortedmag_3 = counting(smag_3)      # Find the N count for each sigma
sN_4, ssortedmag_4 = counting(smag_4)
sN_5, ssortedmag_5 = counting(smag_5)
sN_6, ssortedmag_6 = counting(smag_6)
sN_10, ssortedmag_10 = counting(smag_10)
sN_20, ssortedmag_20 = counting(smag_20)
sN_30, ssortedmag_30 = counting(smag_30)

#%% # Fitting data for 5 sigma

print('Fitting')
def f(x,a,b):       # Defining function to fit the parameters
    return a*x+b

stofitN = np.log10(sN_5[1:100])    # Selecting straight line portion of Log(N) graph
stofitC = ssortedmag_5[1:100]
sfit5, sfiterror5 = curve_fit(f, stofitC, stofitN)
sx1 = np.arange(min(ssortedmag_5[1:]), ssortedmag_5[100],0.01)
sy1 = f(sx1,*sfit5)

#%% Plot log(N) graph with fitted line
fig, ax = plt.subplots()
plt.plot(ssortedmag_3, np.log10(sN_3), label = r'3 $\sigma$')
plt.plot(ssortedmag_4, np.log10(sN_4), label = r'4 $\sigma$')
plt.plot(ssortedmag_5, np.log10(sN_5), label = r'5 $\sigma$')
plt.plot(ssortedmag_6, np.log10(sN_6), label = r'6 $\sigma$')
plt.plot(ssortedmag_10, np.log10(sN_10), label = r'10 $\sigma$')
plt.plot(ssortedmag_20, np.log10(sN_20), label = r'20 $\sigma$')
plt.plot(ssortedmag_30, np.log10(sN_30), label = r'30 $\sigma$')

ax.axvline(ssortedmag_5[1],0,0.19, color = 'black', linestyle = ':')
ax.axvline(ssortedmag_5[100],0,0.83, color = 'black', linestyle = ':')
plt.plot(sx1,sy1,label=r'm = %.3f $\pm$ %.3f' % (sfit5[0],np.sqrt(sfiterror5[0][0])),
          color = 'red')
plt.legend(fontsize = 'small')
plt.xlabel(r'Magnitude')
plt.ylabel(r'$\log_{10}N$')
plt.grid()
plt.show()
plt.savefig('LogNmSplit', bbox_inches='tight')
print('Gradient obtained =',sfit5[0],'+-',np.sqrt(sfiterror5[0][0]))

#%% Show first 25 objects found
# Plotting
fig, ax = plt.subplots(5,5, dpi = 300)
x = 0
y = -1
for i in range(len(sloc_5[:25])):
    x = (i)%5
    if x == 0:
        y += 1
    xmin, xmax, ymin, ymax = [*sloc_5[i+1]] #52nd image wtf
    ax[y, x].imshow(testdata[ymin:ymax, xmin:xmax], cmap = 'viridis')
    ax[y, x].tick_params(
        axis = 'both',       # Changes apply to the x-axis
        which = 'both',      # Both major and minor ticks are affected
        bottom = False,      # Ticks along the bottom edge are off
        top = False,         # Ticks along the top edge are off
        right = False,
        left = False,
        labelbottom = False, # Labels along the bottom edge are off
        labelleft = False)   # Labels along the bottom edge are off
plt.show()
plt.savefig('LogNmSplit25', bbox_inches='tight')
