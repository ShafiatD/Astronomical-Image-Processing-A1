# Astronomical Image Processing. A1
## Shafiat Dewan, Charlotte Feazey-Noble

To invoke the results found in the project report the A1Final.py file should be 
opened and executed. Ensure astropy is installed.

The find function defined in the code takes an input of the astronomical CCD 
image and finds the number of objects within the image and their associated 
locations and magnitudes. The code also then saves the data found as a txt file 
in the source folder. The find function has a splitting feature that helps to 
distinguish two close sources which can be toggled on and off.

The code written beyond the defining of functions fits the histogram of the 
fits image with a gaussian and then uses that data as an input to the find 
function to find the number of counts and assocciated magnitudes for the image 
given different noise tolerances and with/without the splitting feature. 
It also plots the graphs for log(N) given the different inputs.

Figure dpi has been set to 300 at the start of the file, this may be unnecesary
for montitor with a resolution no more than 1080p, so to speed up the execution
dpi can be reduced. The code takes approximately 5-10 minutes to finish running.

**Ensure that the mosaic.fits file is in the same directory as A1Final.py**