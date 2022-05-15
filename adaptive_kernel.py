# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:52:35 2022

@author: gunja
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:21:09 2018

@author: gunjan_naik
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
from phasepack import tools,filtergrid
from scipy import linalg as LA
from pyica.fastica import fastica
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def is_unitary(m): 
    m=np.asmatrix(m)
    return np.allclose(np.eye(m.shape[0]), m.H * m)  
   
annoted_data_path=r"VSD.Brain_3more.XX.XX.OT.6560.mha"
t1_contrast_image=r"VSD.Brain.XX.O.MR_Flair.684.mha"
slice_no=62

reader=sitk.ImageFileReader()
reader.SetFileName(annoted_data_path)
segmented_image=reader.Execute()

reader.SetFileName(t1_contrast_image)
t1c_image=reader.Execute()

#sitk.Show(segmented_image)
segmented_image=segmented_image[:,:,slice_no]
t1c_image=t1c_image[:,:,slice_no]

segmented_image.SetOrigin((0.0,0.0))
t1c_image.SetOrigin((0.0,0.0))
#sitk_show(t1c_image)

np_segmented_image=sitk.GetArrayFromImage(segmented_image)
np_t1c_image=sitk.GetArrayFromImage(t1c_image)

img=cv2.normalize(np_t1c_image,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)



I=img

plt.figure(num='Original image')
plt.subplot(1,2,1)
plt.imshow(I,cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(np_segmented_image,cmap='gray')
plt.axis('off')
plt.show()
[I_row,I_col]=np.shape(I)

I=np.double(I)
#I=imresize(I,(64,64))
patch_size=8
patch_shape = patch_size, patch_size

    
# -- filterbank1 on original image
patches=extract_patches_2d(I,patch_shape)
X =patches.reshape(-1, patch_size * patch_size)
X=np.double(np.transpose(X))

X_mx=X-np.mean(X)
X_temp=np.dot(X_mx,np.transpose(X_mx))
W_0=np.linalg.inv(LA.sqrtm(X_temp))
X_new=np.dot(W_0,X_mx)

A,W,S=fastica(X_new)

g=np.dot(W,A)

W_by_Ainv=np.linalg.inv(A)

g=np.dot(W_by_Ainv,A)

#############Orthogonalization of W ############################
W_temp=np.dot(np.transpose(W_by_Ainv),W_by_Ainv)
w_temp=np.linalg.inv(LA.sqrtm(W_temp))
W_new=np.real(np.dot(W_by_Ainv,w_temp))

A_new=np.linalg.inv(W_new)

print( 'New W is unitary '+str(is_unitary(W_new)))
print( 'New A is unitary '+str(is_unitary(A_new)))

################For Making matrix unitary#######################
W=W_new
Y=np.dot(W,X_new)

[Y_rows,Y_cols]=np.shape(Y)

g_Y=np.asmatrix(np.zeros((Y_rows,Y_cols)))

for Y_rows_index in range(0,Y_rows):
    for Y_cols_index in range(0,Y_cols):
        g_Y[Y_rows_index,Y_cols_index]=1-(2.0/(1+np.exp(-Y[Y_rows_index,Y_cols_index])))
        

del_W=np.dot((np.eye(patch_size*patch_size)+np.dot(g_Y,np.transpose(Y))),W)
W1=del_W
#
W_I=np.dot(W1,W_0)

W_temp1=np.dot(np.transpose(W_I),W_I)
w_temp1=np.linalg.inv(LA.sqrtm(W_temp1))
W_New=np.dot(W_I,w_temp1)

Filters=np.asarray((W_New))

print(is_unitary(Filters))

filter_norm=[]
temp=0

plt.figure(num='ICA components')
for i, f in enumerate(Filters):
    plt.subplot(patch_size,patch_size, i + 1)
    print(i,np.linalg.norm(f,0.1))
    filter_norm.append((np.linalg.norm(f,0.1),f))
    f=f.reshape(patch_size,patch_size)
    plt.imshow(f, cmap="gray")
    plt.axis("off")
plt.show()
normlist=[]


for i, f in enumerate(Filters):
    normlist.append((np.linalg.norm(f,0.1),f))

normlist.sort()



[patches_slice,patches_row,patches_col]=np.shape(patches)

new_patches1=np.zeros(np.shape(patches))    

new_img_list_mix=[]



######################## Working method########################


[X_row,X_col]=X.shape

X_reconstructed=np.zeros(patches.shape)

for X_col_index in range(X_col):
    temp_s=np.dot(W_New,X[:,X_col_index].reshape(-1,1))
    reconstructed_x=np.dot(W_New.T,temp_s)
    X_reconstructed[X_col_index,:,:]=reconstructed_x.reshape(patch_shape)
    
new_img=reconstruct_from_patches_2d(X_reconstructed,(I_row,I_col))   
plt.figure()
plt.imshow(new_img,cmap='gray')
plt.show()

ICA_filters=[]
    
for i, f in enumerate(Filters):   
    new_f=cv2.resize(f.reshape(patch_shape),(I.T.shape),interpolation=cv2.INTER_CUBIC)
    ICA_filters.append((np.linalg.norm(f,0.2),new_f))
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(f.reshape(patch_shape),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(new_f,cmap='gray')
    plt.show()
    

new_img_list=[]

plt.figure(num='Extracted component')
for i, f in enumerate(Filters):    
    X_reconstructed=np.zeros(patches.shape)
    plt.subplot(patch_size,patch_size, i + 1)    
    W_edge=f.reshape(1,-1)
    for X_col_index in range(X_col):
        temp_s=np.dot(W_edge,X[:,X_col_index].reshape(-1,1))
        reconstructed_x=np.dot(W_edge.T,temp_s)
        X_reconstructed[X_col_index,:,:]=reconstructed_x.reshape(patch_shape)
    new_img=reconstruct_from_patches_2d(X_reconstructed,(I_row,I_col))   
    
    plt.imshow(np.abs(new_img), cmap="gray")
    plt.axis("off")
    new_img_list.append((np.linalg.norm(new_img,2),new_img))
plt.show()
new_img_list.sort(reverse=True)    

new_img_temp=0

plt.figure(num='Complete reconstruction')
for new_img_index in range(patch_size*patch_size):
    plt.subplot(patch_size,patch_size, new_img_index + 1)     
    
    new_Img=new_img_list[new_img_index]
    New_img=new_Img[1]    
    
    plt.imshow(np.abs(New_img),cmap='gray')
    plt.axis("off")        
    new_img_temp=new_img_temp + New_img
plt.show()

plt.figure(num='texture',dpi=100)
plt.subplot(1,2,1)
plt.imshow(I,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(np.abs(new_img_temp),cmap='gray')
plt.title('complete reconstruction')
plt.show()

new_img_temp=0

plt.figure(num='Sorted Extracted components')
for new_img_index in range(1,patch_size*patch_size):
    plt.subplot(patch_size,patch_size, new_img_index + 1)     
    
    new_Img=new_img_list[new_img_index]
    New_img=new_Img[1]    
    plt.imshow(np.abs(New_img),cmap='gray')
    plt.axis("off")        

    new_img_temp=new_img_temp + New_img    
plt.show()

#
plt.figure(num='texture',dpi=100)
plt.subplot(1,2,1)
plt.imshow(I,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(np.abs(new_img_temp),cmap='gray')
plt.title('edge map')
plt.show()

edge_map=np.abs(new_img_temp)
edge_map_uc=cv2.normalize(edge_map.astype('float'),None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

# Getting Sparset Adaptive kernel
ICA_filters.sort(reverse=True)


plt.figure()
plt.imshow(ICA_filters[0][1],cmap='gray')
plt.show()

nscale=2
mult=10.1
k=4
cutOff=0.5
g=10.
noiseMethod=-1
deviationGain=1.5
IM=I
epsilon = 1E-4          # used to prevent /0.
_, IM = tools.perfft2(img)     # periodic Fourier transform of image
rows, cols = img.shape
zeromat = np.zeros((rows, cols), dtype='float64')
sumAn = zeromat.copy()
sumf = zeromat.copy()
sumh1 = zeromat.copy()
sumh2 = zeromat.copy()
#for scale_count in range(5,10):
#    print "The wavelet scale is "+ str(scale_count)
#    wavelet_scale=np.double(scale_count)
#    octave.eval("pkg load image")
    #[PC,a]=octave.eval("phasecongmono(img,9);")
#    octave.push("Wavelet_scale",wavelet_scale)
    #octave.eval("Wavelet_scale=5")
#    octave.eval("[PC, OR, ft, T,logGabor]=phasecongmono_with_ICA_filters(img,ICA_filters=Adaptive_filter);")
#    octave.eval("[PC_ICA, OR, ft, T,logGabor]=phasecongmono(ICA_edge,nscale=Wavelet_scale);")
    #octave.eval("[PC, OR, ft, T,logGabor]=phasecongmono(img,nscale=9);")
lp = tools.lowpassfilter((rows, cols), .45, 15)
#logGabor=ICA_filters[0][1]
#logGabor1=cv2.normalize(-1*ICA_filters[0][1],0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)*100
adaptive_filter=ICA_filters[0][1]
if adaptive_filter.min()<0:
    logGabor1=cv2.normalize(-1*adaptive_filter,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
else:
    logGabor1=cv2.normalize(adaptive_filter,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)

#logGabor1 *= lp      # Apply the low-pass filter
#logGabor[0, 0] = 0.  # Undo the radius fudge
IMF = IM * logGabor1   # Frequency bandpassed image
f = np.real(tools.ifft2(IMF))  # Spatially bandpassed image

# Bandpassed monogenic filtering, real part of h contains
# convolution result with h1, imaginary part contains
# convolution result with h2.

radius, u1, u2 = filtergrid.filtergrid(rows, cols)

# Get rid of the 0 radius value at the 0 frequency point (at top-left
# corner after fftshift) so that taking the log of the radius will not
# cause trouble.
radius[0, 0] = 1.

# Construct the monogenic filters in the frequency domain. The two filters
# would normally be constructed as follows:
#    H1 = i*u1./radius
#    H2 = i*u2./radius
# However the two filters can be packed together as a complex valued
# matrix, one in the real part and one in the imaginary part. Do this by
# multiplying H2 by i and then adding it to H1 (note the subtraction
# because i*i = -1).  When the convolution is performed via the fft the
# real part of the result will correspond to the convolution with H1 and
# the imaginary part with H2. This allows the two convolutions to be done
# as one in the frequency domain, saving time and memory.
H = (1j * u1 - u2) / radius

# The two monogenic filters H1 and H2 are not selective in terms of the
# magnitudes of the frequencies. The code below generates bandpass log-
# Gabor filters which are point-wise multiplied by IM to produce different
# bandpass versions of the image before being convolved with H1 and H2
#
# First construct a low-pass filter that is as large as possible, yet falls
# away to zero at the boundaries. All filters are multiplied by this to
# ensure no extra frequencies at the 'corners' of the FFT are incorporated
# as this can upset the normalisation process when calculating phase
# congruency

# Updated filter parameters 6/9/2013:   radius .45, 'sharpness' 15
lp = tools.lowpassfilter((rows, cols), .45, 15)
h = tools.ifft2(IMF * H)
h1, h2 = np.real(h), np.imag(h)

# Amplitude of this scale component
An = np.sqrt(f * f + h1 * h1 + h2 * h2)

# Sum of amplitudes across scales
sumAn += An
sumf += f
sumh1 += h1
sumh2 += h2
ss=0
# At the smallest scale estimate noise characteristics from the
# distribution of the filter amplitude responses stored in sumAn. tau
# is the Rayleigh parameter that is used to describe the distribution.
if ss == 0:
    # Use median to estimate noise statistics
    if noiseMethod == -1:
        tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))
    
    # Use the mode to estimate noise statistics
    elif noiseMethod == -2:
        tau = tools.rayleighmode(sumAn.flatten())
    
    maxAn = An
else:
    # Record the maximum amplitude of components across scales to
    # determine the frequency spread weighting
    maxAn = np.maximum(maxAn, An)


# Form weighting that penalizes frequency distributions that are
# particularly narrow. Calculate fractional 'width' of the frequencies
# present by taking the sum of the filter response amplitudes and
# dividing by the maximum amplitude at each point on the image.   If
# there is only one non-zero component width takes on a value of 0, if
# all components are equal width is 1.
width = (sumAn / (maxAn + epsilon) - 1.) / (nscale - 1)

# Calculate the sigmoidal weighting function for this
# orientation
weight = 1. / (1. + np.exp(g * (cutOff - width)))

# Automatically determine noise threshold

# Assuming the noise is Gaussian the response of the filters to noise
# will form Rayleigh distribution. We use the filter responses at the
# smallest scale as a guide to the underlying noise level because the
# smallest scale filters spend most of their time responding to noise,
# and only occasionally responding to features. Either the median, or
# the mode, of the distribution of filter responses can be used as a
# robust statistic to estimate the distribution mean and standard
# deviation as these are related to the median or mode by fixed
# constants. The response of the larger scale filters to noise can then
# be estimated from the smallest scale filter response according to
# their relative bandwidths.

# This code assumes that the expected reponse to noise on the phase
# congruency calculation is simply the sum of the expected noise
# responses of each of the filters. This is a simplistic overestimate,
# however these two quantities should be related by some constant that
# will depend on the filter bank being used. Appropriate tuning of the
# parameter 'k' will allow you to produce the desired output.

# fixed noise threshold
if noiseMethod >= 0:
    T = noiseMethod


# Estimate the effect of noise on the sum of the filter responses as
# the sum of estimated individual responses (this is a simplistic
# overestimate). As the estimated noise response at succesive scales is
# scaled inversely proportional to bandwidth we have a simple geometric
# sum.
else:
    totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))
    
    # Calculate mean and std dev from tau using fixed relationship
    # between these parameters and tau. See:
    # <http://mathworld.wolfram.com/RayleighDistribution.html>
    EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
    EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)
    
    # Noise threshold, must be >= epsilon
    T = np.max((EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon))

# Final computation of key quantities
ori = np.arctan(-sumh2 / sumh1)

# Wrap angles between -pi and pi and convert radians to degrees
ori = np.fix((ori % np.pi) / np.pi * 180.)

# Feature type (a phase angle between -pi/2 and pi/2)
ft = np.arctan2(sumf, np.sqrt(sumh1 * sumh1 + sumh2 * sumh2))

# Overall energy
energy = np.sqrt(sumf * sumf + sumh1 * sumh1 + sumh2 * sumh2)

# Compute phase congruency. The original measure,
#
#   PC = energy/sumAn
#
# is proportional to the weighted cos(phasedeviation).  This is not very
# localised so this was modified to
#
#   PC = cos(phasedeviation) - |sin(phasedeviation)|
# (Note this was actually calculated via dot and cross products.)  This
# measure approximates
#
#   PC = 1 - phasedeviation.
#
# However, rather than use dot and cross products it is simpler and more
# efficient to simply use acos(energy/sumAn) to obtain the weighted phase
# deviation directly. Note, in the expression below the noise threshold is
# not subtracted from energy immediately as this would interfere with the
# phase deviation computation. Instead it is applied as a weighting as a
# fraction by which energy exceeds the noise threshold. This weighting is
# applied in addition to the weighting for frequency spread. Note also the
# phase deviation gain factor which acts to sharpen up the edge response. A
# value of 1.5 seems to work well. Sensible values are from 1 to about 2.

phase_dev = np.maximum(
    1. - deviationGain * np.arccos(energy / (sumAn + epsilon)), 0)
energy_thresh = np.maximum(energy - T, 0)

M = weight * phase_dev * energy_thresh / (energy + epsilon)
PC=M
#    PC_ICA=octave.pull("PC_ICA")

img_u=cv2.normalize(img.astype('float'),None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
PC_u=cv2.normalize(PC.astype('float'),None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

PC_ICA=cv2.subtract(PC_u,edge_map_uc)

plt.figure(dpi=130)
plt.subplot(1,4,1)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.subplot(1,4,2)
plt.imshow(PC,cmap='gray')
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(PC_ICA,cmap='gray')
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(np_segmented_image,cmap='gray')
plt.axis('off')
plt.show()

def segmentation_to_binary(img):
    [r,c]=img.GetSize()
    
    img_mask=sitk.Image(r,c,sitk.sitkUInt8)
    
    for r_index in range(r):
        for c_index in range(c):     
            if(img[r_index,c_index]>0.0):
                img_mask[r_index,c_index]=1
            else:
                img_mask[r_index,c_index]=0
    return img_mask

Img=segmentation_to_binary(sitk.GetImageFromArray(PC_ICA))
Annotated_Image=segmentation_to_binary(sitk.GetImageFromArray(np_segmented_image))

C=sitk.LabelOverlapMeasuresImageFilter()
C.Execute(Img,Annotated_Image)
print('The dice score is '+str(C.GetDiceCoefficient()))

plt.figure(dpi=130)
plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(PC_ICA,cmap='gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(np_segmented_image,cmap='gray')
plt.axis('off')
plt.show()