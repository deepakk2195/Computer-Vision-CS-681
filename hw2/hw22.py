import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob

def convind(img):
    #img=cv2.imread(img,1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    (r,g,b)=(img[:,:,0], img[:,:,1], img[:,:,2])
    conv=((r//32)*64)+((g//32)*8)+(b//32)
    return conv

def intersect(h1,h2):
    mini=0
    maxi=0
    for i in range(0,len(h1)):
        mini+= min(h1[i],h2[i])
        maxi+= max(h1[i],h2[i])
    return float(mini/maxi)

def chi(h1,h2):
    chi=0
    for i in range(0,len(h1)):
        if (h1[i]+h2[i])>5:
            chi+=(((h1[i]- h2[i])**2)/float(h1[i]+h2[i]))
    return chi
'''
hhist=[]
for img in glob.glob('ST2MainHall4/*.jpg'):
    histr,bins = np.histogram(convind(img),512,[0,512])
    hhist.append(histr)
 
intersectmat = np.zeros(shape=(99,99))
chimat = np.zeros(shape=(99,99))

for i in range(0,99):
    for j in range(0,99):
        intersectmat[i][j]=(intersect(hhist[i],hhist[j]))*255
        chimat[i][j]=chi(hhist[i],hhist[j])

scaling=(max(chimat.flatten()))/255
for i in range(0,99):
    for j in range(0,99):
        chimat[i][j]=(chimat[i][j]/scaling)

       
plt.imshow(intersectmat)
plt.colorbar()
plt.savefig('Intersection_Comparison.png')
plt.title('Intersection_Comparison')
plt.show()
plt.imshow(chimat)
plt.colorbar()
plt.savefig('Chi_Square_Comparison.png')
plt.title('Chi_Square_Comparison')
plt.show()
'''
file=cv2.imread("ST2MainHall4//ST2MainHall4001.jpg")
histr,bins = np.histogram(convind(file),512,[0,512])
plt.plot(histr)
plt.show()
