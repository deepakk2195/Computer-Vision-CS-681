import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob
from skimage.transform import rescale, resize, downscale_local_mean

    
def non_max_suppression(img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

def threshold(img):
        selfweak_pixel=75
        selfstrong_pixel=255
        selflowthreshold=0.05
        selfhighthreshold=0.15
        highThreshold = img.max() * selfhighthreshold;
        lowThreshold = highThreshold * selflowthreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(selfweak_pixel)
        strong = np.int32(selfstrong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

def hysteresis(img):

        M, N = img.shape
        weak = 75
        strong = 255

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
    
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

ghist=[]
chist=[]
i=0
print("start of grayscale")
for image in glob.glob('ST2MainHall4/*.jpg'):
    img = cv2.imread(image,0)
    img=cv2.GaussianBlur(img,(7,7),2)
    canny = cv2.Canny(img,100,200)
    x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    magnitude,angle = cv2.cartToPolar(x,y,angleInDegrees=True)
    #angle=cv2.phase(x,y,angleInDegrees=True)
    angle = np.round(np.divide(angle,10))
##    u = np.cos(x)*y
##    v = np.sin(x)*y
####    img=resize(img, (img.shape[0]*10, img.shape[1]*10),
####                       anti_aliasing=True)
##    plot2 = plt.figure()
##   # plt.imshow(img,interpolation='nearest', aspect='auto')
##    plt.quiver(x,y,u,v)
##    
##    plt.title('Quiver Plot, Single Colour')
##    plt.show(plot2)
    canny=np.uint8(canny)
    angle = np.uint8(angle)
    hist=cv2.calcHist([angle],[0],canny,[36],[0,36])
    ghist.append(hist)

##    if i==0 or i==28 or i==65 or i==95:
##        plt.imshow(img,cmap='gray')
##        plt.title('grayscale image'+str(i))
##        plt.show()
##        plt.imshow(x,cmap='gray')
##        plt.title('grayscale x'+str(i))
##        plt.show()
##        plt.imshow(y,cmap='gray')
##        plt.title('grayscale y'+str(i))
##        plt.show()
##        plt.imshow(magnitude,cmap='gray')
##        plt.title('grayscale magnitude'+str(i))
##        plt.show()
##        plt.plot(hist)
##        plt.title('grayscale hist'+str(i))
##        plt.show()
    i+=1
        

i=0
print("end of grayscale")
print("start of color")
for image in glob.glob('ST2MainHall4/*.jpg'):
    img = cv2.imread(image)
    img=cv2.GaussianBlur(img,(7,7),2)
    img=img.astype(np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    (r,g,b)=(img[:,:,0], img[:,:,1], img[:,:,2])
    ranny = cv2.Canny(r,100,200)
    ganny = cv2.Canny(g,100,200)
    banny = cv2.Canny(b,100,200)
    canny = ranny+ganny+banny
    Rx = cv2.Sobel(r,cv2.CV_64F,1,0,ksize=5)
    Ry = cv2.Sobel(r,cv2.CV_64F,0,1,ksize=5)
    Gx = cv2.Sobel(g,cv2.CV_64F,1,0,ksize=5)
    Gy = cv2.Sobel(g,cv2.CV_64F,0,1,ksize=5)
    Bx = cv2.Sobel(b,cv2.CV_64F,1,0,ksize=5)
    By = cv2.Sobel(b,cv2.CV_64F,0,1,ksize=5)
    x=Rx+Gx+Bx
    y=Ry+Gy+By
    magnitude,angle=cv2.cartToPolar(x,y,angleInDegrees=True)
    #angle=cv2.phase(x,y,angleInDegrees=True)
    angle = np.round(np.divide(angle,10))
##    u = np.cos(x)*y
##    v = np.sin(x)*y
##    img=resize(img, (img.shape[0]*10, img.shape[1]*10),
##                       anti_aliasing=True)
##    plot2 = plt.figure()
##    plt.imshow(img,interpolation='nearest', aspect='auto')
##    plt.quiver(x,y,u,v)
##    
##    plt.title('Quiver Plot, Single Colour')
##    plt.show(plot2)
    canny=np.uint8(canny)
    angle = np.uint8(angle)
    print(i)
    hist=cv2.calcHist([angle],[0],canny,[36],[0,36])
    chist.append(hist)

##    if i==0 or i==28 or i==65 or i==95:
##        plt.imshow(img,cmap='gray')
##        plt.title('color image'+str(i))
##        plt.show()
##        plt.imshow(x,cmap='gray')
##        plt.title('color x'+str(i))
##        plt.show()
##        plt.imshow(y,cmap='gray')
##        plt.title('color y'+str(i))
##        plt.show()
##        plt.imshow(magnitude,cmap='gray')
##        plt.title('color magnitude'+str(i))
##        plt.show()
##        plt.plot(hist)
##        plt.title('color hist'+str(i))
##        plt.show()
    
    #eigen
##    e=[0,0]
##    a = np.square(Rx) + np.square(Gx) + np.square(Bx)
##    c = np.square(Ry) + np.square(Gy) + np.square(By)
##    b = np.multiply(Rx,Ry) + np.multiply(Gx,Gy) + np.multiply(Bx,By)
##    lamb1 = np.around((np.sqrt(np.square(a + c) - 4 * (np.square(b) - np.multiply(a, c))) + (a + c))/2,2)
##    lamb2 =np.around((np.sqrt(np.square(a + c) - 4 * (np.square(b) - np.multiply(a, c))) - (a + c))/2,2)
##
##    x = np.divide(-b,a-lamb1,where=a-lamb1!= 0)
##    e[0]=np.divide(x,np.sqrt(np.square(x) + 1))
##    e[1]=np.divide(1,np.sqrt(np.square(x) + 1))
##
##    phase=cv2.phase(e[0],e[1], angleInDegrees=True)
##    lamb1=np.sqrt(lamb1)
##    supress=non_max_suppression(lamb1,phase)
##    eigen=threshold(supress)
##    phase=np.rint(np.divide(phase,5))
##    hist,bins=np.histogram(phase,36,[0,36])
##    if i==0 or i==28 or i==65 or i==95:
##        plt.plot(hist)
##        plt.title('eigen hist'+str(i))
##        plt.show()
##        plt.imshow(eigen,interpolation='nearest', aspect='auto')
##        plt.savefig('eigen'+str(i)+'.png')
##        plt.title('eigen result'+str(i))
##        plt.show
    i+=1
##    plot2 = plt.figure()
##    plt.imshow(eigen,interpolation='nearest', aspect='auto')
##    plt.title('eigen')
##    plt.show(plot2) 

print("end of color")
gintersectmat = np.zeros(shape=(99,99))
gchimat = np.zeros(shape=(99,99))
cintersectmat = np.zeros(shape=(99,99))
cchimat = np.zeros(shape=(99,99))
print("start of intersect and chi")
for i in range(0,99):
    for j in range(0,99):
        gintersectmat[i][j]=(1-(intersect(ghist[i],ghist[j])))*255
        gchimat[i][j]=chi(ghist[i],ghist[j])
        cintersectmat[i][j]=(1-(intersect(chist[i],chist[j])))*255
        cchimat[i][j]=chi(chist[i],chist[j])

scaling=(max(gchimat.flatten()))/255
scaling1=(max(cchimat.flatten()))/255
for i in range(0,99):
    for j in range(0,99):
        gchimat[i][j]=(gchimat[i][j]/scaling)
        cchimat[i][j]=(cchimat[i][j]/scaling1)
print("end of intersect and chi")
plt.imshow(gintersectmat)
plt.colorbar()
plt.savefig('grayscale-Intersection_Comparison.png')
plt.title('grayscale-Intersection_Comparison')
plt.show()
plt.imshow(gchimat)
plt.colorbar()
plt.savefig('grayscale-Chi_Square_Comparison.png')
plt.title('grayscale-Chi_Square_Comparison')
plt.show()
plt.imshow(cintersectmat)
plt.colorbar()
plt.savefig('color-Intersection_Comparison.png')
plt.title('color-Intersection_Comparison')
plt.show()
plt.imshow(cchimat)
plt.colorbar()
plt.savefig('color-Chi_Square_Comparison.png')
plt.title('color-Chi_Square_Comparison')
plt.show()
