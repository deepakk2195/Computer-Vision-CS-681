from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from collections import defaultdict

def Nmaxelements(list1,list2, N): 
    final_list = [] 
    bins_list=[]
    list3=list1.copy()
    for i in range(0, N):  
        max1 = 0
          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
                  
        list1.remove(max1); 
        final_list.append(max1) 

    for l in final_list:
        bins_list.append(list2[list3.index(l)])
    return bins_list 
  


def segment_by_angle_kmeans(lines, k=2, **kwargs):


    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented



def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections




def ang(img,edges):
    x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    magnitude,angle = cv2.cartToPolar(x,y,angleInDegrees=True)
    print(angle.shape)
    if angle.any()>180:
        angle=angles-180
    print(angle.shape)
    angle = np.round(np.divide(angle,5))
    angle = np.uint8(angle)
    hist,bins=np.histogram(angle,36,[1,36])
    largest=0
    second_largest=0
    plt.plot(hist)
    plt.show()
    hist=hist.tolist()
    bins=bins.tolist()
    largel=Nmaxelements(hist,bins, 5)
##    for j in hist:
##        if j > largest:
##            largest = j
##        elif largest > j > second_largest:
##            second_largest = j
##    largest=bins[hist.index(largest)]
##    second_largest=bins[hist.index(second_largest)]
##    print(largest, second_largest)
    print(largel)
    return largel



def Findlines(img,image,alph,findVanishing = False):
    h=image.shape[0]
    w=image.shape[1]
    img1 = np.zeros((h,w),np.uint8)
    D=defaultdict(list)
    d=0
    for i in range(0,h):
        for j in range(0,w):
            if image[i,j]>0:
                d=(np.multiply(j,np.cos(math.radians(alph))))+(np.multiply(i,math.sin(math.radians(alph))))
                D[d].append((i,j))

##    for i,(k,v) in enumerate(D.items()):
##        if len(v)>1:
##                print(i,k,len(v))
##    DHist = {k:len(D[k]) for k in sorted(D, key=lambda k: len(D[k]), reverse=True)}

##    for i,(k,v) in enumerate(DHist.items()):
##        if v>1:
##                print(i,k,v)

    first2pairs = {k: D[k] for k in sorted(D, key=lambda k: len(D[k]), reverse=True)[:100]}
    returnDList = []
    for k,v in first2pairs.items():
        if len(v)>20:
            returnDList.append([[k,alph]])
            a = math.cos(math.radians(alph))
            b = math.sin(math.radians(alph))
            x0 = a * k
            y0 = b * k
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            #cv2.line(img, pt2, pt1, (255,0,0), 1)
            for i,(x,y) in enumerate(v):
                #cv2.circle(img,(y,x), 5, (0,0,255), -1)
                img1[x,y]=255
            
##    plt.imshow(img)
##    plt.show()
##    plt.imshow(img1)
##    plt.show()
    
    return img,returnDList

img = cv2.imread('5.jpg')
img = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200,None,3)
plt.imshow(edges)
plt.show()


##
##angle=90
##D,lists=Findlines(img,edges,angle)
####print(lists)
##s=ang(gray,edges)
##dlists=[]
##dlists2=[]
##for i in s:
##    i=np.int(np.round(i*5))
##    print(i)
##    for k in range(i-5,i+6):
##        img,lists=Findlines(img,edges,k,True)
##        print(k)
##        dlists=dlists+lists
##    dlists2=dlists2+dlists
##plt.imshow(img)
##plt.show()
####print(dlists2)
####segmented = segment_by_angle_kmeans(dlists2)
####intersections = segmented_intersections(segmented)
####
####for point in intersections:
####    cv2.circle(img,(point[0],point[1]),3, (0,0,255), -1)
####
####plt.imshow(img)
####plt.show()
####
####
####
####
######l=np.int(np.round(l*5))
######s=np.int(np.round(s*5))
######print(l,s)
######for i in range(l-3,l+6):
######    print(i)
######    img=Findlines(img,edges,i)    
######
######for i in range(s-3,s+6):
######    print(i)
######    img=Findlines(img,edges,i)
##
cdst = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
#hough
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
segmented = segment_by_angle_kmeans(lines)
intersections = segmented_intersections(segmented)
print(lines)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 1)

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1)



cv2.imshow("Source", img)
cv2.imshow("canny",edges)
#cv2.imwrite("canny.png",edges)
#cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
plt.imshow(cdst)
plt.show()
plt.imshow(cdstP)
plt.show()

for point in intersections:
    cv2.circle(img,(point[0],point[1]),3, (0,0,255), -1)

plt.imshow(img)
plt.show()

