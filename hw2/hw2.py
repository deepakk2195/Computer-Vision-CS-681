import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import *
import mpldatacursor

img=None
   
def execute():
   global img
   img=e1.get()
   master.destroy()
   img=cv2.imread(img,1)
   img=cv2.resize(img, (1000, 500))
   cv2.imshow('nibondara',img)
   color = ('b','g','r')
   for i,col in enumerate(color):
       histr = cv2.calcHist([img],[i],None,[256],[0,256])
       plt.plot(histr,color = col)
       plt.xlim([0,256])
   #plt.savefig('histogram.png')
   plt.show()
   img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   fig, ax = plt.subplots()
   ax.imshow(img, interpolation='none', extent=[0, 1.5*np.pi, 0, np.pi])
   mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                         formatter='i, j = {i}, {j}\nz = {z}'.format)
   plt.show()
   img=cv2.resize(img, (500, 500))
   img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
   cv2.namedWindow('image')
   cv2.setMouseCallback('image',hover_and_crop)
   while True:
      cv2.imshow('image',img)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("c"):
         break
   cv2.destroyAllWindows()

def hover_and_crop(event, x, y, flags, param):
   global img
   if event == cv2.EVENT_MOUSEMOVE:
      win=img[y-5:y+6,x-5:x+6] 
      emo= cv2.copyMakeBorder(win,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
      cv2.namedWindow('window',cv2.WINDOW_NORMAL)
      cv2.imshow('window',emo)
      r,g,b=img[x,y]
      print("the intensity value at that point is",(r+g+b)/3)
      (means, stds) = cv2.meanStdDev(win)
      print("the mean at this point for all channels are",means)
      print("the standard deviation at this point for all channels are",stds)

master = Tk()
Label(master, text="Enter Image name with extension   P.S.:Please close the windows to continue with operations").grid(row=0)
e1 = Entry(master)
e1.grid(row=0, column=1)
Button(master, text='Execute', command=execute).grid(row=3, column=1, sticky=W, pady=4)
mainloop( )
