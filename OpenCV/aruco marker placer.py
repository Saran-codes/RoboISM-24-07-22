'''
Approach in steps:
>1. Finding the marker ids and correcting the orientation of the ArUco markers.
>2. Finding out the squares in the given image along with coordinates.
>3. Extracting the squares from the given image into seperate images using warp perspective.
>3. Identifying the colour of square.
>4. Pasting the respective ArUco markers on the sqaure.
>5. Reverse warping of above obtained images so they get back to their position and combining all these images into one.
>6. Filling the squares in the given image with black colour
>7. Combining the images obtained in above two steps to get final image

'''
import cv2
import numpy as np
import os


path1 = "Images/XD.jpg"
path2 = "Images/LMAO.jpg"
path3 = "Images/Ha.jpg"
path4 = "Images/HaHA.jpg"

paths = [path1,path2,path3,path4]

#The following code is used to detect aruco marker id and rename the aruco marker image to marker id
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
arucoParams = cv2.aruco.DetectorParameters_create()

for i in paths:

    image = cv2.imread(i)
    ids = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)[1][0][0]
    os.rename(i,"Images/"+str(ids)+".jpg")

#reading the images using renamed aruco paths
markerid_1 = cv2.imread("Images/1.jpg")
markerid_2 = cv2.imread("Images/2.jpg")
markerid_3 = cv2.imread("Images/3.jpg")
markerid_4 = cv2.imread("Images/4.jpg")
markers = [markerid_1,markerid_2,markerid_3,markerid_4]
#the following values are lower and upper range of (HUE, SATURATION, VALUE)
#I have taken the order of the colours as per the given colour for each ArUco marker 
colors_list = [[37,144,95,118,255,255],#green      #markerid_1
               [0,241,221,255,250,255],#orange     #marker_id_2
               [0,0,0,0,0,2],          #black      #marker_id3
               [4,0,228,89,29,235]]    #pink peach #marker_id4


#adding all the markers into a list in order               



img = cv2.imread("Images/CVtask.jpg")
imgcopy = img.copy()#creating a copy image of the given image

'''---------------------------------------------------------------------------------'''
'''
the following function detects all the squares in the given images and outputs the 
coordinates of the corners of all the squares in the image in form of a list
'''
def squaredetector(img):
    
    if len(img.shape) == 3 :
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray,50,50)
    else:
        canny = img.copy()    
    
    c,h = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for i in c:
        if cv2.contourArea(i)>100:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            nofcorners = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            aspectratio = w/float(h)
            if  nofcorners == 4 and (aspectratio >=0.90 and aspectratio<=1.10):
                squares.append(approx)
                
    return squares        

'''-----------------------------------------------------------------------------------'''
'''
the following function is used for warp perspective of the squares in the image
> So basically this function extracts all the squares in the image into seperate images along with coordinate values
'''
def warp(img,squares):
    #As the coordinates are not arranged properly we reaarange them properly by checking the sum and difference of x and y coordinates
    pts = np.squeeze(squares)    
    box_width = np.max(pts[:, 0]) - np.min(pts[:, 0])    
    box_height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)
    bounding_rect = np.array([pts[np.argmin(sum_pts)],
                     pts[np.argmin(diff_pts)],
                     pts[np.argmax(sum_pts)],
                     pts[np.argmax(diff_pts)]], dtype=np.float32)
    warped = np.array([[0, 0],
           [box_width - 1, 0],
            [box_width - 1, box_height - 1],
           [0, box_height - 1]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(bounding_rect, warped)
    warped_img = cv2.warpPerspective(img, transform_matrix, (box_width, box_height))
    return [warped_img,bounding_rect,warped]

'''-----------------------------------------------------------------------------------------'''         
'''
The following function is used to fill up the sqaures in imgcopy with black colour so that the colour of square doesnt
interfere with colour of ArUco markers when we are summing up all images to get final image
'''
def fill_up_sqaures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,50,50)
    c,h = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in c:
        if cv2.contourArea(i)>100:#to exclude any small disturbances
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            nofcorners = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            aspectratio = w/float(h)
            if nofcorners == 4 and (aspectratio >=0.90 and aspectratio<=1.10):
                cv2.drawContours(img,[i],-1,(0,0,0),cv2.FILLED)
    return img            

'''------------------------------------------------------------------------------------'''
'''
The following code can be used to paste/overlay one image on other image
> I used it for pasting the ArUco markers on the warped square images (or) just replacing 
  warped square images with ArUco markers before inverse warping
'''    
def paste(img1,img2):
    
    x,y,z = img1.shape
    img2 = cv2.resize(img2,(y,x))
    img1 = img2
    return img1

'''-------------------------------------------------------------------------------------'''
'''
The following code is used for inverse warping of the warped images
'''
def unwarp(img1,img2,bounding_rect,warped):#inverting both the points and rewarping is equivalent to unwarping
    transformation_matrix = cv2.getPerspectiveTransform(warped,bounding_rect)
    x,y,z = img1.shape
    warped_image = cv2.warpPerspective(img2,transformation_matrix,(y,x))           
    inverse_warp = cv2.addWeighted(warped_image,1,img1,0,0)  
    return inverse_warp           

'''-------------------------------------------------------------------------------------'''
'''
the following code is used to combine all the inverse warped images into one 
'''
def blend(inverse_warped,img):
    
    output = np.zeros(img.shape,np.uint8)
    for i in inverse_warped:
        output = cv2.addWeighted(output,1,i,1,0)
    output = output.astype(np.uint8)
    return output

'''-------------------------------------------------------------------------------------'''
'''
The following function is used to identify the colour of the square
'''
def findcolour(img3,col):
    imgHSV = cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
    
    for c in range(0,len(col)):
        lower = np.array(col[c][0:3])
        upper = np.array(col[c][3:6])
        mask = cv2.inRange(imgHSV,lower, upper)
        
        if len(squaredetector(mask)) > 0:
            return c    #As the markers are ordered according to colours it will be easy for us to assign right marker for the given colour by the index position of marker in markers list
    
'''-------------------------------------------------------------------------------------'''
'''
As the given ArUco markers are not properly oriented we need to first orient them properly into square shape using warp function
'''  
for i in range(0,len(markers)): #orienting ArUco markers    
    markers[i] = warp(markers[i],squaredetector(markers[i]))[0]
    

warpedimages = []                   #creating list to store all warped images
coordinates = squaredetector(img)   #finding the coordinates of squares in image using squaredetector function
inverse_warp = []                   #creating list to store all inverse warped images
pastedimages=[]                     #creating list to store all images from the output of paste function

'''_____________________________________________________________________________________'''
#The following is the code for whole approach to the problem:
for i in range(0,len(coordinates)):
    
    details = warp(img,coordinates[i])                                    #details contains warped sqaures and points used to warp    
    t = details[0].copy()                                                 
    warpedimages.append(details[0])                                            
    c = findcolour(warpedimages[i],colors_list)
    pastedimages.append(paste(t,markers[c]))                            
    inverse_warp.append(unwarp(img,pastedimages[i],details[1],details[2]))
'''____________________________________________________________________________________'''

combined_image = blend(inverse_warp,img) #combining all inverse warped images

imgcopy = fill_up_sqaures(imgcopy)    #filling the squares in given image with black so that colour of square doesnt interfere with that of ArUco markers

finalimage = cv2.addWeighted(combined_image,1,imgcopy,1,0) #finally this gives the combined image

if len(inverse_warp) == 0:
    print("No sqaures with the given colours were found in the given image") #If no square with given colour is found


cv2.imshow("intial", img)#intial image
cv2.imshow("final",finalimage)#final image

#cv2.imwrite("Images/final.jpg",finalimage)

cv2.waitKey(0)
cv2.destroyAllWindows()