import os
import cv2
import numpy as np
from skimage.draw import disk

diff = 0

#Crops black area. Simpler than function outer_crop
def crop_zero(image): 
    y_nonzero, x_nonzero = np.nonzero(image)
    try:
        return (np.min(y_nonzero),np.max(y_nonzero)), (np.min(x_nonzero),np.max(x_nonzero)), \
        image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
    except:
        raise Exception('NG! Empty image')
    
#Calculating nearest distance between detected inner centroids and outer centroids
def findNearest(A, B): 
    minDiff = 9999
    for a in range(len(A)):
        for b in range(len(B)):
            diff = abs(A[a]-B[b])
            if diff < minDiff:
                minDiff = diff  # Update smallest difference found so far
                coord = [a,b]
    return minDiff, coord

#Preprocessing image: read->open->blur->crop
def preprocess(fn): 
    try:
        img = cv2.imread(fn, cv2.IMREAD_COLOR)  # Read image. 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale. 
    except:
        raise Exception(f'NG! {fn} is invalid image')
    
    kernel_size = int(8 * diff)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, \
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))) #Open Morph
    y_range, x_range, cropped = crop_zero(opened) #Crop
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(cropped, (3, 3)) # Blur using 3 * 3 kernel
    orig_cropped = img[y_range[0]:y_range[1],x_range[0]:x_range[1]]

    return gray_blurred, orig_cropped

#Judging tip quality based on centroid distance (concentricity) and inner quality (cleanliness)
def verdict(outer, inner, orig, fn): 
    outer = np.uint16(np.around(outer)) 
    inner = np.uint16(np.around(inner))

    #Storing coordinates as complex for simpler calculation
    centroid_outer = [complex(a[0], a[1]) for a in outer[0,:]]
    centroid_inner = [complex(a[0], a[1]) for a in inner[0,:]]

    minDiff, coord = findNearest(centroid_inner, centroid_outer)
    
    #Drawing circles (outer)
    a, b, r = outer[0,:][coord[1]][0], outer[0,:][coord[1]][1], \
        outer[0,:][coord[1]][2] 
    cv2.circle(orig, (a, b), r, (0, 255, 0), 2) 
    cv2.circle(orig, (a, b), 1, (0, 0, 255), 3) 

    #Drawing circles (inner)
    a, b, r = inner[0,:][coord[0]][0], inner[0,:][coord[0]][1], \
    inner[0,:][coord[0]][2] 
    cv2.circle(orig, (a, b), r, (0, 255, 0), 2) 
    cv2.circle(orig, (a, b), 1, (0, 0, 255), 3) 

    rr, cc = disk((a, b), r)
    cv2.imwrite(f'{fn[:-4]}_out.png',orig) #Even almost good result is considered (for verification)

    #Calculating cleanliness of inner circle
    bad_shape = 0
    for a in range(len(rr)):
        if orig[cc[a]][rr[a]][0]>0: bad_shape +=1

    #Verdict. All NGs are raised, and goods (and arguably good) are returned
    if (bad_shape>(orig[:,:,0].size)/2):        
        raise Exception("NG! Almost good")
    elif (bad_shape>(orig[:,:,0].size)/3):
        if (minDiff < (abs(complex(orig.shape[0], orig.shape[1])) * 0.3 / diff)):
            return ("Arguably good")
        else:
            raise Exception("NG! Almost good")
    elif (minDiff > (abs(complex(orig.shape[0], orig.shape[1])) * 0.5 / diff)):
        raise Exception("NG! Almost good")
        
    return "GO!"

#Crops area outside the outer circle. 
# #Uses HoughCircle--more resource intensive than crop_zero
def outer_crop(orig, blur): 
    crop_area = cv2.HoughCircles(blur, 
                                cv2.HOUGH_GRADIENT, 1, int(min(gray_blurred.shape)/6), param1 = int(50*diff), 
                                param2 = int(30*diff), minRadius = int(min(gray_blurred.shape)/4), 
                                maxRadius = max(gray_blurred.shape))
    if crop_area is None: raise Exception(f'NG!{fn} outer is scrambled')
    cr_area = crop_area[0,:][0]
    rr, cc = disk((cr_area[1], cr_area[0]), cr_area[2])
    rr[rr<0]=0
    cc[cc<0]=0
    blur_crop = blur[np.min(rr):np.max(rr), np.min(cc):np.max(cc)]
    orig_crop = orig[np.min(rr):np.max(rr), np.min(cc):np.max(cc)]

    return orig_crop, blur_crop

#Main function
if __name__ == "__main__":
    fns = ''
    while(not os.path.isfile(fns)): #Takes image file name
        fns = input("Enter file/folder name: ")
        if (os.path.isdir(fns)):
            fns = [fns+f'/{file}' for file in os.listdir(fns)]
            break
        if not isinstance(fns, list):
            fns = [fns]
    while(diff < 0.8 or diff > 1.2): #Takes difficulty value (higher is stricter)
        diff = float(input("Enter strictness (0.8-1.2): "))
    for fn in fns:
        try:
            gray_blurred, orig_cropped = preprocess(fn) #Preprocess image

            #Determining need for outer_crop
            if (float(min(gray_blurred.shape))/float(max(gray_blurred.shape)) < (0.8 * diff)): 
                orig_cropped, gray_blurred = outer_crop(orig_cropped, gray_blurred)

            # Hough for verdict. This is where shape quality is determined
            detected_circles_inner = cv2.HoughCircles(gray_blurred, 
                                    cv2.HOUGH_GRADIENT, 1, min(40, int(min(gray_blurred.shape)/6)), param1 = 80*diff, 
                                    param2 = 50*diff, minRadius = min(40, int(min(gray_blurred.shape)/8)), 
                                    maxRadius = int(min(gray_blurred.shape)/4)) 
            
            detected_circles_outer = cv2.HoughCircles(gray_blurred, 
                                    cv2.HOUGH_GRADIENT, 1, int(min(gray_blurred.shape)/6), param1 = 50*diff, 
                                    param2 = 30*diff, minRadius = int(min(gray_blurred.shape)/4), 
                                    maxRadius = max(gray_blurred.shape))

            #Calculating verdict from Hough Circles
            if (detected_circles_outer is not None) and (detected_circles_inner is not None): 
                v = verdict(detected_circles_outer, detected_circles_inner, orig_cropped, fn)
                print(v)
            else:
                raise Exception("NG! Not well-built")
        except Exception as e:
            print(e)
