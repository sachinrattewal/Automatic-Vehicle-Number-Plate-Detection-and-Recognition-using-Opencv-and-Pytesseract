import cv2
import imutils #to resize the image
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#reading the image
image = cv2.imread('image_7.jpg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)
cv2.waitKey(0)

# converting to grayscale image

gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray)
cv2.waitKey(0)


# Smoothening the image
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Smoother Image" , gray)
cv2.waitKey(0)

edged = cv2.Canny(gray , 170, 200)
cv2.imshow("Canny Edged", edged)
cv2.waitKey(0)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0,255,0), 3)
# cv2.imshow("Canny after Contouring", image1)
cv2.waitKey(0)


cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

image2 = image.copy()
cv2.drawContours(image2 , cnts, -1, (0,255,0),3)
# cv2.imshow("Top 30 Contours", image2)
cv2.waitKey(0)

count = 0
name = 1

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i , 0.02*perimeter, True)
    if(len(approx) == 4):
        NumberPlateCount = approx
        x , y ,w ,h = cv2.boundingRect(i)
        crp_img = image[y:y+h , x:x+w]
        cv2.imwrite(str(name)+ '.png' , crp_img)
        name += 1
        break

cv2.drawContours(image, [NumberPlateCount], -1, (0,255,0), 3)
cv2.imshow("Final Image", image)
cv2.waitKey(0)
# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCount],0,255,-1,)
new_image = cv2.bitwise_and(image,image,mask=mask)
# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]
crop_img_loc = '1.png'
cv2.imshow("Cropped Image" , cv2.imread(crop_img_loc))
cv2.waitKey(0)
#Read the number plate
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Detected license plate Number is:",text)
