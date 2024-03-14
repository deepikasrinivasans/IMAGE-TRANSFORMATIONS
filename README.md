# EX-4 IMAGE TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import necessary libraries (NumPy, OpenCV, Matplotlib).
<br>

### Step2:
Read an image, convert it to RGB format, and display it using Matplotlib.Define translation parameters (e.g., shifting by 100 pixels horizontally and 200 pixels vertically).Perform translation using cv2.warpAffine().Display the translated image using Matplotlib.
<br>

### Step3:
Obtain the dimensions (rows, cols, dim) of the input image.Define a scaling matrix M with scaling factors of 1.5 in the x-direction and 1.8 in the y-direction.Perform perspective transformation using cv2.warpPerspective(), scaling the image by a factor of 1.5 in the x-direction and 1.8 in the y-direction.Display the scaled image using Matplotlib.
<br>

### Step4:
Define shear matrices M_x and M_y for shearing along the x-axis and y-axis, respectively.Perform perspective transformation using cv2.warpPerspective() with the shear matrices to shear the image along the x-axis and y-axis.Display the sheared images along the x-axis and y-axis using Matplotlib.
<br>

### Step5:
Define reflection matrices M_x and M_y for reflection along the x-axis and y-axis, respectively.Perform perspective transformation using cv2.warpPerspective() with the reflection matrices to reflect the image along the x-axis and y-axis.Display the reflected images along the x-axis and y-axis using Matplotlib.
<br>

### Step 6 :
Define an angle of rotation in radians (here, 10 degrees).Construct a rotation matrix M using the sine and cosine of the angle.Perform perspective transformation using cv2.warpPerspective() with the rotation matrix to rotate the image.Display the rotated image using Matplotlib.
<br>
### Step 7 :
Define a region of interest by specifying the desired range of rows and columns to crop the image (here, from row 100 to row 300 and from column 100 to column 300).Use array slicing to extract the cropped region from the input image.Display the cropped image using Matplotlib.
<br>


## Program:
```

#### Developed By: DEEPIKA S
#### Register Number: 212222230028
```
<table>
  <tr>
   <td width=50%>
     
### i)Image Translation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image trans.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M = np.float32([[1, 0, 100],[0, 1, 200],[0, 0, 1]])

translated_image = cv2.warpPerspective(input_image, M, (cols, rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()
```
</td>
<td>
  
### Output:
### i)Image Translation
![1](https://github.com/deepikasrinivasans/IMAGE-TRANSFORMATIONS/assets/119393935/a28e4c6f-3da3-48a0-9a37-53b04c37de11)
</td>
</tr>



<tr>
  <td width=50%>
  
### ii) Image Scaling
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image trans.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

rows, cols, dim = input_image.shape 
M = np.float32([[1.5, 0, 0],[0, 1.8, 0],[0, 0, 1]])

scaled_img=cv2.warpPerspective (input_image, M, (cols*2, rows*2))
plt.imshow(scaled_img)
plt.show()
```
</td>
<td>
  
### Output:
### ii) Image Scaling
![2](https://github.com/deepikasrinivasans/IMAGE-TRANSFORMATIONS/assets/119393935/f0b45940-2560-4a44-9b26-588dcf9750d1)
</td>
</tr>



<tr>
  <td width=50%>

### iii)Image shearing
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image trans.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

M_x = np.float32([[1, 0.5, 0],[0, 1 ,0],[0,0,1]])
M_y =np.float32([[1, 0, 0],[0.5, 1, 0],[0, 0, 1]])

sheared_img_xaxis=cv2.warpPerspective(input_image,M_x, (int(cols*1.5), int(rows *1.5)))
sheared_img_yaxis = cv2.warpPerspective(input_image,M_y,(int(cols*1.5), int(rows*1.5)))

plt.imshow(sheared_img_xaxis)
plt.show()

plt.imshow(sheared_img_yaxis)
plt.show()
```
</td>
<td>
  
### Output:
### iii)Image shearing
![3](https://github.com/deepikasrinivasans/IMAGE-TRANSFORMATIONS/assets/119393935/90dcce42-54b6-4cb2-a48c-731b82d307a0)
</td>
</tr>



<tr>
  <td width=50%>

### iv)Image Reflection
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image trans.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

M_x= np.float32([[1,0, 0],[0, -1, rows],[0, 0, 1]])
M_y =np.float32([[-1, 0, cols],[ 0, 1, 0 ],[ 0, 0, 1 ]])
# Apply a perspective transformation to the image
reflected_img_xaxis=cv2.warpPerspective (input_image, M_x,(int(cols), int(rows)))
reflected_img_yaxis= cv2.warpPerspective (input_image, M_y, (int(cols), int(rows)))

                                         
plt.imshow(reflected_img_xaxis)
plt.show()

plt.imshow(reflected_img_yaxis)
plt.show()
```
</td>
<td>

### Output:
### iv)Image Reflection
![4](https://github.com/deepikasrinivasans/IMAGE-TRANSFORMATIONS/assets/119393935/d938dfd2-b980-4a6a-be3b-c1dbd90a83a4)
</td>
</tr>



<tr>
 <td width=50%>

   ### v)Image Rotation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image trans.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rotated_img = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))

plt.imshow(rotated_img)
plt.show()
```
</td>
<td>
  
### Output:
### v)Image Rotation
![5](https://github.com/deepikasrinivasans/IMAGE-TRANSFORMATIONS/assets/119393935/dea3a326-bcc8-454b-9559-2d75c0b81ceb)
</td>
</tr>



<tr>
 <td width=50%>

### vi)Image Cropping
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("image trans.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

cropped_img= input_image[100:300,100:300]

plt.imshow(cropped_img)
plt.show()
```
</td>
<td>
  
### Output:
### vi)Image Cropping
![6](https://github.com/deepikasrinivasans/IMAGE-TRANSFORMATIONS/assets/119393935/c37a135e-61f9-4c1e-acdc-bab53e8fadbe)
</td>
</tr>
</table>

### Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
