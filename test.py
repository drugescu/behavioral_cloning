import cv2

# Load image
#center_image = cv2.cvtColor(cv2.imread('./IMG/center_2016_12_01_13_30_48_287.jpg'), cv2.COLOR_BGR2RGB)
center_image = cv2.imread('./IMG/center_2016_12_01_13_30_48_287.jpg') #convertion to rgb messes it up here but required in drive.py

# Trim image upper side
FROM_TOP = 60
FROM_BOTTOM = 20
output = center_image[0 + FROM_TOP : 160 - FROM_BOTTOM, 0 : 320, 0 : 3]

# Save image
#cv2.imwrite('./IMG/center_cropped.jpg', output) 

### ------------------------------------------------------------

new_size_x, new_size_y = 200, 66
old_size_x, old_size_y = 320, 160

center2 = cv2.imread('./IMG/center_2016_12_01_13_30_48_287.jpg')
FROM_TOP    = 60
FROM_BOTTOM = 20

output2  = center2[0 + FROM_TOP : old_size_y - FROM_BOTTOM, 0 : old_size_x, 0 : 3]

output2 = cv2.resize(output2, (new_size_x, new_size_y), interpolation = cv2.INTER_AREA)
cv2.imwrite('./IMG/center_cropped.jpg', output2) 

### ------------------------------------------------------------
# Test drive.py
import base64
from PIL import Image
from io import BytesIO

#image = Image.open(BytesIO(base64.b64decode(imgString)))
image = Image.open('./IMG/center_2016_12_01_13_30_48_287.jpg')
# left, upper, righ, lower
image = image.crop((0, FROM_TOP, 320, 160 - FROM_BOTTOM))
image = image.resize((200, 66))

# left, left + width, upper, upper+height
image.save('./IMG/center_cropped_drivepy.jpg')
#image_array = np.asarray(image)


