import torch
import numpy as np
from PIL import Image

# Initialize an empty list to store images
images = []

# Loop through the range of image files
for i in range(18):  # 0.png to 17.png
    # Load the image
    img = Image.open(f'test_images/{i}.png')
    img = img.resize((224, 224))
    # Convert the image to a numpy array with dtype uint8 (range 0-255)
    img_array = np.array(img, dtype=np.uint8)

    # Append to the images list
    images.append(img_array)

# Convert the list of images into a PyTorch tensor
images_tensor = torch.tensor(images, dtype=torch.uint8)
torch.save(images_tensor, 'images_tensor.pth')
# Print the shape of the tensor
print(images_tensor.shape)