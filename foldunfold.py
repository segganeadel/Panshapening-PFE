import cv2 as cv
import numpy as np

def image_to_patches(image, patch_size, stride=1):
  """
  This function splits an image into fixed-size patches.

  Args:
      image: A numpy array representing the image with dimensions (height, width, channels).
      patch_size: A tuple representing the height and width of the patch.
      stride: The number of pixels to move between patches (default: 1).

  Returns:
      A numpy array of shape (num_patches, patch_height, patch_width, channels) containing the patches.
  """
  image_height, image_width, channels = image.shape
  patch_height, patch_width = patch_size

  # Validate input
  if image_height < patch_height or image_width < patch_width:
    raise ValueError("Image dimensions are smaller than patch size.")

  # Calculate the number of patches in each dimension
  num_patches_y = int((image_height - patch_height) / stride + 1)
  num_patches_x = int((image_width - patch_width) / stride + 1)

  num_total_patches = num_patches_y * num_patches_x
  # Create an empty array to store the patches
  patches = np.zeros((num_total_patches, patch_height, patch_width, channels), dtype=image.dtype)

  # Iterate over the image and extract patches
  for y in range(num_patches_y):
    for x in range(num_patches_x):
      y_start = y * stride
      y_end = y_start + patch_height
      x_start = x * stride
      x_end = x_start + patch_width
      patch = image[y_start:y_end, x_start:x_end, :]
      patches[y * num_patches_y + x, :, :, :] = patch

  return patches

def unpatch_image(patches, image_shape, stride=1):
  """
  This function reconstructs an image from a collection of patches.

  Args:
      patches: A numpy array of shape (num_patches, patch_height, patch_width, channels) containing the patches.
      image_shape: A tuple representing the original image dimensions (height, width, channels).
      stride: The number of pixels used during patching (default: 1).

  Returns:
      A numpy array representing the reconstructed image with the original dimensions.
  """
  num_total_patches, patch_height, patch_width, channels = patches.shape
  image_height, image_width, _ = image_shape

  # Validate input based on patch count and image dimensions
  expected_patches_y = int((image_height - patch_height) / stride + 1)
  expected_patches_x = int((image_width - patch_width) / stride + 1)
  if num_total_patches != expected_patches_y * expected_patches_x:
    raise ValueError("Patch count does not match expected image dimensions.")

  # Create an empty array to store the reconstructed image
  image = np.zeros(image_shape, dtype= np.float32)

  # Overlap handling using weights (optional for better reconstruction)
  weights = np.ones((patch_height, patch_width), dtype=np.float32)
  center_y = int(patch_height / 2)
  center_x = int(patch_width / 2)
  weights[center_y:, center_x:] = 0.5  # Reduce weight for overlapping areas
  weights[:center_y, center_x:] = 0.5
  weights[:center_y, :center_x] = 0.5
  weights[center_y:, :center_x] = 0.5
  weights = np.stack([weights] * channels, axis=2)
  print(weights)
  # Iterate over the patches and place them in the reconstructed image
  for y in range(expected_patches_y):
    for x in range(expected_patches_x):
      y_start = y * stride
      y_end = y_start + patch_height
      x_start = x * stride
      x_end = x_start + patch_width
      index = y * expected_patches_y + x
      image[y_start:y_end, x_start:x_end, :] += patches[index, :, :, :] * weights

  # Uncomment these lines for averaging overlapping areas (alternative to weights)
#   image = image / np.maximum(np.sum(weights, axis=2, keepdims=True), 1)
  image = np.clip(image, 0, 255)
  return image.astype(np.uint8)


image = cv.imread("./output/mambfuse/image_out_0.png")
print(image.shape)

kernel_size = (32,32)
stride = 16

patches = image_to_patches(image, kernel_size, stride)
print(patches.shape)
print(patches[0].max(), patches[0].min())

reconstructed = unpatch_image(patches, image.shape, stride)
print(reconstructed.shape)

cv.imwrite("./reconstructed.png", reconstructed)
