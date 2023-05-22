from skimage import io, metrics

# Load the two images
image1 = io.imread('demo.png', as_gray=True)
image2 = io.imread('finalRef.png', as_gray=True)

# Calculate SSIM
ssim_value = metrics.structural_similarity(image1, image2, data_range=image2.max() - image2.min())

print(f"SSIM value: {ssim_value}")