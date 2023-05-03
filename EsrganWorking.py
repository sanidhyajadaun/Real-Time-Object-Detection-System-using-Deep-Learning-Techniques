import cv2
import matplotlib.pyplot as plt
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer
import time

def plot_images(img1, img2):

    # Convert BGR images to RGB for Matplotlib
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Display the first image in the left subplot
    ax[0].imshow(img1)
    ax[0].set_title('Original Image')
    
    # Display the second image in the right subplot
    ax[1].imshow(img2)
    ax[1].set_title('ESRGAN upscaled Image')
    # Show the plot
    plt.show()


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4

model_path = 'weights\RealESRGAN_x4plus.pth'

# restorer
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
)

img = cv2.imread('inputs/0014.jpg')
# Record the start time
start_time = time.time()
output, _ = upsampler.enhance(img, outscale=3)
# Record the end time
end_time = time.time()
# Compute the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time in seconds
print(f"Elapsed time: {elapsed_time:.2f} seconds")
cv2.imwrite("realesrganimage.jpg",output)
plot_images(img,output)


# cv2.imshow("resultant",output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
