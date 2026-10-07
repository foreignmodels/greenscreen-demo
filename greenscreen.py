#Code follows this tutorial: https://www.youtube.com/watch?v=l-llIQh0DsI&t=11s and https://docs.rs/crate/oximedia-effects/0.1.7/source/src/video/chromakey.rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path(__file__).resolve().parent
background_path = project_dir / "background.jpg"
greenscreen_path = project_dir / "greenscreen.jpg"


# Read the background image
background = cv2.imread(str(background_path))
if background is None:
    raise FileNotFoundError(f"Could not read background image: {background_path}")

background = cv2.resize(background, (640, 480))

# Read the greenscreen image/video
capture = cv2.VideoCapture(str(greenscreen_path))
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

if not capture.isOpened():
    raise OSError(f"Could not open green-screen input: {greenscreen_path}")

while True:
    ret, frame = capture.read()
    if not ret or frame is None:
        break
    frame = cv2.resize(frame, (640, 480))

    #Green color explicit threshold
    keyColor = (0, 255, 0) 
    T1, T2 = 50, 100

    image = frame.astype(np.float32)
    # Calculate the Euclidean distance from the key color
    d = np.sqrt(np.sum((image - keyColor) ** 2, axis=-1))
    # Create a transparency mask based on the distance
    alpha = np.clip((d - T1) / (T2 - T1), 0, 1)
    # Blend the original image with the background using the transparency mask
    a = alpha.reshape(480, 640, 1)
    result = (1 - a) * image + a * background.astype(np.float32)
    mask = (alpha * 255).astype(np.uint8)

    #Caps green channel to avoid green spill. This section lines 52-56 were generated with help from ClaudeCode
    blue_cap, green_cap, red_cap = cv2.split(image)
    # Cap the green channel to be no more than 10 above the maximum of the blue and red channels
    green_cap = np.minimum(green_cap, np.maximum(blue_cap, red_cap) + 10)
    final_cap = cv2.merge([blue_cap, green_cap, red_cap])

    #Type is np.unit8 because OpenCV uses this type for images
    result = np.clip(a * final_cap + (1 - a) * background.astype(np.float32),0,255,).astype(np.uint8)
    mask_inv = (255 - mask).astype(np.uint8)
    cutout = (a * final_cap).astype(np.uint8)
    background_cutout = ((1 - a) * background.astype(np.float32)).astype(np.uint8)
    
    plt.figure(Path(__file__).name, figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(mask_inv, cmap='gray')
    plt.title('Mask Inverse')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
    plt.title('Cutout')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(background_cutout, cv2.COLOR_BGR2RGB))
    plt.title('Background Cutout')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Result')
    plt.axis('off')

    #comment out the following lines to disable the matplotlib display
    plt.show()
    
    #Uncomment the following lines to display a video
    #cv2.imshow('Video', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

