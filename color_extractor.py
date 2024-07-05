import streamlit as st
import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import webcolors

st.title("Image Deconstruction for Water Colors")
num_colors = st.slider("Select the number of dominant colors:", min_value=1, max_value=10, value=10)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def to_sketch(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3),np.uint8)
    sketch = cv2.dilate(edges, kernel, iterations = 1)
    inverted_edges = cv2.bitwise_not(sketch)  # Invert the sketch
    return inverted_edges

def get_complementary(color):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    # Add 180 to Hue channel
    hsv[0] = (hsv[0] + 180) % 180
    # Convert HSV back to RGB
    complementary = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]
    return complementary

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    adjusted = cv2.convertScaleAbs(image, alpha=1+contrast/127.5, beta=brightness-127.5)
    return adjusted



def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_dominant_colors(image, num_colors):
    # Perform k-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    # Convert the colors to integers
    colors = colors.round(0).astype(int)
    colors = sorted(colors, key=lambda rgb : 0.2989*rgb[0] + 0.5870*rgb[1] + 0.1140*rgb[2])

    return colors

def highlight_color(image, color, threshold=30):
    # Calculate the Euclidean distance between the color of each pixel and the target color
    distances = np.linalg.norm(image - color, axis=2)
    # Create a mask for the pixels whose color is close to the target color
    mask = distances < threshold
    # Create a copy of the image
    highlighted = image.copy()
    # Change the color of the pixels that are not close to the target color
    highlighted[~mask] = [255, 255, 255]  # white
    return highlighted


if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    color_greyscale_option = st.selectbox(
        'Choose Full Colour or Monochrome',
        ('Full Color', 'Monochrome'))
    if color_greyscale_option == 'Full Color':
        color_greyscale_choice = 1
    elif color_greyscale_option == 'Monochrome':
        color_greyscale_choice = 0
    image = cv2.imdecode(file_bytes, color_greyscale_choice)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Get 10 most dominant colors
    colors = get_dominant_colors(pixels, num_colors)

    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)




    sketch = to_sketch(image)
    st.image(sketch, caption='Sketch.', use_column_width=True)
    #complementary_colors = [get_complementary(color) for color in colors]
    #st.image([complementary_colors], caption='Complementary Colors.', use_column_width=True)
    #brightness = st.slider("Brightness:", min_value=-127, max_value=127, value=0)
    #contrast = st.slider("Contrast:", min_value=-127, max_value=127, value=0)
    #adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
    #st.image(adjusted_image, caption='Adjusted Image.', use_column_width=True)







# Display two color palettes
    fig, axs = plt.subplots(2, 1, figsize=(5, 2), dpi=100)
    for i in range(2):
        axs[i].axis("off")
        axs[i].imshow([[color.astype(int) for color in colors[i*num_colors//2:(i+1)*num_colors//2]]], aspect='auto')
    st.pyplot(fig)

    # Create three columns
col1, col2, col3 = st.columns(3)

subplot_height = 50  # You can adjust this as needed
cumulative_colors = []

for i in range(num_colors-1, -1, -1):  # Start from the last color and move to the first
    cumulative_colors.append(colors[i])
    cumulative_image = np.ones_like(image) * 255
    for color in cumulative_colors:
        highlighted = highlight_color(image, color)
        cumulative_image = np.where(highlighted != [255, 255, 255], highlighted, cumulative_image)

    # Display progression image
    col2.image(cumulative_image, caption=f'Progression {num_colors-i}.', use_column_width=True)

    highlighted_height, highlighted_width = highlighted.shape[:2]

    # Create and display color swatch for the current color
    swatch = np.full((100, 100, 3), colors[i], dtype=np.uint8)  # Creates an image filled with the current color
    swatch = cv2.resize(swatch, (highlighted_width, highlighted_height))
    col3.image(swatch, caption=f'Color {num_colors-i}', use_column_width=True)

    # Replace the previously highlighted colors with the current highlighted color for display in col1
    display_image = np.where(highlighted != [255, 255, 255], highlighted, np.ones_like(image) * 255)
    col1.image(display_image, caption=f'Highlight Color {num_colors-i}', use_column_width=True)




