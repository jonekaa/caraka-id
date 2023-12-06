import cv2
import pytesseract
from matplotlib import pyplot as plt


def segment_words(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, binary_image = cv2.threshold(gray, 200, 300, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a list to store the bounding boxes of words
    word_boxes = []

    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small regions (adjust this threshold based on your needs)
        if w > 10 and h > 10:
            # Draw the bounding box on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Append the bounding box to the list
            word_boxes.append((x, y, w, h))

    # Visualize the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    return word_boxes, binary_image


# Example usage
image_path = "../data/cobacoba.jpg"
word_masks, image = segment_words(image_path)
# Now you have the bounding boxes of individual words
# You can crop these regions and use OCR to recognize the text within each box
for i, box in enumerate(word_masks):
    x, y, w, h = box
    word_image = image[y:y + h, x:x + w]

    # Perform OCR on the cropped word image
    word_text = pytesseract.image_to_string(word_image, lang='sundanese')
    print(f"Word {i + 1}: {word_text}")
