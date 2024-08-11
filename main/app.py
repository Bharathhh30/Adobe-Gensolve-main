from flask import Flask, render_template, redirect, url_for, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from werkzeug.utils import secure_filename
from pyngrok import ngrok


# ngrok.set_auth_token("2iXWagsnReYrk3koiGdh5L4ccPC_3a1QdhdzMBEvSGH8aQvD1")  # Replace with your ngrok authentication token

# Start ngrok with the named tunnel configuration
# ngrok_tunnel = ngrok.connect("my_tunnel")  # Assuming ngrok.yml is in the default location

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/outputkeliye', methods=['POST', 'GET'])
def outputkeliye():
    if request.method == 'POST':
        file = request.files['input-image']
        filename = secure_filename(file.filename)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to open image file '{img_path}'. Please check the file path and try again.")
            return "Error processing image"
        
        img = cv2.resize(img, (128, 128))

        def preprocess_img(img):
            img = cv2.equalizeHist(img)
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            return thresh
        
        def complete_shape(img):
            kernel = np.ones((5, 5), np.uint8)
            closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            closed_img = cv2.bitwise_not(closed_img)

            contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(img)
            for contour in contours:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mask = cv2.dilate(mask, kernel, iterations=2)
            return mask
        
        def detect_shape(img):
            edges = preprocess_img(img)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                print("No contours found.")
                return "unknown"

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            num_vertices = len(approx)
            print(f"Number of vertices detected: {num_vertices}")

            if num_vertices == 3:
                return "triangle"
            elif num_vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    return "square"
                else:
                    return "rectangle"
            elif num_vertices == 5:
                return "pentagon"
            elif num_vertices >= 6 and num_vertices <= 8:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if circularity > 0.7:
                    return "circle"
                else:
                    return "ellipse"
            else:
                return "star"
        
        def generate_shape_img_with_outline(shape, img_size=(128, 128)):
            img = np.ones(img_size, dtype=np.uint8) * 255
            if shape == "circle":
                center = (img_size[0] // 2, img_size[1] // 2)
                radius = min(img_size) // 3
                cv2.circle(img, center, radius, 255, -1)
                cv2.circle(img, center, radius, 0, 2)
            elif shape == "square":
                side = int(min(img_size) // 1.5)
                start = int((img_size[0] - side) // 2)
                end = int(start + side)
                cv2.rectangle(img, (start, start), (end, end), 255, -1)
                cv2.rectangle(img, (start, start), (end, end), 0, 2)
            elif shape == "rectangle":
                width = int(img_size[0] // 1.5)
                height = int(img_size[1] // 2)
                start_x = int((img_size[0] - width) // 2)
                start_y = int((img_size[1] - height) // 2)
                cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), 255, -1)
                cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), 0, 2)
            elif shape == "star":
                points = np.array([[64, 10], [74, 40], [104, 40], [80, 60], [90, 90], [64, 70], [38, 90], [48, 60], [24, 40], [54, 40]], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.fillPoly(img, [points], 255)
                cv2.polylines(img, [points], True, 0, 2)
            elif shape == "pentagon":
                points = np.array([[64, 10], [104, 40], [84, 90], [44, 90], [24, 40]], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.fillPoly(img, [points], 255)
                cv2.polylines(img, [points], True, 0, 2)
            elif shape == "ellipse":
                center = (img_size[0] // 2, img_size[1] // 2)
                axes = (min(img_size) // 3, min(img_size) // 4)
                cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)
                cv2.ellipse(img, center, axes, 0, 0, 360, 0, 2)

            return img
        
        def detect_symmetry(img, detected_shape):
            num_sides = {"triangle": 3, "square": 4, "rectangle": 4, "pentagon": 5, "circle": 360, "ellipse": 360, "star": 5}.get(detected_shape, 1)
            symmetries = 0
            img_center = (img.shape[1] // 2, img.shape[0] // 2)

            original_img = img.copy()

            for i in range(num_sides):
                angle = 360 / num_sides * i
                rot_matrix = cv2.getRotationMatrix2D(img_center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

                score, _ = ssim(original_img, rotated_img, full=True)
                if score > 0.75:
                    symmetries += 1

            return symmetries
        
        def highlight_symmetry_lines(img, detected_shape):
            img_with_lines = img.copy()
            img_center = (img.shape[1] // 2, img.shape[0] // 2)

            num_sides = {"triangle": 3, "square": 4, "rectangle": 4, "pentagon": 5, "circle": 360, "ellipse": 360, "star": 5}.get(detected_shape, 1)

            for i in range(num_sides):
                angle = 360 / num_sides * i
                end_x = int(img_center[0] + img_center[0] * np.cos(np.radians(angle)))
                end_y = int(img_center[1] - img_center[1] * np.sin(np.radians(angle)))
                cv2.line(img_with_lines, img_center, (end_x, end_y), 0, 2)

            return img_with_lines
        
        detected_shape = detect_shape(img)
        print(f"Detected Shape: {detected_shape}")

        completed_img = complete_shape(img)
        output_img = generate_shape_img_with_outline(detected_shape)
        num_symmetries = detect_symmetry(output_img, detected_shape)
        print(f"Number of symmetries: {num_symmetries}")
        symmetry_img = highlight_symmetry_lines(output_img, detected_shape)

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.title('Input Image')
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Completed Image')
        plt.imshow(completed_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title('Detected Symmetries')
        plt.imshow(symmetry_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Generated Shape')
        plt.imshow(output_img, cmap='gray')
        plt.axis('off')

        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'output_plot.png'))
        
        return redirect(url_for('show_results', img_name='output_plot.png'))

@app.route('/show-results/<img_name>')
def show_results(img_name):
    img_url = url_for('static', filename='uploads/' + img_name)
    return render_template('output.html', img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
