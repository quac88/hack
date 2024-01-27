from flask import Flask, request, jsonify
from run import add_gaussian_noise_and_save_dynamic, add_gaussian_noise_and_save_static

app = Flask(__name__)

@app.route('/api/processImageDynamic', methods=['POST'])
def process_image():
    # receive the base64 encoded image
    image_base64 = request.json['image']

    # process the image
    processed_image_base64 = add_gaussian_noise_and_save_dynamic(image_base64)

    # Return the processed image as base64 string
    return jsonify({'image': processed_image_base64})

@app.route('/api/processImageStatic', methods=['POST'])
def process_image_static():
    # receive the base64 encoded image
    image_base64 = request.json['image']

    # process the image with fixed sigma
    processed_image_base64 = add_gaussian_noise_and_save_static(image_base64)

    # Return the processed image as base64 string
    return jsonify({'image': processed_image_base64})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000, threaded=True)  # Listen on port 3000 for HTTP traffic


