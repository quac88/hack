from flask import Flask, request, jsonify
from run import add_gaussian_noise_and_save

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    # receive the base64 encoded image
    image_base64 = request.json['image']

    # process the image
    processed_image_base64 = add_gaussian_noise_and_save(image_base64)

    # Return the processed image as base64 string
    return jsonify({'image': processed_image_base64})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=37015)  # Listen on port 8888 for HTTP traffic
