from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from engine import SmartOverrideEngine

app = Flask(__name__, template_folder='.')
override_engine = SmartOverrideEngine(threshold=0.35)

def decode_image_from_request(req):
    """Helper function to convert the incoming camera image bytes into an OpenCV frame."""
    file = req.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    """Serves your dex.html page when you go to http://localhost:5000"""
    return render_template('dex.html')

@app.route('/api/scan', methods=['POST'])
def scan_frame():
    """Regular scanning endpoint used by your live camera loop."""
    try:
        frame = decode_image_from_request(request)
        
        # Replace "Unknown Object" with your base model's default classification output
        raw_model_output = "Unknown Object" 
        
        # Run it through the override checker
        final_result = override_engine.classify(frame, raw_model_output)
        return jsonify(final_result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/override', methods=['POST'])
def force_override():
    """Triggered strictly when you click the 'Save Override' button in Manual Mode."""
    try:
        frame = decode_image_from_request(request)
        custom_label = request.form.get('label').strip().lower()
        
        if not custom_label:
            return jsonify({"error": "Label cannot be empty"}), 400
            
        message = override_engine.learn_override(frame, custom_label)
        return jsonify({"status": "success", "message": message}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Starts your server application
    app.run(host='0.0.0.0', port=5000, debug=True)
