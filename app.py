"""
Flask REST API for Siamese Pill Classifier.

Usage:
    from app import create_app
    
    app = create_app(
        model_path="path/to/model.keras",
        cfg=CFG,
        best_threshold=0.45,
        load_and_preprocess_image=load_and_preprocess_image
    )
    app.run(host='0.0.0.0', port=5000)
"""

from flask import Flask, request, jsonify, g, current_app
from werkzeug.utils import secure_filename
import tempfile
import os
import numpy as np
from tensorflow import keras
from typing import Callable, Any


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    """Get model from request context, loading it on first access."""
    if 'model' not in g:
        g.model = keras.models.load_model(
            current_app.config['MODEL_PATH'],
            compile=False,
            safe_mode=False,
        )
    return g.model


def create_app(
    model_path: str,
    cfg: Any,
    best_threshold: float,
    load_and_preprocess_image: Callable,
) -> Flask:
    """
    Create and configure Flask application for pill similarity comparison.
    
    Args:
        model_path: Path to trained Siamese model (.keras file)
        cfg: Configuration object with image_size and channels attributes
        best_threshold: Optimal decision threshold from validation
        load_and_preprocess_image: Function to load and preprocess images
    
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
    # Store configuration in app context
    app.config['MODEL_PATH'] = model_path
    app.config['CFG'] = cfg
    app.config['BEST_THRESHOLD'] = best_threshold
    app.config['LOAD_AND_PREPROCESS_IMAGE'] = load_and_preprocess_image
    
    @app.route('/compare_pills', methods=['POST'])
    def compare_pills():
        """
        REST endpoint to compare two pill images for similarity.
        
        Expected request: multipart/form-data with:
            - image_a: Image file (PIL/JPEG/PNG)
            - image_b: Image file (PIL/JPEG/PNG)
            - threshold: (optional) Custom similarity threshold (default: best_threshold)
        
        Response: JSON with:
            - probability_same_class: float (0.0-1.0)
            - prediction: str ("same class" or "different class")
            - threshold: float used for decision
            - image_a_shape: tuple of processed image shape
            - image_b_shape: tuple of processed image shape
            - confidence: float confidence level
        """
        try:
            # Check that both images are provided
            if 'image_a' not in request.files or 'image_b' not in request.files:
                return jsonify({
                    "error": "Missing required files: 'image_a' and 'image_b'"
                }), 400
            
            file_a = request.files['image_a']
            file_b = request.files['image_b']
            
            if file_a.filename == '' or file_b.filename == '':
                return jsonify({"error": "File names cannot be empty"}), 400
            
            if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
                return jsonify({
                    "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
            
            # Get optional threshold parameter
            custom_threshold = request.form.get('threshold', None)
            if custom_threshold:
                try:
                    custom_threshold = float(custom_threshold)
                except ValueError:
                    return jsonify({"error": "Threshold must be a float"}), 400
            else:
                custom_threshold = current_app.config['BEST_THRESHOLD']
            
            # Save uploaded files to temporary locations
            with tempfile.TemporaryDirectory() as tmpdir:
                path_a = os.path.join(tmpdir, secure_filename(file_a.filename))
                path_b = os.path.join(tmpdir, secure_filename(file_b.filename))
                
                file_a.save(path_a)
                file_b.save(path_b)
                
                # Load and preprocess images
                try:
                    load_func = current_app.config['LOAD_AND_PREPROCESS_IMAGE']
                    cfg = current_app.config['CFG']
                    img_a = load_func(path_a, cfg.image_size, cfg.channels)
                    img_b = load_func(path_b, cfg.image_size, cfg.channels)
                except ValueError as e:
                    return jsonify({"error": f"Failed to load image: {str(e)}"}), 400
                
                # Prepare batch input
                batch_a = np.expand_dims(img_a, axis=0)
                batch_b = np.expand_dims(img_b, axis=0)
                
                # Run prediction
                model = get_model()
                prob_same = float(model.predict([batch_a, batch_b], verbose=0)[0, 0])
                pred_same = prob_same >= custom_threshold
                
                # Calculate confidence as distance from threshold
                confidence = abs(prob_same - 0.5) * 2
                
                response = {
                    "probability_same_class": round(prob_same, 4),
                    "prediction": "same class" if pred_same else "different class",
                    "threshold": round(custom_threshold, 4),
                    "confidence": round(confidence, 4),
                    "image_a_shape": list(img_a.shape),
                    "image_b_shape": list(img_b.shape),
                }
                
                return jsonify(response), 200
        
        except Exception as e:
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "service": "siamese-pill-classifier"}), 200
    
    return app


if __name__ == "__main__":
    print("Error: This module should be imported, not run directly.")
    print("Use create_app() to initialize the Flask application.")
