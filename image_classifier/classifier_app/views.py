import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
import io
import os
import uuid
import json
import base64

# Load Pretrained Model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Create feature extraction models for visualization
feature_models = {}

def create_feature_models():
    """Create models to extract features from different layers of MobileNetV2"""
    global feature_models
    
    # Extract features from different layers
    layer_names = [
        'Conv1',  # First conv layer
        'block_1_depthwise',  # Early depthwise conv
        'block_3_expand',  # Middle expansion layer
        'block_6_project',  # Later projection layer
        'block_13_expand',  # Deep features before classification
    ]
    
    for layer_name in layer_names:
        feature_models[layer_name] = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )

# Call this function at module load time
create_feature_models()

def get_feature_maps(img_array, layer_name):
    """Extract and process feature maps for visualization"""
    features = feature_models[layer_name].predict(img_array)
    
    # For convolutional layers, extract a subset of channels
    features = features[0]  # First image in batch
    
    # Normalize features for visualization (0-255)
    features = (features - features.min()) / (features.max() - features.min() + 1e-7) * 255
    features = features.astype(np.uint8)
    
    # Select representative channels (max 16)
    step = max(1, features.shape[-1] // 16)
    selected_channels = features[:, :, ::step][:, :, :16]
    
    return selected_channels

def convert_features_to_image(features, rows=4, cols=4):
    """Convert feature maps to a displayable grid image"""
    # Get dimensions
    n_features = min(rows * cols, features.shape[-1])
    height, width = features.shape[0], features.shape[1]
    
    # Create an empty grid
    grid = np.zeros((rows * height, cols * width), dtype=np.uint8)
    
    # Fill the grid with feature maps
    for i in range(min(n_features, rows * cols)):
        row = i // cols
        col = i % cols
        channel_data = features[:, :, i]
        grid[row * height:(row + 1) * height, col * width:(col + 1) * width] = channel_data
    
    # Convert to RGB for display
    grid_rgb = np.stack([grid, grid, grid], axis=2)
    
    # Convert to image and then to base64
    img = Image.fromarray(grid_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_str

def classify_image(request):
    print("🔹 Received Request:", request.method)
    context = {}
    if request.method == "POST":
        print("✅ POST request received!")
        if "image" not in request.FILES:
            print("⚠️ No image uploaded!")
            context["error"] = "No file uploaded!"
            return render(request, "index.html", context)
        try:
            # Get the uploaded file
            uploaded_file = request.FILES["image"]
            
            # Save the file temporarily with a unique name
            temp_file_name = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = default_storage.save(temp_file_name, ContentFile(uploaded_file.read()))
            
            # Get the full path to the saved file
            full_path = default_storage.path(temp_path)
            
            # Open the saved file with error handling for various formats
            try:
                img = Image.open(full_path)
                img = img.convert("RGB")  # Ensure it's in RGB mode
                img = img.resize((224, 224))  # Resize to match model input
                
                # Save as optimized JPEG to handle various formats
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                buffer.seek(0)
                
                # Save the converted image
                converted_path = default_storage.save(f"conv_{temp_file_name}", ContentFile(buffer.getvalue()))
                full_path = default_storage.path(converted_path)
                
                # Clean up original file
                default_storage.delete(temp_path)
                temp_path = converted_path
                
                # Load the image as numpy array
                img = Image.open(full_path)
                img_array = np.array(img)
            except Exception as img_error:
                raise Exception(f"Unsupported image format or corrupt file: {str(img_error)}")
            
            # Preprocess for model
            img_array_batch = np.expand_dims(img_array, axis=0)
            preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_batch)
            
            # Extract feature maps for visualization
            visualization_data = {}
            
            # Original image (resized)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            visualization_data["input_image"] = img_str
            
            # Get feature maps from different layers
            layer_mapping = {
                "conv1": "Conv1",
                "pool1": "block_1_depthwise",
                "conv2": "block_3_expand",
                "features": "block_6_project",
                "deep_features": "block_13_expand"
            }
            
            for viz_name, layer_name in layer_mapping.items():
                features = get_feature_maps(preprocessed_img, layer_name)
                feature_img = convert_features_to_image(features)
                visualization_data[viz_name] = feature_img
            
            # Predict
            preds = model.predict(preprocessed_img)
            decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
            predictions = [{"class": p[1], "confidence": float(p[2] * 100)} for p in decoded_preds]
            print("🎯 Predictions:", predictions)
            
            # Add predictions to visualization data
            visualization_data["predictions"] = predictions
            
            # Add class activation mapping (approximated for simplicity)
            # For a more accurate CAM, additional code would be needed
            
            # Create a URL to the saved file
            file_url = default_storage.url(temp_path)
            
            context["predictions"] = predictions
            context["file_url"] = file_url
            context["temp_path"] = temp_path  # Store to delete later if needed
            context["visualization_data"] = json.dumps(visualization_data)
            
        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            context["error"] = f"Error processing image: {str(e)}. Please upload a valid image file."
            
            # Clean up temp file if it exists and there was an error
            if 'temp_path' in context:
                try:
                    default_storage.delete(context['temp_path'])
                except:
                    pass
    return render(request, "index.html", context)