import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat, ImageDraw, ImageFont
import colorsys

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Material Detection Logic ---
def detect_materials_in_image(img_data: Image.Image):
    """
    Detects multiple waste materials present in the image using color segmentation
    and texture analysis. Simulates multi-object detection.
    """
    img = img_data.convert('RGB')
    width, height = img.size
    pixels = np.array(img)
    
    detected_materials = []
    material_regions = []
    
    # Define material detection profiles (color ranges and characteristics)
    material_profiles = {
        "Plastic Bottle": {
            "color_range": [(100, 150, 200), (200, 220, 255)],  # Blue/transparent
            "min_pixels": 500,
            "characteristics": "transparent, blue tint, smooth"
        },
        "Food Waste": {
            "color_range": [(50, 80, 20), (180, 220, 100)],  # Green/brown organic
            "min_pixels": 400,
            "characteristics": "organic colors, irregular texture"
        },
        "Paper/Cardboard": {
            "color_range": [(180, 160, 140), (255, 240, 220)],  # Beige/tan
            "min_pixels": 600,
            "characteristics": "fibrous, light colored, matte"
        },
        "Metal Can": {
            "color_range": [(150, 150, 150), (220, 220, 230)],  # Gray/silver
            "min_pixels": 300,
            "characteristics": "reflective, metallic sheen, cylindrical"
        },
        "Glass": {
            "color_range": [(200, 230, 230), (255, 255, 255)],  # Clear/white
            "min_pixels": 400,
            "characteristics": "transparent, reflective, smooth"
        },
        "Plastic Bag": {
            "color_range": [(200, 200, 200), (255, 255, 255)],  # White/transparent
            "min_pixels": 800,
            "characteristics": "thin, flexible, wrinkled texture"
        },
        "Food Container": {
            "color_range": [(220, 220, 220), (255, 255, 255)],  # White plastic
            "min_pixels": 500,
            "characteristics": "rigid plastic, white/clear"
        },
        "Aluminum Foil": {
            "color_range": [(180, 180, 180), (240, 240, 240)],  # Shiny silver
            "min_pixels": 300,
            "characteristics": "highly reflective, crinkled"
        },
        "Battery": {
            "color_range": [(20, 20, 20), (80, 80, 80)],  # Dark/black
            "min_pixels": 200,
            "characteristics": "cylindrical, dark, small"
        },
        "Electronic Device": {
            "color_range": [(10, 10, 10), (60, 60, 60)],  # Dark plastic/metal
            "min_pixels": 600,
            "characteristics": "dark, rigid, complex shape"
        }
    }
    
    # Analyze image for each material type
    for material_name, profile in material_profiles.items():
        color_min, color_max = profile["color_range"]
        
        # Create color mask
        mask = np.all((pixels >= color_min) & (pixels <= color_max), axis=2)
        pixel_count = np.sum(mask)
        
        if pixel_count >= profile["min_pixels"]:
            # Calculate percentage of image
            percentage = (pixel_count / (width * height)) * 100
            
            # Find bounding region
            coords = np.argwhere(mask)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Extract region for detailed analysis
                region = pixels[y_min:y_max+1, x_min:x_max+1]
                region_stats = {
                    "avg_color": region.mean(axis=(0, 1)).tolist(),
                    "brightness": region.mean(),
                    "variance": region.var()
                }
                
                detected_materials.append({
                    "material": material_name,
                    "confidence": min(0.95, 0.70 + (percentage / 100)),
                    "coverage": round(percentage, 2),
                    "characteristics": profile["characteristics"],
                    "region_stats": region_stats,
                    "position": {
                        "x": int((x_min + x_max) / 2),
                        "y": int((y_min + y_max) / 2),
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                    }
                })
                
                material_regions.append({
                    "name": material_name,
                    "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                })
    
    # If no materials detected by color, use general image analysis
    if not detected_materials:
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        brightness = (r + g + b) / 3
        variance = sum(stat.var) / 3
        
        # Fallback detection
        if brightness > 180:
            detected_materials.append({
                "material": "Light-colored Waste",
                "confidence": 0.65,
                "coverage": 100.0,
                "characteristics": "unidentified light material",
                "region_stats": {"avg_color": [r, g, b], "brightness": brightness, "variance": variance},
                "position": {"x": width // 2, "y": height // 2, "bbox": [0, 0, width, height]}
            })
        else:
            detected_materials.append({
                "material": "Dark-colored Waste",
                "confidence": 0.65,
                "coverage": 100.0,
                "characteristics": "unidentified dark material",
                "region_stats": {"avg_color": [r, g, b], "brightness": brightness, "variance": variance},
                "position": {"x": width // 2, "y": height // 2, "bbox": [0, 0, width, height]}
            })
    
    # Sort by coverage (most prominent first)
    detected_materials.sort(key=lambda x: x["coverage"], reverse=True)
    
    return detected_materials, material_regions


# --- Classification Logic for Each Material ---
def classify_waste_material(material_name, region_stats):
    """
    Classifies each detected material into waste categories with disposal instructions.
    """
    
    # Material to category mapping
    classification_map = {
        "Plastic Bottle": {
            "category": "Recyclable - Plastic",
            "subcategory": "PET/HDPE Bottle",
            "bin_color": "Blue Bin",
            "instructions": "Empty and rinse bottle. Remove cap. Crush to save space. Check for recycling symbol (#1 or #2).",
            "color": "blue",
            "recyclable": True,
            "hazardous": False
        },
        "Food Waste": {
            "category": "Compost/Organic",
            "subcategory": "Biodegradable Organic Matter",
            "bin_color": "Green/Brown Bin",
            "instructions": "Place in compost bin. Avoid meat, dairy, or oily foods if composting at home.",
            "color": "green",
            "recyclable": False,
            "hazardous": False
        },
        "Paper/Cardboard": {
            "category": "Recyclable - Paper",
            "subcategory": "Clean Paper/Cardboard",
            "bin_color": "Blue Bin",
            "instructions": "Flatten cardboard boxes. Remove plastic tape and labels. Keep dry and clean.",
            "color": "yellow",
            "recyclable": True,
            "hazardous": False
        },
        "Metal Can": {
            "category": "Recyclable - Metal",
            "subcategory": "Aluminum/Steel Can",
            "bin_color": "Blue Bin",
            "instructions": "Rinse thoroughly. Remove paper labels if possible. Crush to save space.",
            "color": "gray",
            "recyclable": True,
            "hazardous": False
        },
        "Glass": {
            "category": "Recyclable - Glass",
            "subcategory": "Glass Container",
            "bin_color": "Blue/Green Bin",
            "instructions": "Rinse thoroughly. Remove metal caps. Separate by color if required locally.",
            "color": "blue",
            "recyclable": True,
            "hazardous": False
        },
        "Plastic Bag": {
            "category": "Soft Plastic",
            "subcategory": "Film Plastic",
            "bin_color": "Special Collection",
            "instructions": "Take to grocery store collection point. DO NOT put in regular recycling bin.",
            "color": "yellow",
            "recyclable": True,
            "hazardous": False
        },
        "Food Container": {
            "category": "Recyclable - Plastic",
            "subcategory": "Food-grade Plastic",
            "bin_color": "Blue Bin",
            "instructions": "Wash thoroughly to remove food residue. Check recycling number (usually #5).",
            "color": "blue",
            "recyclable": True,
            "hazardous": False
        },
        "Aluminum Foil": {
            "category": "Recyclable - Metal",
            "subcategory": "Aluminum Foil",
            "bin_color": "Blue Bin",
            "instructions": "Clean off food residue. Ball up to golf-ball size before recycling.",
            "color": "gray",
            "recyclable": True,
            "hazardous": False
        },
        "Battery": {
            "category": "Hazardous Waste",
            "subcategory": "Electronic/Chemical Waste",
            "bin_color": "Special Disposal",
            "instructions": "NEVER throw in regular trash. Take to designated battery collection point or e-waste facility.",
            "color": "red",
            "recyclable": False,
            "hazardous": True
        },
        "Electronic Device": {
            "category": "E-Waste",
            "subcategory": "Electronic Equipment",
            "bin_color": "E-Waste Collection",
            "instructions": "Take to certified e-waste recycling center. Contains valuable and hazardous materials.",
            "color": "red",
            "recyclable": False,
            "hazardous": True
        }
    }
    
    # Get classification or use default
    classification = classification_map.get(material_name, {
        "category": "General Waste",
        "subcategory": "Unidentified",
        "bin_color": "Black Bin",
        "instructions": "Unable to classify. Dispose in general waste bin.",
        "color": "gray",
        "recyclable": False,
        "hazardous": False
    })
    
    return classification


# --- API Endpoint ---
@app.route("/api/predict", methods=["POST"])
def classify_waste_api():
    """
    Multi-material detection and classification endpoint.
    Returns detailed information about all detected materials.
    """
    
    if 'file' not in request.files:
        return jsonify({
            "error": "No file uploaded",
            "message": "Please upload an image file."
        }), 400
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return jsonify({
            "error": "Empty filename",
            "message": "No file selected."
        }), 400
    
    try:
        # Read and validate image
        img_bytes = uploaded_file.read()
        img_stream = io.BytesIO(img_bytes)
        img = Image.open(img_stream)
        img.verify()
        
        # Reopen for processing
        img_stream.seek(0)
        img = Image.open(img_stream)
        
        # Step 1: Detect all materials in the image
        detected_materials, material_regions = detect_materials_in_image(img)
        
        # Step 2: Classify each detected material
        classified_materials = []
        
        for material in detected_materials:
            classification = classify_waste_material(
                material["material"], 
                material["region_stats"]
            )
            
            classified_materials.append({
                "detected_material": material["material"],
                "confidence": round(material["confidence"], 2),
                "coverage_percentage": material["coverage"],
                "characteristics": material["characteristics"],
                "position": material["position"],
                "classification": classification
            })
        
        # Step 3: Generate overall summary
        total_recyclable = sum(1 for m in classified_materials if m["classification"]["recyclable"])
        total_hazardous = sum(1 for m in classified_materials if m["classification"]["hazardous"])
        
        primary_material = classified_materials[0] if classified_materials else None
        
        response = {
            "success": True,
            "total_materials_detected": len(classified_materials),
            "materials": classified_materials,
            "summary": {
                "recyclable_items": total_recyclable,
                "hazardous_items": total_hazardous,
                "general_waste_items": len(classified_materials) - total_recyclable - total_hazardous
            },
            "primary_classification": {
                "category": primary_material["classification"]["category"] if primary_material else "Unknown",
                "status": primary_material["classification"]["subcategory"] if primary_material else "No materials detected",
                "message": primary_material["classification"]["instructions"] if primary_material else "Upload a clearer image.",
                "color": primary_material["classification"]["color"] if primary_material else "gray",
                "bin_color": primary_material["classification"]["bin_color"] if primary_material else "N/A"
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Processing failed",
            "message": f"Image processing error: {str(e)}"
        }), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Multi-Material Waste Classifier API is running",
        "version": "2.0"
    }), 200


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Multi-Material AI Waste Classifier Backend")
    print("=" * 60)
    print("📡 Server: http://127.0.0.1:5000")
    print("🔗 Endpoint: POST /classify_waste")
    print("💡 Features: Multi-object detection + Classification")
    print("=" * 60)

    app.run(debug=True, port=5000, host='127.0.0.1')

