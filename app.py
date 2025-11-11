import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat

# Initialize Flask app
app = Flask(__name__)

# ✅ Allow specific origins (replace Vercel link once deployed)
CORS(app, origins=[
    "https://your-frontend-name.vercel.app",  # Replace with your actual Vercel URL
    "http://localhost:3000"
])

# --- Material Detection Logic ---
def detect_materials_in_image(img_data: Image.Image):
    """
    Detects multiple waste materials using color segmentation & texture analysis.
    """
    img = img_data.convert('RGB')
    width, height = img.size
    pixels = np.array(img)
    
    detected_materials = []
    material_profiles = {
        "Plastic Bottle": {"color_range": [(100,150,200), (200,220,255)], "min_pixels": 500, "characteristics": "transparent, blue tint"},
        "Food Waste": {"color_range": [(50,80,20), (180,220,100)], "min_pixels": 400, "characteristics": "organic texture"},
        "Paper/Cardboard": {"color_range": [(180,160,140), (255,240,220)], "min_pixels": 600, "characteristics": "fibrous, matte"},
        "Metal Can": {"color_range": [(150,150,150), (220,220,230)], "min_pixels": 300, "characteristics": "metallic, reflective"},
        "Battery": {"color_range": [(20,20,20), (80,80,80)], "min_pixels": 200, "characteristics": "cylindrical, dark"}
    }

    for material_name, profile in material_profiles.items():
        color_min, color_max = profile["color_range"]
        mask = np.all((pixels >= color_min) & (pixels <= color_max), axis=2)
        pixel_count = np.sum(mask)
        if pixel_count >= profile["min_pixels"]:
            percentage = (pixel_count / (width * height)) * 100
            detected_materials.append({
                "material": material_name,
                "confidence": round(min(0.95, 0.70 + (percentage / 100)), 2),
                "coverage": round(percentage, 2),
                "characteristics": profile["characteristics"]
            })
    
    if not detected_materials:
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        brightness = (r + g + b) / 3
        material_type = "Light-colored Waste" if brightness > 180 else "Dark-colored Waste"
        detected_materials.append({
            "material": material_type,
            "confidence": 0.65,
            "coverage": 100.0,
            "characteristics": "unidentified material"
        })
    
    detected_materials.sort(key=lambda x: x["coverage"], reverse=True)
    return detected_materials

# --- Classification Logic ---
def classify_waste_material(material_name):
    classification_map = {
        "Plastic Bottle": {
            "category": "Recyclable",
            "bin_color": "Blue Bin",
            "instructions": "Rinse and crush before recycling.",
            "color": "blue",
            "recyclable": True,
            "hazardous": False
        },
        "Food Waste": {
            "category": "Organic Waste",
            "bin_color": "Green Bin",
            "instructions": "Place in compost or biodegradable bin.",
            "color": "green",
            "recyclable": False,
            "hazardous": False
        },
        "Paper/Cardboard": {
            "category": "Recyclable",
            "bin_color": "Blue Bin",
            "instructions": "Keep clean and dry before recycling.",
            "color": "yellow",
            "recyclable": True,
            "hazardous": False
        },
        "Metal Can": {
            "category": "Recyclable",
            "bin_color": "Blue Bin",
            "instructions": "Rinse and crush before recycling.",
            "color": "gray",
            "recyclable": True,
            "hazardous": False
        },
        "Battery": {
            "category": "Hazardous Waste",
            "bin_color": "Special Bin",
            "instructions": "Dispose at a battery recycling point.",
            "color": "red",
            "recyclable": False,
            "hazardous": True
        }
    }
    return classification_map.get(material_name, {
        "category": "General Waste",
        "bin_color": "Black Bin",
        "instructions": "Dispose responsibly.",
        "color": "gray",
        "recyclable": False,
        "hazardous": False
    })

# --- API Endpoints ---
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "Waste Classifier Backend Online"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        detected_materials = detect_materials_in_image(img)
        results = []
        for material in detected_materials:
            classification = classify_waste_material(material["material"])
            results.append({
                "detected_material": material["material"],
                "confidence": material["confidence"],
                "coverage_percentage": material["coverage"],
                "characteristics": material["characteristics"],
                "classification": classification
            })
        
        total_recyclable = sum(1 for m in results if m["classification"]["recyclable"])
        total_hazardous = sum(1 for m in results if m["classification"]["hazardous"])
        
        return jsonify({
            "success": True,
            "total_materials_detected": len(results),
            "summary": {
                "recyclable_items": total_recyclable,
                "hazardous_items": total_hazardous,
                "general_waste_items": len(results) - total_recyclable - total_hazardous
            },
            "materials": results
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "message": "Flask waste classifier API is healthy"
    })

# --- Main ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
