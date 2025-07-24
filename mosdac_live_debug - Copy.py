from selenium import webdriver
import time
import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify, send_file

# -----------------------------
# Configuration
# -----------------------------
Tb_min_k = 190
Tb_max_k = 290
Tb_threshold = 240  # IRBT threshold in Kelvin

# -----------------------------
# 1. Capture Image from MOSDAC
# -----------------------------
def capture_mosdac_image(output_dir="screenshots"):
    os.makedirs(output_dir, exist_ok=True)

    driver = webdriver.Chrome()  # Open normal browser (not headless for debugging)

    try:
        driver.get("https://mosdac.gov.in/gallery/index.html?&prod=3RIMG_%27*_L1C_SGP_3D_IR1_V%27*.jpg&date=2025-07-07&count=60")
        time.sleep(20)  # Wait for page to load

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"mosdac_live_debug_{timestamp}.png")
        driver.save_screenshot(filename)
        print(f"[✅] Screenshot saved: {filename}")
        return filename

    finally:
        driver.quit()

# -----------------------------
# 2. Analyze Image for TCC
# -----------------------------
def analyze_tcc(image_path):
    print(f"\n[📊] Analyzing image: {image_path}")
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Could not read image or image is invalid.")

    # Map pixel to brightness temperature
    Tb = Tb_min_k + (img_gray / 255.0) * (Tb_max_k - Tb_min_k)
    mask = Tb < Tb_threshold
    tcc_pixels = Tb[mask]

    if tcc_pixels.size == 0:
        print("❌ No TCC pixels found below threshold.")
        return

    # TCC Metrics
    mean_tb = np.mean(tcc_pixels)
    min_tb = np.min(tcc_pixels)
    max_tb = np.max(tcc_pixels)
    median_tb = np.median(tcc_pixels)
    std_tb = np.std(tcc_pixels)
    pixel_count = np.sum(mask)

    # Contour detection
    _, binary_mask = cv2.threshold(img_gray, ((Tb_threshold - Tb_min_k)/(Tb_max_k - Tb_min_k)) * 255, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ No contours found.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # Radius metrics
    distances = [np.linalg.norm(np.array([cx, cy]) - pt[0]) for pt in largest_contour]
    min_radius = np.min(distances)
    max_radius = np.max(distances)
    mean_radius = np.mean(distances)

    def tb_to_height_km(tb_k):
        return (290 - tb_k) * 0.15  # lapse rate approx. 6.5 K/km

    max_height = tb_to_height_km(min_tb)
    mean_height = tb_to_height_km(mean_tb)

    # Display results
    print("🌩️ TCC Metrics:")
    print(f"Pixel Count: {pixel_count}")
    print(f"Mean Tb: {mean_tb:.2f} K")
    print(f"Min Tb: {min_tb:.2f} K")
    print(f"Max Tb: {max_tb:.2f} K")
    print(f"Median Tb: {median_tb:.2f} K")
    print(f"Std Dev Tb: {std_tb:.2f} K")
    print(f"Convective Center: (x={cx}, y={cy})")
    print(f"Min Radius: {min_radius:.2f} px")
    print(f"Max Radius: {max_radius:.2f} px")
    print(f"Mean Radius: {mean_radius:.2f} px")
    print(f"Max Cloud-Top Height: {max_height:.2f} km")
    print(f"Mean Cloud-Top Height: {mean_height:.2f} km")

    # Visual
    plt.figure(figsize=(8, 6))
    plt.imshow(Tb, cmap="inferno")
    plt.colorbar(label="Brightness Temperature (K)")
    plt.scatter(cx, cy, c="cyan", s=50, label="Convective Center")
    plt.title("Tropical Cloud Cluster Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return f"Pixel Count: {pixel_count}\nMean Tb: {mean_tb:.2f} K\nMin Tb: {min_tb:.2f} K\nMax Tb: {max_tb:.2f} K\nMedian Tb: {median_tb:.2f} K\nStd Dev Tb: {std_tb:.2f} K\nConvective Center: (x={cx}, y={cy})\nMin Radius: {min_radius:.2f} px\nMax Radius: {max_radius:.2f} px\nMean Radius: {mean_radius:.2f} px\nMax Cloud-Top Height: {max_height:.2f} km\nMean Cloud-Top Height: {mean_height:.2f} km"

app = Flask(__name__)

# HTML template with Google Maps integration
HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>MOSDAC TCC Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        #container { max-width: 480px; margin: auto; padding: 10px; }
        #map { height: 300px; width: 100%; border-radius: 8px; }
        #controls { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
        #detectBtn, #locateBtn { padding: 14px; font-size: 1.1em; border-radius: 6px; border: none; background: #1976d2; color: #fff; cursor: pointer; }
        #detectBtn:active, #locateBtn:active { background: #1565c0; }
        #result { margin-top: 20px; word-break: break-word; }
        img { max-width: 100%; border-radius: 8px; }
        @media (max-width: 600px) {
            #container { padding: 2px; }
            #map { height: 220px; }
        }
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_GOOGLE_MAPS_API_KEY"></script>
</head>
<body>
    <div id="container">
        <h2 style="text-align:center;">MOSDAC TCC Detection</h2>
        <div id="map"></div>
        <div id="controls">
            <button id="locateBtn">Use My Current Location</button>
            <button id="detectBtn">Detect TCC at Selected Location</button>
        </div>
        <div id="result"></div>
    </div>
    <script>
        let map, marker, selectedLatLng;
        function initMap() {
            map = new google.maps.Map(document.getElementById('map'), {
                center: {lat: 20.5937, lng: 78.9629},
                zoom: 5,
                gestureHandling: 'greedy'
            });
            map.addListener('click', function(e) {
                placeMarker(e.latLng);
            });
        }
        function placeMarker(location) {
            if (marker) marker.setMap(null);
            marker = new google.maps.Marker({ position: location, map: map });
            selectedLatLng = location;
        }
        window.onload = initMap;
        document.getElementById('detectBtn').onclick = function() {
            if (!selectedLatLng) {
                alert('Please select a location on the map.');
                return;
            }
            document.getElementById('result').innerHTML = 'Processing...';
            fetch('/detect_tcc', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat: selectedLatLng.lat(), lng: selectedLatLng.lng() })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('result').innerHTML = `<img src="${data.image_url}" style="max-width:100%"><br><a href="${data.image_url}" download>Download Screenshot</a><br><pre>${data.metrics}</pre>`;
                } else {
                    document.getElementById('result').innerHTML = 'Error: ' + data.error;
                }
            });
        };
        document.getElementById('locateBtn').onclick = function() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    const pos = { lat: position.coords.latitude, lng: position.coords.longitude };
                    map.setCenter(pos);
                    map.setZoom(8);
                    placeMarker(pos);
                }, function() {
                    alert('Unable to retrieve your location.');
                });
            } else {
                alert('Geolocation is not supported by your browser.');
            }
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/detect_tcc', methods=['POST'])
def detect_tcc():
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    # For now, just use the existing capture and analysis (location not used yet)
    try:
        screenshot = capture_mosdac_image()
        metrics = analyze_tcc(screenshot)
        # Serve the image via a static route
        image_url = f"/screenshot/{os.path.basename(screenshot)}"
        return jsonify({"success": True, "image_url": image_url, "metrics": metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/screenshot/<filename>')
def serve_screenshot(filename):
    return send_file(os.path.join('screenshots', filename))

if __name__ == "__main__":
    app.run(debug=True)
