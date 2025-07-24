🌩️ Tropical Cloud Cluster (TCC) Detection using MOSDAC Satellite Imagery
This project implements a web-based tool to detect Tropical Cloud Clusters (TCCs) using satellite infrared brightness temperature (IRBT) data from MOSDAC (ISRO). It automates screenshot capture, processes the image, detects TCCs based on IR temperature thresholding, and computes cloud-top metrics like Tb, radius, and estimated cloud-top heights. The user interface integrates Google Maps for selecting any location over India.

🚀 Features
📷 Automated Satellite Screenshot Capture from MOSDAC

🧠 TCC Detection Algorithm using brightness temperature threshold (IR BT < 240K)

📊 Computes key metrics:

Mean, Min, Max, Median, Std Dev of Tb

Pixel count of TCC

Convective center (x, y)

Min/Max/Mean radial extent

Estimated cloud-top heights (km)

🗺️ Google Maps Integration to select or auto-detect location

⚡️ Fast image analysis using OpenCV + NumPy

🌐 Web app built with Flask

📸 Sample Output

🧰 Tech Stack
Python (Flask, OpenCV, NumPy, Matplotlib)

Selenium (for live image capture)

HTML/CSS/JavaScript

Google Maps API

