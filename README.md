# Digital Image Processing & Object Detection System

A Python desktop application for image processing, visualization, segmentation, and object detection. It was developed during my Summer Research Internship as an AI/ML Engineer Intern at **DRDO-DLJ, Jodhpur** from **2 June 2025 to 16 July 2025**.

The main application uses a dual-canvas workflow: the original image is shown on the left and the processed result is shown on the right. Processing operations can be combined and the final image can be exported.

## Features

- Open and save JPG, PNG, and BMP images
- Dual input/output image canvas using Tkinter
- Image filters:
  - Gaussian blur
  - Canny edge detection
  - Sobel edge detection
  - Prewitt edge detection
  - Laplacian edge detection
  - Binary thresholding
- Morphological operations:
  - Dilation
  - Erosion
  - Opening
  - Closing
- Image statistics and analysis:
  - Mean, mode, and median
  - BGR channel histogram
  - Histogram equalization
- Non-Local Means color denoising
- K-Means color quantization and image segmentation
- YOLOv8 object detection with annotated output
- Interactive line, rectangle, and circle drawing
- Image blending, resizing, and configurable Gaussian blur

## Technologies

- Python
- OpenCV
- Tkinter
- NumPy
- Pillow (PIL)
- Matplotlib
- Ultralytics YOLOv8

## Project Structure

```text
.
|-- OPENCV.py             # Main desktop application
|-- yolov8n.pt            # YOLOv8 Nano model weights
|-- requirements.txt      # Python dependencies
|-- files/                # Beginner OpenCV practice scripts
|-- Photos/               # Sample images and reference material
|-- Videos/               # Sample video media
|-- best2.py              # Earlier application version
|-- OPENCV_02.py          # Earlier application version
`-- ...                   # Additional image-processing experiments
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tkinter is included with most standard Python installations on Windows. If Python was installed without Tcl/Tk support, reinstall Python and enable the Tcl/Tk option.

## Run the Application

From the repository root, run:

```bash
python OPENCV_Final.py
```

The application expects `yolov8n.pt` in the project root. Ultralytics may download the model automatically if it is not present, depending on the local installation and network access.

## Basic Usage

1. Open an image through **File > Open**.
2. Select an operation from **Filters**, **Morphology**, **Mathematical**, **Restore**, **Segment**, **Detection**, or **Extra**.
3. Select a drawing tool from **Shapes**, then draw on the output canvas.
4. Export the result through **File > Save Output**.

Operations modify the current output image. Use **File > Reset** to clear the canvases and begin again.

## Internship Outcome

This project combined classical digital image processing with deep-learning-based object detection in one interactive tool. It provided practical experience with GUI development, image representation, filtering, morphology, denoising, segmentation, visualization, and YOLOv8 inference.

## Notes

- The current threshold menu item implements fixed binary thresholding at intensity `127`.
- YOLOv8 inference is most useful on images containing objects from the model's training classes.
- Large media files and model weights can increase repository size. Git LFS may be useful if the repository grows significantly.

## License

No license has been selected for this project yet. Add a license before presenting the repository as reusable open-source software.