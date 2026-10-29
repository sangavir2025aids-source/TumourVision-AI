
# TumourVision AI
### Precision Liver Tumour Segmentation Using AI

## Project Overview
TumourVision AI is a deep-learning project that uses a U-Net model to segment liver and tumour regions in CT scan images. It provides a web interface for uploading scans, viewing segmentation results, and generating PDF reports.

## Objectives
- Segment liver and tumour regions in CT images.
- Visualize predicted segmentation results.
- Display image-based tumour region measurements.
- Generate downloadable PDF reports.

## Technologies Used
- Python
- TensorFlow and Keras
- U-Net Architecture
- Flask
- HTML, CSS, JavaScript
- OpenCV, SimpleITK, NiBabel
- ReportLab

## Dataset
The project uses the Liver Tumor Segmentation Benchmark (LiTS) dataset for training and validation.

## Main Features
- CT scan upload interface
- AI-based segmentation visualization
- Patient details form
- Image-based tumour region analysis
- PDF report generation

## How to Run
1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the required Python packages.
4. Configure the dataset and trained model paths.
5. Run the Flask application:

```bash
python app.py
```

6. Open the local URL displayed in the terminal.

## Limitations
This is an academic prototype and is not clinically validated. Its segmentation output and measurements must not be used as a medical diagnosis or treatment recommendation.

## Future Enhancements
- Improve segmentation accuracy, especially for small tumours.
- Evaluate performance using appropriate segmentation metrics.
- Conduct further validation with medical experts.

## Author
Sangavi R

## License
No license has been specified yet.
