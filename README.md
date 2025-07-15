
# Brain Tumor MRI Classifier

A simple deep learning web app that classifies brain MRI images into:

- Glioma
- Meningioma
- Pituitary
- No Tumor

## Features

- Upload MRI or capture with webcam
- Auto-predict tumor type using a trained CNN model
- Shows confidence and explanation
- Downloadable prediction report

## How to Run

1. Make sure `model.h5` and `streamlit_app.py` are in the same folder
2. Open terminal and run:

```
streamlit run streamlit_app.py
```

## Requirements

- Python
- TensorFlow
- Streamlit
- OpenCV
- PIL

## Files

- `file.ipynb` – CNN training notebook
- `model.h5` – Trained model
- `streamlit_app.py` – Streamlit web app
- `README.md` – Project description

## Author

Madhusudan
