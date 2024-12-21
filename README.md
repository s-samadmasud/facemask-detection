
# Face Mask Detection App
This project implements a face mask detection application.

A brief description of what this project does and who it's for

## Project Structure


 
 * [face_mask_detection](./face_mask_detection)
 * [images](./images)
   * [test.png](./images/test.png)
   * [test1.png](./images/test1.png)
   * [test2.png](./images/test2.png)
   * [test3.png](./images/test3.png)
 * [results](./results)
   * [predicted.png](./reslts/predicted.png)
   
 * [app.py](./app.py)
 * [README.md](./README.md)

## Description
This application can be used to detect whether a person in an image is wearing a face mask or not. It utilizes a deep learning model trained on a dataset of images with and without masks. The application takes an image as input, predicts the presence of a mask, and displays the result.

## Setting Up the Environment
### Create a virtual environment:

Open your terminal or command prompt and run the following command to create a virtual environment named venv:

  


```bash
  conda create -p venv python==3.10.0 -y
```

Activate the virtual environment using:

```bash
  conda activate venv
```

### Install dependencies:

Install the required Python libraries listed in the requirements.txt file using:

```bash
  pip install -r requirements.txt
```

## Training the Model
This project requires a pre-trained model for face mask detection. You can find various pre-trained models available online or train your own model using a suitable dataset.

Note: Training a deep learning model can be computationally expensive and time-consuming.

## Using the Application
Place images for prediction in the images folder.

Run the app.py script using:

```bash
  streamlit run app.py
```
