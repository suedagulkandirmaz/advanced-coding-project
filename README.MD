
To run this app please ensure you have at least version 3.10  of python installed
If you prefer to use virtualenv, create a virtualenv first 
 '''python -m venv my_env'''
Activate your virtual env
''' . my_env/Scripts/activate'''
Install dependencies
pip install -r requirement.txt
once all dependencies installed, you are ready to run the app 
'''python app_flask.py'''

#Plant Disease Detection:
****Unfortunately, all the previous commits I made were deleted. I can't explain why, but the large file images were deleted while doing gitignore.


Some plants get some diseases due to various factors such as environmental conditions. These diseases harm many things.
One of them is the economy.
I tried to do this project to overcome this problem. With this, disease detection in plants will occur very easily.
This project include CNN model that detect disease from plant images.
This project contain user friendly interface Flak Web. 


#Features:
Detects 13 different plant diseases.
Model trained with CNN.
Web interface with flask.

#Prediction Feature:
Disease prediction is obtained from the model by uploading the leaf image.
Results are saved in the database: user information, class label, confidence score, timestamp.

#Project Structure:

├── app/
| ├── core/ 
│ │ ├── _init_.py
│ │ ├── dataset_loader.py
│ │ ├── predictor
│ ├── models/ 
│ │ ├── _init_.py
│ │ ├── cnn_model.py
│ ├── persistence
│ │ ├── _init_.py
│ │ ├──db_handler.py
│ ├── utils/ # Visual processing and prediction auxiliary functions
│ │ ├── image_utils.py
├── static/ # image
├──templates/ # HTML file
│ │ ├── index.html
├── data/ # Training/test dataset
├── trained_model/ # training file (pt)
│ │ ├── model_trainer.py
├── main.py 
├── app_flask.py
├── predict_images.py
├── split_dataset.py
├── requirements.txt # python 
└── README.md # this is