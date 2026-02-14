# Chest X-Ray Disease Detection

• Designed deep learning pipeline with TensorFlow and DenseNet121 for multi-GPU chest X-ray classification

• Streamlined preprocessing and caching for medical imaging, boosting training and validation speed by 30%

• Tuned hyperparameters and evaluated achieving 0.9495 binary accuracy and 0.857 multi-label accuracy

We created different models to classify many types of diseases in chest x-ray images
Diseases include pneumonia, hernia, emphysema, and many more
Our goal was to create models that would detect if an image contains a disease and also predict which specific disease the image shows
We used a random forest classifier for our simple model and a CNN for our more advanced model
The steps for this project included loading and preprocessing the dataset, training the models, and comparing their performance to determine which had the best results

## Why did we do this?

Our goal was to create a model that would help health care professionals in diagnosing diseases more efficiently through chest x-ray images
This model could be a big help to doctors when trying to figure out if someone has a disease or not by allowing faster disease diagnoses
By using our model to detect diseases, doctors will be able to treat patients earlier potentially allowing more lives to be saved
Creating a model for classifying diseases in chest x-rays would reduce the burden on healthcare workers
Fast diagnosis would help understaffed healthcare facilities
It would allow for more patients to be seen

## Data info
Obtained from the picture archiving and communication system (PACS) database at National Institutes of Health Clinical Center in Bethesda, Maryland
Around 60% of all frontal chest x-rays in the hospital
Representative of real population distributions
112,120 images of 30,805 unique patients
1028x1028 pixel images
21 binary disease labels (1 if has disease, 0 if not)
Atelectasis, Hernia, No Finding, etc.

## Random Forest Model
We used a Random Forest Classifier model provided by sklearn
After splitting the data into training and testing sets, a Random Forest Classifier was created using 100 trees and a max depth of 10
We implemented two different Random Forest Classifier models
1 attempted to classify if any disease was present
1 attempted to classify the specific disease that was present
Used a one-vs-the-rest classifier to enable multilabel classification
We evaluated the models based on metrics provided by the accuracy_score and classification_report functions
Classification accuracy, precision, recall, f1-score

## CNN Binary classification

Model Setup
- Tensorflow, Pandas, Scikit-Learn, Numpy
- Binary Labels (1 for any disease and 0 for healthy)
- Preprocessing images
- Loads and resizes the images
- Converts the pixels to arrays and normalizes the pixel values
- Create Dataset function
- Takes the image filenames from csv file 
- Finds the images and normalizes the images
- Create Model function
- Calculates metrics for evaluation (Accuracy, Precision, Recall, F1 Score)
- Used the DenseNet121 model 
- Uses Binary cross -entropy loss

## Limitation & Improvements

- Not enough time / computational power to account for extremely long runtimes due to size/amount of images
  - Required manual scaling down of images in order to reduce processing time
  - In a medical setting, scaling down could miss out on key information depending on the scaling values
  - Could result in less overfitting
  - Could improve parameter selection using algorithms like GridSearchCV
  - Would further increase computation time, but would result in model optimization
- Increase size of trees in forest
- For individual disease classification, select specific part of images to do segmentation with the X-Rays
  - This could include lung segmentation, heart segmentation etc. Done to focus on specific, relevant anatomy 
  - Would reduce overfitting of background details within the images
  - It does include an extra step, which means longer pipeline process to implement the model
- Different Dataset selection
  -  Find a dataset that contains more information about the patients (age, gender, family history, past medical records, etc.)
  -  This could theoretically allow for more parameters to consider when creating a model, as we would be able to categorize images by age group, as something like age can be a big factor in susceptibility to diseases.




