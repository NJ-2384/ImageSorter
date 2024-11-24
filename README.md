# ImageSorter
ImageSorter is a two-part program:
- Firstly it trains on given dataset and generates an encoding model of a face.
- Then we run the algorithm to scan all the images in a given file structure and cross-reference the trained model to separate those images with more than 55% match with about 99.58% accuracy

## Model Training:

- All the photos inside the "Dataset/{Your Name}" directory will be converted into a "numpy array" from where every image is then converted into a list of 128 vector encodings.
  
- All the encodings of all the images are then stored into a file called "Your Name.p" to create a model of your face.

## Image Sorter

The model created using "Model Training.py" is now used here to cross-reference your face in the directory you provide.

- The algorithm "walks" the entire file directory system and generates a list of all the ".png" and ".jpg/.jpeg" files and then converts all of them into vector encodings.
  
- It then finds the vector encoding distance of the image with all of the vector encodings the model file using the "Euclidean" method.
  
- if the minimum vector distance and the average vector distance is above a certain threshold, it classify the image as identified and move it to a new folder on your desktop
