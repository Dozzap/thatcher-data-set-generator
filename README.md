This program creates the resized outputs (flipped,upright of both original and thacherized), please install all the dependencies first (not the one in yml)

dataset inputs (original images) are taken from [Kaggle.com](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset), please download the dataset here and use it when generating the data.

Note that the dataset contains different file formats, I only configured it to use jpg

make sure that the images are going to be processed are inputted in the input_images original.



Before downloading the file below, make sure that you have the Git Large File Storage application installed [downloadable from this link](https://git-lfs.com/)

Please download the shape_predictor_68_landmarks.dat file using this [link](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and make sure to put it in the same directory as the main file.


run 

```Bash
python main.py
```

the directories are created as you run the code