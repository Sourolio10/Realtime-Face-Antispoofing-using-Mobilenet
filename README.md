# Realtime Face Antispoofing using Mobilenet

Realtime Face Antispoofing using Mobilenet is a project aimed at detecting and distinguishing real faces from fake ones in realtime using deep learning techniques. The project utilizes the Mobilenet deep learning architecture for face detection and classification.

## Features

- Realtime face detection and classification
- Uses pre-trained Mobilenet model for efficient computation
- Integration with webcam for live video feed
- Face detection using Haar Cascade classifier
- Display of prediction (Fake/Real) on video feed

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (cv2)
- Keras
- NumPy

## Setup

1. Clone this repository to your local machine.

git clone https://github.com/Sourolio10/Realtime-Face-Antispoofing-using-Mobilenet.git

2. Install the required dependencies using pip.


3. Download the pre-trained Mobilenet model and Haar Cascade classifier XML file and place them in the project directory.

- [Mobilenet model](https://github.com/Sourolio10/Realtime-Face-Antispoofing-using-Mobilenet/blob/main/antispoofmobilenet2.h5)
- [Haar Cascade classifier XML file](https://github.com/Sourolio10/Realtime-Face-Antispoofing-using-Mobilenet/blob/main/haarcascade_frontalface_default.xml)

## Usage

1. Run the Python script `realtime_face_antispoofing.py`.


2. The webcam feed will open, and the program will start detecting and classifying faces in realtime.

3. The classification result (Fake/Real) will be displayed on the video feed.

4. Press 'q' to exit the program.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Sourolio10/Realtime-Face-Antispoofing-using-Mobilenet/blob/main/LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/) for computer vision functionalities.
- [Keras](https://keras.io/) for deep learning model building and training.
- [NumPy](https://numpy.org/) for numerical computations.
- [Mobilenet](https://keras.io/api/applications/mobilenet/) for efficient deep learning architecture.
- [Haar Cascade Classifier](https://docs.opencv.org/4.5.4/d7/d8b/tutorial_py_face_detection.html) for face detection.

## Output:
![alt text](https://github.com/Sourolio10/Realtime-Face-Antispoofing-using-Mobilenet/blob/main/1616342877373.jpg)
