# AIpiano: Music Generation using Deep Learning

This project focuses on generating music using deep learning techniques. It utilizes the music21 library for music processing and the Keras library for building and training a deep learning model. The goal of the project is to create a web application in which you can interact with an online piano and receive a bluesy piano
output. Currently, I am working on improving the model and its output files. This is my first project that incorporates deep learning and is inspired by my own experiences as a pianist.

## Project Structure

The project consists of the following files and directories:

- `use_model_piano.py`: The main script for generating music.
- `best_model1.h5`: The current up to date trained model file.
- `unique_x.pkl`: A serialized file containing the unique_x list.
- `train_model_piano.py`: The script for training the model
- Various example files: examples of the midi output
## Installation

To run the code, follow these steps:

1. Clone the repository or download the project files.
2. Install the required dependencies:
   - music21
   - numpy
   - scikit-learn
   - matplotlib
   - Keras

## Usage

1. Ensure that the necessary dependencies are installed.
2. Run the `use_model_piano.py` script, it will generate midi files based on the current most up-to-date model.
4. The script will load the trained model and unique_x list.
5. The `useModel` function generates music using the model and saves it as MIDI files.
6. The generated MIDI files will be saved in the current directory.
7. It is also possible to retrain the model on a different dataset

## Model Training

The music generation model was trained using the following steps:

1. MIDI files of blues music were collected and stored in the `BluesMidiSample/` directory, this directory is currently not uploaded onto github.
2. The `read_midi` function was used to parse the MIDI files and extract the notes played by the piano.
3. The frequency of each note was computed and only the frequent notes were considered for training.
4. The notes were converted into sequences of fixed length and prepared as input and output sequences.
5. The sequences were split into training and validation sets using a 80:20 split.
6. A deep learning model with convolutional and dense layers was defined and compiled.
7. The model was trained on the training data for 50 epochs with a batch size of 128.
8. The best model based on validation loss was saved as `best_model1.h5`.

## Contributing

Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## Credits

For this project I followed various youtube tutorials and specifically https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/.
The code in this article is the baseline for the program from which I adapted it to better suit blues piano.

