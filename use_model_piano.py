#library for understanding music
from music21 import *
import numpy as np
#imports for folder access/training split
import os
from sklearn.model_selection import train_test_split
#importing library
from collections import Counter

#library for visualiation
import matplotlib.pyplot as plt

#importing model
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K


from music21 import stream, note, chord, tempo
import random
from random import choice
import pickle

with open('unique_x.pkl', 'rb') as f:
    unique_x = pickle.load(f)

x_val = np.load("x_val.npy")

model = load_model('best_model1.h5')


def useModel(fileName, no_of_timesteps, model, x_val, unique_x, top_k=5, temperature = 0.6):
    start = np.random.randint(0, len(x_val) - 1)
    rand_music = x_val[start]
    predictions = []

    for i in range(32):
        rand_music = rand_music.reshape(1, no_of_timesteps)
        prob = model.predict(rand_music)[0]
        
        #temperature
        prob = np.log(prob) / temperature
        exp_prob = np.exp(prob)
        top_prob = exp_prob / np.sum(exp_prob)

        # top K selection
        top_index = np.argsort(top_prob)[-top_k:]


        top_prob = top_prob[top_index] / np.sum(top_prob[top_index])

        y_pred = np.random.choice(top_index, p =top_prob)

        predictions.append(y_pred)

        rand_music = np.insert(rand_music[0], len(rand_music[0]), y_pred)
        rand_music = rand_music[1:]

    x_note_to_int = dict((number, note_) for number, note_ in enumerate(unique_x))
    predicted_notes = [x_note_to_int[i] for i in predictions]
    convert_to_midi(predicted_notes, fileName)


def convert_to_midi(prediction_output, fileName, tempo_value=120, time_signature='4/4'):
    offset = 0
    output_notes = []
    
    # Create a meter object based on the time signature
    time_sig_obj = meter.TimeSignature(time_signature)

    # Define the three possible durations
    durations = [0.25, 0.5, 1.0]  # Example durations

    # Create a tempo indication for the desired tempo value
    tempo_indication = tempo.MetronomeMark(number=tempo_value)

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                cn = int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                
                # Randomly select duration from the list
                new_note.duration.quarterLength = choice(durations)
                
                notes.append(new_note)

            output_notes.append(chord.Chord(notes, quarterLength=new_note.duration.quarterLength, offset=offset))

        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.storedInstrument = instrument.Piano()
            
            # Randomly select duration from the list
            new_note.duration.quarterLength = choice(durations)
            
            output_notes.append(new_note)

        offset += new_note.duration.quarterLength
        
        # Check if the measure is complete and move to the next measure if necessary
        while offset >= time_sig_obj.barDuration.quarterLength:
            offset -= time_sig_obj.barDuration.quarterLength

    # Create a Stream object and add the tempo indication and output notes
    midi_stream = stream.Stream()
    midi_stream.append(time_sig_obj)  # Add the time signature
    midi_stream.append(tempo_indication)
    midi_stream.append(output_notes)

    # Write the MIDI file
    midi_stream.write('midi', fp=fileName + '.mid')


useModel("example_file1", 32, model, x_val, unique_x)
useModel("example_file2", 32, model, x_val, unique_x)
useModel("example_file3", 32, model, x_val, unique_x)
useModel("example_file4", 32, model, x_val, unique_x)
useModel("example_file5", 32, model, x_val, unique_x)
