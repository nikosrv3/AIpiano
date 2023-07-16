#library for understanding music
from music21 import *
import numpy as np
import os
from sklearn.model_selection import train_test_split
#importing library
from collections import Counter

#library for visualiation
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K

def convert_to_midi(prediction_output):
   
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # pattern is a note
        else:
            
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='music1.mid')

convert_to_midi([52, 50, 52, 52, 50, 50, 42, 50, 42, 24, 24, 24, 33, 0, 24, 25, 0, 25, 52, 25, 52, 25, 25, 25, 52, 52, 52, 52, 52, 52])

