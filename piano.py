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

#defining function to read MIDI files
def read_midi(file):
    
    print("Loading Music File:",file)
    
    notes=[]
    notes_to_parse = None
    
    #parsing a midi file
    midi = converter.parse(file)
  
    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #Looping over all the instruments
    for part in s2.parts:
    
        #select elements of only piano
        if 'Piano' in str(part): 
        
            notes_to_parse = part.recurse() 
      
            #finding whether a particular element is note or a chord
            for element in notes_to_parse:
                
                #note
                if isinstance(element, note.Note):
                    note_name = str(element.pitch)
                    octave = int(note_name[-1])  # Extract the octave value from the note string
                    # Check if the octave is within the desired range
                    if octave == 4 or octave == 5:
                        notes.append(note_name)
                
                #  #chord
                # elif isinstance(element, chord.Chord):
                #     notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)

#folder selection
path = 'BluesMidiSample/'
files=[i for i in os.listdir(path) if i.endswith(".mid")]
print(files)

#pass each file to array
notes_array = np.array([read_midi(path+i) for i in files], dtype = object)

#converting 2D array into 1D array
notes_ = [element for note_ in notes_array for element in note_]

#No. of unique notes
unique_notes = list(set(notes_))
print(len(unique_notes))



#computing frequency of each note
freq = dict(Counter(notes_))

print(freq.items())

#consider only the frequencies
no=[count for _,count in freq.items()]

#set the figure size
plt.figure(figsize=(5,5))

#plot
plt.hist(no)

#may need to change based on data collection
frequent_notes = [note_ for note_, count in freq.items() if count>=15]
print(len(frequent_notes))

new_music=[]

for notes in notes_array:
    temp=[]
    for note_ in notes:
        if note_ in frequent_notes:
            temp.append(note_)            
    new_music.append(temp)
    
new_music = np.array(new_music, dtype=object)

#could maybe change timesteps, if all I want is the riff
no_of_timesteps = 32
x = []
y = []

for note_ in new_music:
    for i in range(0, len(note_) - no_of_timesteps, 1):
        
        #preparing input and output sequences
        input_ = note_[i:i + no_of_timesteps]
        output = note_[i + no_of_timesteps]
        
        x.append(input_)
        y.append(output)
        
x=np.array(x, dtype = object)
y=np.array(y, dtype = object)
unique_x = list(set(x.ravel()))
x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

print (x_note_to_int.items())
#somehow need to access dict later to map to blues scale






#preparing input sequences
x_seq=[]
for i in x:
    temp=[]
    for j in i:
        #assigning unique integer to every note
        temp.append(x_note_to_int[j])
    x_seq.append(temp)
    
x_seq = np.array(x_seq, dtype=object)

unique_y = list(set(y))
y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
print(y_note_to_int.items())
y_seq=np.array([y_note_to_int[i] for i in y], dtype = object)


## maybe adjsut random_state a little for bluesy variability
x_tr, x_val, y_tr, y_val = train_test_split(x_seq,y_seq,test_size=0.2,random_state=0)

x_tr = np.array(x_tr).astype(int)
y_tr = np.array(y_tr).astype(int)
x_val = np.array(x_val).astype(int)
y_val = np.array(y_val).astype(int)


#defining training model
#experiment with different layers/hidden units

def lstm(num_lstm_layers=2, lstm_units=128):
    model = Sequential()

    # Add LSTM layers
    for _ in range(num_lstm_layers):
        model.add(LSTM(lstm_units, return_sequences=True))

    # Add the final LSTM layer
    model.add(LSTM(lstm_units))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense('n_vocab'))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model


K.clear_session()
model = Sequential()
    
#embedding layer
model.add(Embedding(len(unique_x), 100, input_length=32,trainable=True)) 

model.add(Conv1D(64,3, padding='causal',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
    
model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))

model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
          
#model.add(Conv1D(256,5,activation='relu'))    
model.add(GlobalMaxPool1D())
    
model.add(Dense(256, activation='relu'))
model.add(Dense(len(unique_y), activation='softmax'))
    
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.summary()

mc=ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)

history = model.fit(np.array(x_tr),np.array(y_tr),batch_size=128,epochs=50, validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc])


#loading best model
from keras.models import load_model
model = load_model('best_model.h5')


def useModel(fileName, no_of_timesteps, model, x_val, unique_x, top_k=5, temperature = 1.0):
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

# def convert_to_midi(prediction_output, fileName, tempo_value=160, ):
#     offset = 0
#     output_notes = []

#     # Create a tempo indication for the desired tempo value
#     tempo_indication = tempo.MetronomeMark(number=tempo_value)

#     # Create note and chord objects based on the values generated by the model
#     for pattern in prediction_output:
#         # Pattern is a chord
#         if ('.' in pattern) or pattern.isdigit():
#             notes_in_chord = pattern.split('.')
#             notes = []
#             for current_note in notes_in_chord:
#                 cn = int(current_note)
#                 new_note = note.Note(cn)
#                 new_note.storedInstrument = instrument.Piano()
#                 notes.append(new_note)

#             # Apply the swing rhythm to the chord's duration
#             duration = 1.5

#             new_chord = chord.Chord(notes)
#             new_chord.offset = offset
#             new_chord.duration.quarterLength = duration
#             output_notes.append(new_chord)

#         # Pattern is a note
#         else:
#             new_note = note.Note(pattern)

#             # Apply the swing rhythm to the note's duration
#             duration = 0.75

#             new_note.offset = offset
#             new_note.duration.quarterLength = duration
#             new_note.storedInstrument = instrument.Piano()
#             output_notes.append(new_note)

#         # Increase the offset based on the duration of the note or chord
#         offset += duration

#     # Create a Stream object and add the tempo indication and output notes
#     midi_stream = stream.Stream()
#     midi_stream.append(tempo_indication)
#     midi_stream.append(output_notes)

#     # Write the MIDI file
#     midi_stream.write('midi', fp= fileName + '.mid')

useModel("example_file1", 32, model, x_val, unique_x)
useModel("example_file2", 32, model, x_val, unique_x)
useModel("example_file3", 32, model, x_val, unique_x)
useModel("example_file4", 32, model, x_val, unique_x)
useModel("example_file5", 32, model, x_val, unique_x)