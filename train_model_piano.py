from music21 import *
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K
import pickle


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

np.save('x_val.npy', x_val)

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

mc=ModelCheckpoint('best_model1.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)

history = model.fit(np.array(x_tr),np.array(y_tr),batch_size=128,epochs=50, validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc])


#loading best model
from keras.models import load_model
model = load_model('best_model1.h5')



# Save the unique_x list to a file
with open('unique_x.pkl', 'wb') as f:
    pickle.dump(unique_x, f)
