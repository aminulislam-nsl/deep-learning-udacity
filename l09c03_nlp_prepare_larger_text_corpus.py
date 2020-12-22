# Import Tokenizer and pad_sequences
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import numpy and pandas
import numpy as np 
import pandas as pd 

# Read the csv file
dataset = pd.read_csv('combined_data.csv')

# Review the first few entries in the dataset
# print(dataset.head())

# Get the reviews from the text column
reviews = dataset['text'].tolist()
# print(reviews)

tokenizer = Tokenizer(oov_token="<oov>")
tokenizer.fit_on_texts(reviews)

word_index = tokenizer.word_index
# print(len(word_index))
# print(word_index)

sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, padding = 'post')

print(padded_sequences.shape)
print(reviews[0])
print(padded_sequences[0])