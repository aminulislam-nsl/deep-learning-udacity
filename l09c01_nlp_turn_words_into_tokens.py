# Import the Tokenizer 

from tensorflow.keras.preprocessing.text import Tokenizer 


# Write some sentences

sentences = [
    'My favorite food is ice cream',
    'do you like ice cream too?',
    'My dog likes ice cream!',
    'your favorite flavor of icecream is chocolate',
    "chocolate isn't good for dogs",
    'your dog, your cat, and your parrot prefer broccoli'
]

tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)

# Examine the word index
word_index = tokenizer.word_index
# print(word_index)

# Get the number for a given word
# print(word_index['favorite'])

# Create sequences for the sentences
sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)

sentences2 = ["I like hot chocolate", "My dogs and my hedgedog like kibble but my squirrel prefers grapes and my chickens like ice cream, preferably vanilla"]

sequences2 = tokenizer.texts_to_sequences(sentences2)
print(sequences2)
