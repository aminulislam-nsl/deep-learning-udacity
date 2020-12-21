# Import Tokenizer and pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'My favorite food is ice cream',
    'do you like ice cream too?',
    'My dog likes ice cream',
    'your favorite flavor of icecream is chocolate',
    "chocolate isn't good for dogs",
    'your dog, your cat, and your parrot prefer broccoli'
]

# print(sentences)

tokenizer = Tokenizer(num_words = 100, oov_token = '<oov>')

# Tokenize the words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
# print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)

padded = pad_sequences(sequences)
# print("\nWord Index = ", word_index)
# print("\nSequences = ", sequences)
# print("\nPadded Sequences:")
# print(padded)

# Specify a max length for the padded sequences
padded = pad_sequences(sequences, maxlen = 15, padding="post")
# print(padded)

padded = pad_sequences(sequences, maxlen = 3)
print(padded)

test_data = [
    "my best friend's favorite ice cream flavor is strawberry",
    "my dog's best friend is a manatee"
]

# print(test_data)

print("<oov> has the number", word_index['<oov>'], "in the word index.")

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen = 10)
print("\nPadded Test Sequence: ")
print(padded)