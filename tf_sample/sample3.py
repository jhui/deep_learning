from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from IPython.core.display import display, HTML

#Length of longest sentence
max_caption_len = 3
#Size of vocabulary
vocab_size = 3

# Load one screenshot for each word, and turn them into digits
images = []
for i in range(2):
    images.append(img_to_array(load_img('screenshot.jpg', target_size=(224, 224))))
images = np.array(images, dtype=float)
# Preprocess input for the VGG16 model
images = preprocess_input(images)

#Turn start tokens into one-hot encoding
html_input = np.array(
            [[[0., 0., 0.], #start
             [0., 0., 0.],
             [1., 0., 0.]],
             [[0., 0., 0.], #start <HTML>Hello World!</HTML>
             [1., 0., 0.],
             [0., 1., 0.]]])

#Turn next word into one-hot encoding
next_words = np.array(
            [[0., 1., 0.], # <HTML>Hello World!</HTML>
             [0., 0., 1.]]) # end

# Load the VGG16 model trained on imagenet and output the classification feature
VGG = VGG16(weights='imagenet', include_top=True)
# Extract the features from the image
features = VGG.predict(images)

#Load the feature to the network, apply a dense layer, and repeat the vector
vgg_feature = Input(shape=(1000,))
vgg_feature_dense = Dense(5)(vgg_feature)
vgg_feature_repeat = RepeatVector(max_caption_len)(vgg_feature_dense)
# Extract information from the input seqence
language_input = Input(shape=(vocab_size, vocab_size))
language_model = LSTM(5, return_sequences=True)(language_input)


# Concatenate the information from the image and the input
decoder = concatenate([vgg_feature_repeat, language_model])
# Extract information from the concatenated output
decoder = LSTM(5, return_sequences=False)(decoder)
# Predict which word comes next
decoder_output = Dense(vocab_size, activation='softmax')(decoder)
# Compile and run the neural network
model = Model(inputs=[vgg_feature, language_input], outputs=decoder_output)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Train the neural network
model.fit([features, html_input], next_words, batch_size=2, shuffle=False, epochs=1000)


start_token = [1., 0., 0.] # start
sentence = np.zeros((1, 3, 3)) # [[0,0,0], [0,0,0], [0,0,0]]
sentence[0][2] = start_token # place start in empty sentence

# Making the first prediction with the start token
second_word = model.predict([np.array([features[1]]), sentence])


# Put the second word in the sentence and make the final prediction
sentence[0][1] = start_token
sentence[0][2] = np.round(second_word)
third_word = model.predict([np.array([features[1]]), sentence])

sentence[0][0] = start_token
sentence[0][1] = np.round(second_word)
sentence[0][2] = np.round(third_word)


# Transform our one-hot predictions into the final tokens
vocabulary = ["start", "<HTML><center><H1>Hello World!</H1><center></HTML>", "end"]
html = ""
for i in sentence[0]:
    html += vocabulary[np.argmax(i)] + ' '

from IPython.core.display import display, HTML
display(HTML(html[6:49]))






