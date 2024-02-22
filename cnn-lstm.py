from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu



print("complit")


# extract features from each photo in the directory
def extract_features(directory):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	features = dict()
	print("------------")
	print(features)
 
	for name in listdir(directory):
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		#print(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		print(feature, feature.shape)
        #print(feature.type)
		image_id = name.split('.')[0]
		features[image_id] = feature
	return features


# extract features from all images
directory = '/Users/haegwangpark/Downloads/archive/Flickr8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('/Users/haegwangpark/Downloads/archive/features.pkl', 'wb'))



# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text




# extract descriptions for images
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        #print("=-=-=-=-=")
        #print(line)
    # split line by white space
        tokens = line.split()
        print("=-=-=-=-=")
        print(tokens)
        if len(line) < 2:
            continue
    
    # take the first token as the image id, the rest as the description
        
        print("=========================")
        print(tokens)

        image_id, image_desc = tokens[0], tokens[1:]

        
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        print("image_desc")
        print(image_desc)
        image_desc = ' '.join(image_desc)
        print(image_id)
        print("image_desc")
        print(image_desc)


        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
            # store description
            mapping[image_id].append(image_desc)

        print("-----------------mapping")
        print(mapping)

    return mapping

import string

def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
    # tokenize
    desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word)>1]
    # remove tokens with numbers in them
    desc = [word for word in desc if word.isalpha()]
    # store as string
    desc_list[i] =  ' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc
 
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()




filename = '/Users/haegwangpark/Downloads/archive/Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)

print("================= doc")
print(doc)

 
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))


# clean descriptions
clean_descriptions(descriptions)

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

# save to file
save_descriptions(descriptions, '/Users/haegwangpark/Downloads/archive/descriptions.txt')

"""



def model(vocab_size, max_length):

	input1 = Input(shape=(4096,))
	drop_1 = Dropout(0.5)(inputs1)
	dense_1 = Dense(256, activation='relu')(fe1)

	inputs2 = Input(shape=(max_length,))
	embed = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	drop_2 = Dropout(0.5)(se1)
	dense_2 = LSTM(256)(se2)

	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.summary()
	return model

	
"""	