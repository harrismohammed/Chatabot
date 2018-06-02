# Building a ChatBot with Deep NLP
 
 
 
# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
from tensorflow.python.layers.core import Dense
 
 
########## PART 1 - DATA PREPROCESSING ##########
 
 
 
# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
 
# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
 
# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
 
# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
 
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
 
# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
 
# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
 
# Creating a dictionary that maps each word to its number of occurrences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
 
# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshold_questions = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1
threshold_answers = 20
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1
 
# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
 
# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
 
# Adding the End Of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
 
# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT> 
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
 
# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
 
    
    
########## NEURAL ARCHITECTURE ######################################
            

def model_inputs():
    
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    encoder_seq_len = tf.placeholder_with_default(25, None, name='encoder_seq_len')
    decoder_seq_len = tf.placeholder_with_default(25, None, name='decoder_seq_len')
    max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')

    return inputs, targets, keep_prob, lr, encoder_seq_len, decoder_seq_len, max_seq_len


def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>']) 
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1], [1,1]) 
    #tf.strided_slice(target, start, end (all columns except last), slide)
    preprocessed_targets = tf.concat([left_side,right_side], 1)
    #axis = 1 horizontal else vertical
    return preprocessed_targets



def encoder(inputs, rnn_size, num_layers, encoder_seq_len, keep_prob, encoder_vocab_size, encoder_embed_size) :
    def cell(units, rate):
        layer = tf.contrib.rnn.BasicLSTMCell(units)
        return tf.contrib.rnn.DropoutWrapper(layer, rate)
    
    encoder_cell = tf.contrib.rnn.MultiRNNCell([cell(rnn_size, keep_prob) for _ in range(num_layers)])
     
    encoder_embedings = tf.contrib.layers.embed_sequence(inputs, encoder_vocab_size, encoder_embed_size) #used to create embeding layer for the encoder
    
    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                      encoder_embedings,
                                                      encoder_seq_len, 
                                                      dtype=tf.float32)
    
    return encoder_output, encoder_state



def attention_mech(rnn_size, keep_prob, encoder_output, encoder_state, encoder_seq_len, batch_size):
   
    #using internal function to easier create RNN cell
    def cell(units, probs):
        layer = tf.contrib.rnn.BasicLSTMCell(units)
        return tf.contrib.rnn.DropoutWrapper(layer, probs)
    
    #defining rnn_cell
    decoder_cell = cell(rnn_size, keep_prob)
    
    #using helper function from seq2seq sub_lib for Bahdanau attention
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, 
                                                               encoder_output, 
                                                               encoder_seq_len)
    
    #finishin attention with the attention holder - Attention Wrapper
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
                                                   attention_mechanism, 
                                                   rnn_size/2)
    
    #Here we are usingg zero_state of the LSTM (in this case) decoder cell, and feed the value of the last encoder_state to it
    attention_zero = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    enc_state_new = attention_zero.clone(cell_state=encoder_state[-1])
    
    return dec_cell, enc_state_new



def decoder(decoder_embedded_input, encoder_state, dec_cell, decoder_embed_size, vocab_size, dec_seq_len, max_seq_len, word2int, batch_size):

	#Defining embedding layer for the Decoder
    embed_layer = tf.Variable(tf.random_uniform([vocab_size, decoder_embed_size]))
    embedings = tf.nn.embedding_lookup(embed_layer, decoder_embedded_input) 
    
    #Creating Dense (Fully Connected) layer at the end of the Decoder -  used for generating probabilities for each word in the vocabulary
    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    

    with tf.variable_scope('decoder'):
        #Training helper used only to read inputs in the TRAINING stage
        train_helper = tf.contrib.seq2seq.TrainingHelper(embedings, 
                                                         dec_seq_len)
        
        #Defining decoder - You can change with BeamSearchDecoder, just beam size
        train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                                        train_helper, 
                                                        encoder_state, 
                                                        output_layer)
        
        #Finishing the training decoder
        train_dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, 
                                                                   impute_finished=True, 
                                                                   maximum_iterations=max_seq_len)
        
    with tf.variable_scope('decoder', reuse=True): #we use REUSE option in this scope because we want to get same params learned in the previouse 'decoder' scope
        #getting vector of the '<SOS>' tags in the int representation
        starting_id_vec = tf.tile(tf.constant([word2int['<SOS>']], dtype=tf.int32), [batch_size], name='starting_id_vec')
        
        #using basic greedy to get next word in the inference time (based only on probs)
        test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed_layer, 
                                                               starting_id_vec, 
                                                               word2int['<EOS>'])
        
        #Defining decoder - for inference time
        test_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       test_helper, 
                                                       encoder_state, 
                                                       output_layer)
        
        
        test_dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(test_decoder, 
                                                                  impute_finished=True, 
                                                                  maximum_iterations=max_seq_len)
        
    train_prediction = train_dec_output
    test_prediction = test_dec_output
    return train_prediction, test_prediction

# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, max_seq_len, answers_num_words, questions_num_words, encoder_embed_size, decoder_embed_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embed_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_vocab_size = answers_num_words + 1
    vocab_size = questions_num_words + 1
    encoder_state = encoder(encoder_embedded_input, rnn_size, num_layers, keep_prob, encoder_seq_len, encoder_vocab_size, encoder_embed_size)
    
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embed_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    training_predictions, test_predictions = decoder(decoder_embedded_input,
                                                     decoder_embeddings_matrix,
                                                     encoder_state,
                                                     questions_num_words,
                                                     max_seq_len,
                                                     rnn_size,
                                                     num_layers,
                                                     questionswords2int,
                                                     keep_prob,
                                                     batch_size)
    return training_predictions, test_predictions
 
 

 
 
# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoder_embed_size = 512
decoder_embed_size = 512
lr = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_prob = 0.5
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob, encoder_seq_len, decoder_seq_len, max_seq_len = model_inputs()
 
# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
 

# Getting the training and test predictions

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       max_seq_len,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoder_embed_size,
                                                       decoder_embed_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
