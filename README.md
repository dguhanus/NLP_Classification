# NLP_Classification
Objective: Doing text classification using BiLSTM and Transformer model. 

The BiLSTM can be trained by any user.

Steps:
1. You just define your LSTM architecture as given in my code.

# Build neural network architecture
model = Sequential()

model.add(Embedding(vocabSize, embedding_dim, input_length=X_train.shape[1], weights=[embedding_matrix], trainable=False))

#model.add(Bidirectional(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))

model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))

model.add(Bidirectional(LSTM(64, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))

model.add(Bidirectional(LSTM(32, dropout=0.2,recurrent_dropout=0.2)))
#model.add(Bidirectional(LSTM(12, dropout=0.2,recurrent_dropout=0.2)))

model.add(Dense(6, activation='softmax'))


2. A new user don't have to train this Neural Network by downloading training and testing data again. Just load the pre-trained model (that training was done by me) .
Pre-trained model file : Kaggle_MultiClass.h5

Using below code:
transferredModel = keras.models.load_model("Kaggle_MultiClass.h5")

3.  Find out the sentiment of a given sentence like : 'my elder brother is dead' .
Follow the below codes: 

sentence= 'my elder brother is dead'

print(sentence)

sentence = normalized_sentence(sentence)

sentence = tokenizer.texts_to_sequences([sentence])

sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]

proba =  np.max(transferredModel.predict(sentence))

print(f"{result} : {proba}\n\n") 

Important Note:
Please do the preprocessing of any given sentence that you are interested to find the sentiment of that unknown sentence.
Preprocessing functions are:

normalized_sentence()

tokenizer.texts_to_sequences()
pad_sequences()
le.inverse_transform
