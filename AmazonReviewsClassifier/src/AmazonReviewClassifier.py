# -*- coding: utf-8 -*-

'''
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import pandas as pd

from textwrap import wrap
import re
import itertools
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow 
import torchtext
from torchtext import data
from torchtext.legacy.data import Field
#from torchtext import Field
from torchtext import vocab
from torchtext.vocab import GloVe
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import f1_score
from torch import nn
from sklearn.metrics import precision_score,accuracy_score,recall_score,balanced_accuracy_score
from sklearn.metrics import classification_report




weights = [0.7,0,0,0,0.3]

# Credits - https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
def plot_confusion_matrix(correct_labels, predict_labels, labels, display_labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):

  ''' 
  Parameters:
      correct_labels                  : These are your true classification categories.
      predict_labels                  : These are you predicted classification categories
      labels                          : This is a lit of labels which will be used to display the axix labels
      title='Confusion matrix'        : Title for your matrix
      tensor_name = 'MyFigure/image'  : Name for the output summay tensor

  Returns:
      summary: TensorFlow summary 

  Other itema to note:
      - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
      - Currently, some of the ticks dont line up due to rotations.
  '''
  cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
  if normalize:
      cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
      cm = np.nan_to_num(cm, copy=True)
      cm = cm.astype('int')

  np.set_printoptions(precision=2)
  ###fig, ax = matplotlib.figure.Figure()

  fig = matplotlib.pyplot.figure(figsize=(2, 2), dpi=320, facecolor='w', edgecolor='k')
  ax = fig.add_subplot(1, 1, 1)
  im = ax.imshow(cm, cmap='Oranges')

  classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in display_labels]
  classes = ['\n'.join(wrap(l, 40)) for l in classes]

  tick_marks = np.arange(len(classes))

  ax.set_xlabel('Predicted', fontsize=7)
  ax.set_xticks(tick_marks)
  c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  ax.set_ylabel('True Label', fontsize=7)
  ax.set_yticks(tick_marks)
  ax.set_yticklabels(classes, fontsize=4, va ='center')
  ax.yaxis.set_label_position('left')
  ax.yaxis.tick_left()

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
  fig.set_tight_layout(True)
  matplotlib.pyplot.show()

  return

"""## Load Splits"""

train_df = pd.read_csv('../dataset/train.csv')

print(train_df.shape)


TEXT = torchtext.legacy.data.Field(sequential=True, tokenize="spacy", lower=True, include_lengths=True)

SCORE = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

## Create datafields:
def populateDatafields(somedf):
  datafields = []
  for col in somedf.columns:
    if(col == "Score"):
      datafields.append(("Score", SCORE))
    elif col == "Text":
      datafields.append(("Text", TEXT))
    else:
      datafields.append((col, None))
  return datafields

train_datafields = populateDatafields(train_df)

print(train_datafields)

training_data=torchtext.legacy.data.TabularDataset(path = '../dataset/train.csv',\
                                  format = 'csv',\
                                  fields = train_datafields,\
                                  skip_header = True)

count = 0
for example in training_data:
  print("*******************************")
  print("Text: ", example.Text)
  print("Score: ", example.Score)

  if count > 5:
    break
  count += 1


TEXT.build_vocab(training_data, min_freq = 3, vectors=GloVe(name='6B', dim=300))

print(len(TEXT.vocab))

print(TEXT.vocab.itos)

print(TEXT.vocab.freqs)

print(TEXT.vocab.stoi)

print(TEXT.vocab.vectors)

print(len(TEXT.vocab))



device = torch.device('cuda:0')
#device = torch.device('cuda:1')

BATCH_SIZE = 64

# Define the train iterator
train_iterator = torchtext.legacy.data.BucketIterator(
    training_data, 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.Text),
    sort_within_batch = True,
    repeat=False, 
    shuffle=True,
    device = device)

# # Define the train iterator
# train_iterator = torchtext.legacy.data.BucketIterator(
#     training_data, 
#     batch_size = BATCH_SIZE,
#     sort_key = lambda x: len(x.Text),
#     sort_within_batch = True,
#     repeat=False, 
#     shuffle=True,
#     device = device)

def getFrequencyDistribution(dataset):
  freqDist = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

  for item in dataset:
    if item.Score == '1':
      freqDist['1'] += 1
    elif item.Score == '2':
      freqDist['2'] += 1
    elif item.Score == '3':
      freqDist['3'] += 1
    elif item.Score == '4':
      freqDist['4'] += 1
    else:
      freqDist['5'] += 1
  return freqDist

training_freqDist = getFrequencyDistribution(training_data)
print("Training Frequency Distribution: " ,training_freqDist)



fig, ax = plt.subplots(num=None,figsize=(20,10),dpi=30, facecolor='w', edgecolor='k')

ax.bar(training_freqDist.keys(),training_freqDist.values(),0.9)   
plt.xticks(rotation=0,fontsize=50)
plt.yticks(fontsize=30)
plt.margins(0)

plt.xlabel("class",fontsize=40)
plt.ylabel("frequency",fontsize=40)  
plt.show()
plt.close()


class ReviewClassifier(nn.Module):

  def __init__(self, mode, output_size, hidden_size, vocab_size, embedding_length, word_embeddings):
    super(ReviewClassifier, self).__init__()

    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length
    self.mode = mode

    # Embedding Layer
    self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
    self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

    if mode not in ['rnn', 'lstm', 'gru', 'bilstm']:
      raise ValueError("Choose a mode from - rnn / lstm / gru / bilstm")

    if mode == 'rnn':
      print("RNN")
      self.recurrent = nn.RNN(self.embedding_length, self.hidden_size)

    elif mode == 'lstm':
      print("LSTM")
      self.recurrent = nn.LSTM(self.embedding_length, self.hidden_size)


    elif mode == 'gru':
      print("GRU")
      self.recurrent = nn.GRU(self.embedding_length, self.hidden_size)

    elif mode == 'bilstm':
      print("BiLSTM")
      self.recurrent = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True)

    # Fully-Connected Layer
    self.fc1 = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, text, text_lengths):

    embedded_text = self.embeddings(text)

    # Pack the embedded inputs - This makes sure the hidden output for each example in the batch is from the last token in the sequence (instead of the padded token)
    #packed_text = nn.utils.rnn.pack_padded_sequence(embedded_text, text_lengths)
    packed_text = nn.utils.rnn.pack_padded_sequence(embedded_text, text_lengths.cpu())

    if self.mode in ('lstm' , 'bilstm'):
      _, (hidden_text, _) = self.recurrent(packed_text)

      if self.mode == 'bilstm':
        hidden_text = hidden_text[0,:,:] + hidden_text[1,:,:]
    else:
      _, hidden_text = self.recurrent(packed_text)

    fc_input = hidden_text.squeeze(0)
    prediction = (self.fc1(fc_input))


    return prediction

gru_losses = []
bilstm_losses = []
lstm_losses = []
rnn_losses = []


gru_f1score = []
bilstm_f1score = []
lstm_f1score = []
rnn_f1score = []

print(len(lstm_losses))
print(len(lstm_f1score))

def train_classifier(model, dataset_iterator, loss_function, optimizer, \
                     num_epochs = 10, log = "runs", verbose = True, print_every = 100, recurrent = False, mode='xx'):

  model.train()
  step = 0
  predict_labels = []
  correct_labels = []

  for epoch in range(num_epochs):
    correct = 0
    total = 0
    total_loss = 0
    for batch in dataset_iterator:
      text,text_length = batch.Text
      labels = batch.Score

      batch_size = len(labels)

      if(torch.sum(text_length) < batch_size):
        continue


      for each in labels:
        correct_labels.append(each.item())
      
      labels = labels - 1

      optimizer.zero_grad()
      prediction = model.forward(text, text_length)
      y_pred = torch.argmax(prediction, dim=1)

      labels = labels.type_as(prediction)
      loss = loss_function(prediction,labels.long())
      loss.backward()
      optimizer.step()

      correct += (torch.sum(y_pred == labels)).item()

      for each in y_pred:
        predict_labels.append(each.item() + 1)


      total += len(labels)
      total_loss += (loss.item())

      if ((step % print_every) == 0):
        # writer.add_scalar("Loss/train", total_loss/total, step)
        
        if mode == 'lstm':
          lstm_losses.append(total_loss/total)
        if mode == 'bilstm':
          bilstm_losses.append(total_loss/total)          
        if mode == 'rnn':
          rnn_losses.append(total_loss/total)   
        if mode == 'gru':
          gru_losses.append(total_loss/total)   

        # writer.add_scalar("Acc/train", correct/total, step)
        if verbose:
          print("--- Step: %s Acc: %s Loss: %s" %(step, correct/total, total_loss/total))
      step = step+1

    print("Epoch: %s Acc: %s Loss: %s"%(epoch+1, correct/total, total_loss/total))
    
    f1score = f1_score(correct_labels, predict_labels, average='macro')
    print('f1score: ',f1score)
    
    if mode == 'lstm':
      lstm_f1score.append(f1score)
    if mode == 'bilstm':
      bilstm_f1score.append(f1score)
    if mode == 'rnn':
      rnn_f1score.append(f1score)
    if mode == 'gru':
      gru_f1score.append(f1score)

  



def evaluate_classifier(model, dataset_iterator, loss_function, mode, recurrent = False):
  model.eval()

  correct = 0
  total = 0
  total_loss = 0
  predictions = []
  actual = []
  for batch in dataset_iterator:
    text, text_length = batch.Text
    labels = batch.Score

    prediction = model.forward(text, text_length)
    y_pred = torch.argmax(prediction, dim=1)
    # labels = labels - 1

    for each in y_pred:
      predictions.append(each.item() + 1)    

    for each in labels:
      actual.append(each.item())          

  if mode == 'lstm':
    lstm_f1score.append(f1score)
  if mode == 'bilstm':
    bilstm_f1score.append(f1score)
  if mode == 'rnn':
    rnn_f1score.append(f1score)
  if mode == 'gru':
    gru_f1score.append(f1score)

  print('f1score: ',f1score)
  model.train()

def generate_predictions(model,data_iterator):

  model.eval()
  predictions = []
  for batch in data_iterator:
    text, text_length = batch.Text
    
    prediction = model.forward(text, text_length)
    y_pred = torch.argmax(prediction, dim=1)

    for each in y_pred:
      predictions.append(each.item() + 1)


  return (predictions)

output_size = 5
hidden_size = 256
vocab_size = len(TEXT.vocab)
embedding_length = 300
word_embeddings = TEXT.vocab.vectors
num_epochs = 10

model = ReviewClassifier('lstm',output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = model.to('cuda:1')
model = model.to('cuda:0')

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

log_dir = 'runs/recurrent'

train_classifier(model, train_iterator, loss_function, optimizer, log = log_dir, num_epochs = num_epochs, print_every = 100, recurrent = True,mode='DontAppend')

actual_scores_validation = train_df['Score'].tolist()
prediction_val = generate_predictions(model,train_iterator)
print("Training F1 score ", f1_score(actual_scores_validation, prediction_val, average='micro'))

plot_confusion_matrix(actual_scores_validation, prediction_val, [1,2,3,4,5], ['1','2','3','4','5'], title='Initial Confusion matrix for LSTM', tensor_name = 'MyFigure/image', normalize=False)


print("Accuracy score: ",accuracy_score(actual_scores_validation, prediction_val))
print("Precision score: ",precision_score(actual_scores_validation, prediction_val,average='micro'))
print("Recall score: ",recall_score(actual_scores_validation, prediction_val,average='micro'))
print("Balanced Accuracy score: ",balanced_accuracy_score(actual_scores_validation, prediction_val))


print(classification_report(actual_scores_validation, prediction_val, target_names=['1','2','3','4','5']))



target_names = ['1','2','3','4','5']


cm = confusion_matrix(actual_scores_validation, prediction_val)


#Now the normalize the diagonal entries
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


cm.diagonal()


# weights = [0.25, 0.30, 0.20 ,0.20 , 0.05]

# weights = [0.3,0.2,0.13,0.07,0.3]
# weights = [0.20987038694950402, 0.36800036350868515, 0.25585541935662426, 0.13612068751191228, 0.030153142673274437]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

# criterion = nn.CrossEntropyLoss()

output_size = 5
hidden_size = 256
vocab_size = len(TEXT.vocab)
embedding_length = 300
word_embeddings = TEXT.vocab.vectors
num_epochs = 1





log_dir = 'runs/recurrent'

model = ReviewClassifier('lstm',output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = model.to('cuda:1')
model = model.to('cuda:0')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_classifier(model, train_iterator, criterion, optimizer, log = log_dir, num_epochs = 5, print_every = 100, recurrent = True,mode='lstm')

train_classifier(model, train_iterator, criterion, optimizer, log = log_dir, num_epochs = 5, print_every = 100, recurrent = True,mode='lstm')


#actual_scores_validation = val_df['Score'].tolist()
actual_scores_validation = train_df['Score'].tolist()
#prediction_val = generate_predictions(model,test_iterator)
prediction_val = generate_predictions(model,train_iterator)
print("Tuned model validation F1 score ", f1_score(actual_scores_validation, prediction_val, average='macro'))

plot_confusion_matrix(actual_scores_validation, prediction_val, [1,2,3,4,5], ['1','2','3','4','5'], title='Tuned Model Confusion matrix for LSTM', tensor_name = 'MyFigure/image', normalize=False)

with open('lstm_train.real', 'w') as filehandle:
    for rating in actual_scores_validation:
        filehandle.write('%s\n' % rating)

with open('lstm_train.pred', 'w') as filehandle:
    for rating in prediction_val:
        filehandle.write('%s\n' % rating)



# RNN 


class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

rnn_model = ReviewClassifier('rnn',output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
rnn_model = rnn_model.to('cuda:0')
#rnn_model = rnn_model.to('cuda:1')
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-4,amsgrad=True)

train_classifier(rnn_model, train_iterator, criterion, optimizer, log = log_dir, num_epochs = 10, print_every = 100, recurrent = True,mode='rnn')


#actual_scores_validation = val_df['Score'].tolist()
actual_scores_validation = train_df['Score'].tolist()
#prediction_val = generate_predictions(rnn_model,test_iterator)
prediction_val = generate_predictions(rnn_model,train_iterator)
print("RNN validation F1 score ", f1_score(actual_scores_validation, prediction_val, average='macro'))

plot_confusion_matrix(actual_scores_validation, prediction_val, [1,2,3,4,5], ['1','2','3','4','5'], title='RNN Confusion matrix for LSTM', tensor_name = 'MyFigure/image', normalize=False)

actual_scores_validation = train_df['Score'].tolist()
prediction_val = generate_predictions(rnn_model,train_iterator)
print("Training F1 score ", f1_score(actual_scores_validation, prediction_val, average='micro'))

print("Accuracy score: ",accuracy_score(actual_scores_validation, prediction_val))
print("Precision score: ",precision_score(actual_scores_validation, prediction_val,average='micro'))
print("Recall score: ",recall_score(actual_scores_validation, prediction_val,average='micro'))
print("Balanced Accuracy score: ",balanced_accuracy_score(actual_scores_validation, prediction_val))


target_names = ['1','2','3','4','5']


cm = confusion_matrix(actual_scores_validation, prediction_val)

print(cm)

#Now the normalize the diagonal entries
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


cm.diagonal()

#GRU Train

output_size = 5
hidden_size = 256
vocab_size = len(TEXT.vocab)
embedding_length = 300
word_embeddings = TEXT.vocab.vectors
num_epochs = 10


log_dir = 'runs/recurrent'


class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

gru_model = ReviewClassifier('gru',output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#gru_model = gru_model.to('cuda:1')
gru_model = gru_model.to('cuda:0')
optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-4,amsgrad=True)

train_classifier(gru_model, train_iterator, criterion, optimizer, log = log_dir, num_epochs = num_epochs, print_every = 100, recurrent = True,mode='gru')

#GRU Eval


#actual_scores_validation = val_df['Score'].tolist()
actual_scores_validation = train_df['Score'].tolist()
#prediction_val = generate_predictions(gru_model,test_iterator)
prediction_val = generate_predictions(gru_model,train_iterator)
print("GRU validation F1 score ", f1_score(actual_scores_validation, prediction_val, average='macro'))

plot_confusion_matrix(actual_scores_validation, prediction_val, [1,2,3,4,5], ['1','2','3','4','5'], title='GRU Confusion matrix for LSTM', tensor_name = 'MyFigure/image', normalize=False)

actual_scores_validation = train_df['Score'].tolist()
prediction_val = generate_predictions(gru_model,train_iterator)
print("Training F1 score ", f1_score(actual_scores_validation, prediction_val, average='micro'))


print("Accuracy score: ",accuracy_score(actual_scores_validation, prediction_val))
print("Precision score: ",precision_score(actual_scores_validation, prediction_val,average='micro'))
print("Recall score: ",recall_score(actual_scores_validation, prediction_val,average='micro'))
print("Balanced Accuracy score: ",balanced_accuracy_score(actual_scores_validation, prediction_val))


target_names = ['1','2','3','4','5']


cm = confusion_matrix(actual_scores_validation, prediction_val)


#Now the normalize the diagonal entries
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


cm.diagonal()

#BiLSTM Train

output_size = 5
hidden_size = 256
vocab_size = len(TEXT.vocab)
embedding_length = 300
word_embeddings = TEXT.vocab.vectors
num_epochs = 10


log_dir = 'runs/recurrent'

class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

bilstm_model = ReviewClassifier('bilstm',output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#bilstm_model = bilstm_model.to('cuda:1')
bilstm_model = bilstm_model.to('cuda:0')
optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=1e-4,amsgrad=True)

train_classifier(bilstm_model, train_iterator, criterion, optimizer, log = log_dir, num_epochs = num_epochs, print_every = 100, recurrent = True, mode='bilstm')

actual_scores_validation = train_df['Score'].tolist()
prediction_val = generate_predictions(bilstm_model,train_iterator)
print("Training F1 score ", f1_score(actual_scores_validation, prediction_val, average='micro'))


print("Accuracy score: ",accuracy_score(actual_scores_validation, prediction_val))
print("Precision score: ",precision_score(actual_scores_validation, prediction_val,average='micro'))
print("Recall score: ",recall_score(actual_scores_validation, prediction_val,average='micro'))
print("Balanced Accuracy score: ",balanced_accuracy_score(actual_scores_validation, prediction_val))

target_names = ['1','2','3','4','5']


cm = confusion_matrix(actual_scores_validation, prediction_val)


#Now the normalize the diagonal entries
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


cm.diagonal()

# BiLSTM Eval


def generate_predictions_bilstm(model,data_iterator):

  model.eval()
  predictions = []
  for batch in data_iterator:
    text, text_length = batch.Text
    
    prediction = model.forward(text, text_length)

    y_pred = torch.argmax(prediction)

    predictions.append(y_pred.item() + 1)

  return (predictions)




#actual_scores_validation = val_df['Score'].tolist()
actual_scores_validation = train_df['Score'].tolist()
#prediction_val = generate_predictions_bilstm(bilstm_model,test_iterator)
prediction_val = generate_predictions_bilstm(bilstm_model,train_iterator)
print("BiLSTM validation F1 score ", f1_score(actual_scores_validation, prediction_val, average='macro'))

plot_confusion_matrix(actual_scores_validation, prediction_val, [1,2,3,4,5], ['1','2','3','4','5'], title='BiLSTM Confusion matrix for LSTM', tensor_name = 'MyFigure/image', normalize=False)


l1 = lstm_losses
l2 = rnn_losses
l3 = gru_losses
l4 = bilstm_losses

plt.plot(lstm_f1score, label = 'LSTM')
plt.plot(rnn_f1score, label = 'RNN')
plt.plot(gru_f1score, label = 'GRU')
plt.plot(bilstm_f1score, label = 'BiLSTM')
plt.xlabel('Epoch')
plt.ylabel('Training F1 Scores')
plt.legend(loc="upper right")
plt.show()


l1 = lstm_losses
l2 = rnn_losses
l3 = gru_losses
l4 = bilstm_losses


plt.plot(lstm_losses, label = 'LSTM')
plt.plot(rnn_losses, label = 'RNN')
plt.plot(gru_losses, label = 'GRU')
plt.plot(bilstm_losses, label = 'BiLSTM')
plt.xlabel('steps x 100')
plt.ylabel('Training Loss')
plt.legend(loc="upper right")
plt.show()