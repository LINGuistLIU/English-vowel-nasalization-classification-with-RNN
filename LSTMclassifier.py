# This script was created based on the following tutorials
## https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# https://github.com/yuchenlin/lstm_sentence_classifier

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import random
import sys
import os

import MFCCfeatures


class LSTMclassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMclassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.hidden = self.initHidden(1)

    def initHidden(self, batch_size=1):
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda(),
                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)),
                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)))

    def forward(self, docs, batched=False):
        #docs is a MFCC matrix
        # Lookup the embeddings for each word.
        #embeds = self.word_embeddings(docs)
        embeds = docs
        # Run the LSTM
        if batched:
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        else:
            lstm_out, self.hidden = self.lstm(embeds.view(len(docs), 1, -1), self.hidden)

        # Compute score distribution
        category_space = self.hidden2label(lstm_out[-1])
        category_scores = F.log_softmax(category_space, dim=1)

        return category_scores


################################### Training the model #################################

#### Prepare for training

def categoryFromOutput(output, all_categories):
    '''
    This function interprets the output of the network.
    The output of the network is a likelihood of each category.
    We can use Tensor.topk to get the index of the greatest value.
    '''

    top_n, top_i = output.topk(1) #Tensor.topk gives us the index of the greatest value
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(c):
    return c[random.randint(0, len(c) - 1)]

def randomTrainingExample(all_categories, category_lines):
    '''
    This function picks a random training example (a name and its category/language) quickly.
    '''

    category = randomChoice(all_categories) #pick a category/language randomly
    line = randomChoice(category_lines[category]) #pick a name in that category/language randomly
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) #change the language category into a tensor
    if torch.cuda.is_available():
        category_tensor = category_tensor.cuda()
    #change the sound into a tensor
    line_tensor = MFCCfeatures.sound2tensor(line)
    return category, line, category_tensor, line_tensor

#### Train the model

def train(rnn, criterion, optimizer, category_tensor, line_tensor):
    rnn.hidden = rnn.initHidden() #initialize hidden states to zeros
    rnn.zero_grad() #clear up gradients
    category_scores = rnn(line_tensor)
    loss = criterion(category_scores, category_tensor) # Compare the final output of the RNNclassifier to the target
    loss.backward() # back-propagate to compute the gradients
    optimizer.step() # update the parameters
    return category_scores, loss.item()

def trackTraining(rnn, criterion, optimizer, all_categories, category_lines, n_iters = 100000, print_every = 5000, plot_every = 1000):

    current_loss = 0
    all_losses = []
    train_sum = 0
    train_correct = 0

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output, loss = train(rnn, criterion, optimizer, category_tensor, line_tensor)
        current_loss += loss
        train_sum += 1

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            if guess == category:
                correct = '✓'
                train_correct += 1
            else:
                correct = '✗ (%s)' % category

        #Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

            # Save the model every time we track the loss, overwriting the last each time.
            #modelpath = 'models/'
            #if not os.path.exists(modelpath):
            #    os.makedirs(modelpath)
            #torch.save(rnn, modelpath+'LSTM_model_%i' % iter)

    print("Training accuracy: %.4f" % (train_correct / train_sum))

    return all_losses

#### Plot the training process
## Plotting the historical loss from all_losses shows the network learning


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plotLosses(all_losses, pltname):
    plt.figure()
    plt.plot(all_losses)
    plt.savefig(pltname)
    return


################################### Evaluate the Result #################################
def evaluate(rnn, line_tensor):

    hidden = rnn.initHidden()
    output = rnn(line_tensor)
    return output

def plotTrainPerform(rnn, category_lines, all_categories, n_categories = 2, n_confusion = 10000):

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)

    train_guess = 0
    train_correct = 0
    #Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output, all_categories)

        train_guess += 1
        if guess == category:
            train_correct += 1

        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(['']+all_categories, rotation=90)
    ax.set_yticklabels(['']+all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('./plots/LSTM_confusion_plot_seed%s.png' % sys.argv[1])

    print("ACCURACY ON THE TRAINING SET: %d / %d = %.4f" % (train_correct, train_guess, train_correct/train_guess))

    return


######################### Running on User Input ###########################
def predict(rnn, input_line, all_categories, n_predictions=1):
    with torch.no_grad():
        output = evaluate(rnn, MFCCfeatures.sound2tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append([value, all_categories[category_index]])
    return all_categories[category_index]














