# This script was created based on the following tutorial
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

############################## Classifier: NN Model #################################
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNNclassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNclassifier, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size+hidden_size, hidden_size) # a linear layer that map input-states to hidden-states
        self.i2o = nn.Linear(input_size+hidden_size, output_size) # a linear layer that map hidden-states to output-states

        self.softmax = nn.LogSoftmax(dim=1) # a LogSoftmax layer after the output

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)



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
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = category_tensor.to(device)
    #change the sound into a tensor
    line_tensor = MFCCfeatures.sound2tensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn, criterion, optimizer, category_tensor, line_tensor):

    hidden = rnn.initHidden() #initialize hidden states to zeros
    rnn.zero_grad() #clear up gradients
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor) # Compare the final output of the RNNclassifier to the target
    loss.backward() # back-propagate

    optimizer.step()  # update the parameters

    return output, loss.item()


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

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            if guess == category:
                correct = '✓'
                train_correct += 1
            else:
                correct = '✗ (%s)' % category

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

            #modelpath = './models/'
            #if not os.path.exists(modelpath):
            #    os.makedirs(modelpath)
            #torch.save(rnn, './models/model_%i' % iter)

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
# To see how well the network performs on different categories,
    # we will create a confusion matrix,
    # indicating for eavy actual language (rows) which language the network guesses (columns).
        # To calculate the confusion matrix, a bunch of samples are run through the network with evaluate(),
        # evaluate() is the same as train() minus the backprop

def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def plotTrainPerform(rnn, category_lines, all_categories, n_categories = 2, n_confusion = 10000):

    confusion = torch.zeros(n_categories, n_categories)

    #Go through a bunch of examples and record which are correctly guessed
    guess_num = 0
    correct_num = 0
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output, all_categories)

        guess_num += 1
        if guess == category:
            correct_num += 1

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

    plt.savefig('./plots/rnn_confusion_plot_seed%s.png' % sys.argv[1])

    print('ACCURACY ON TRAINING DATA: %.2f%s' % (correct_num / guess_num * 100, '%'))
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






