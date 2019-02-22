import MFCCfeatures
import RNNclassifier
import torch
import torch.nn as nn
import random
import torch.optim as optim
import sys
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


def parameterTune(n_hidden, n_iters, learning_rate, plotpath, n_letters, n_categories, all_categories, category_lines, seedNum=10):
    overallAccuracyList = []

    oralPrecisionList = []
    oralRecallList = []
    oralF1List = []

    nasalPrecisionList = []
    nasalRecallList = []
    nasalF1List = []

    nasal_file_pred_dict = defaultdict(list)
    oral_file_pred_dict = defaultdict(list)

    for seed in range(1, seedNum+1):
        print('Seed ', seed)
        torch.manual_seed(seed)
        rnn = RNNclassifier.RNNclassifier(n_letters, n_hidden, n_categories)
        if torch.cuda.is_available():
            rnn.cuda()

        criterion = nn.NLLLoss() #NLLLoss -- negative log likelihood loss
        if torch.cuda.is_available():
            criterion.cuda()


        optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

        print("Training models ... ... ... ")

        # To train the model
        all_losses = RNNclassifier.trackTraining(rnn, criterion, optimizer, all_categories, category_lines, n_iters, print_every = 1000, plot_every = 100)
        print('all_losses\t', all_losses)
        RNNclassifier.plotLosses(all_losses, plotpath+'test-RNN_TrainingLoss_plot_seed'+str(seed)+'.pdf')

        # To test the performance of the trained model on new data, i.e. test data
        test_guess_nasal = 0
        test_correct_nasal = 0
        actualNasals = 0.0
        predictNasals = 0.0
        predictOrals = 0.0
        predictNasalsTrue = 0.0
        predictNasalsFalse = 0.0
        for soundfile in test_cfdict['NASAL']:
            test_guess_nasal += 1
            actualNasals += 1
            pred = RNNclassifier.predict(rnn, soundfile, all_categories, n_predictions=1)
            nasal_file_pred_dict[soundfile].append(pred)
            if pred == 'NASAL':
                test_correct_nasal += 1
                predictNasals += 1
                predictNasalsTrue += 1
            else:
                predictNasalsFalse += 1
                predictOrals += 1

        test_guess_non = 0
        test_correct_non = 0
        actualOrals = 0.0
        #predictOrals = 0.0
        predictOralsTrue = 0.0
        predictOralsFalse = 0.0
        for soundfile in test_cfdict['ORAL']:
            test_guess_non += 1
            actualOrals += 1
            pred = RNNclassifier.predict(rnn, soundfile, all_categories, n_predictions=1)
            oral_file_pred_dict[soundfile].append(pred)
            if pred == 'ORAL':
                test_correct_non += 1
                predictOrals += 1
                predictOralsTrue += 1
            else:
                predictOralsFalse += 1
                predictNasals += 1

        testsize = actualOrals + actualNasals
        predictTrue = predictOralsTrue + predictNasalsTrue
        predictFasle = predictOralsFalse + predictNasalsFalse
        overallAccuracy = round(predictTrue/testsize, 4)

        oralPrecision = round(predictOralsTrue/predictOrals, 4)
        oralRecall = round(predictOralsTrue/actualOrals, 4)
        oralF1 = round(MFCCfeatures.getF1score(oralPrecision, oralRecall), 4)

        nasalPrecision = round(predictNasalsTrue/predictNasals, 4)
        nasalRecall = round(predictNasalsTrue/actualNasals, 4)
        nasalF1 = round(MFCCfeatures.getF1score(nasalPrecision, nasalRecall), 4)

        print('Overall accuracy: %.4f' % (predictTrue/testsize))

        print('Oral precision: %.4f' % oralPrecision)
        print('Oral recall:%.4f' % oralRecall)
        print('Oral F1: %.4f' % oralF1)

        print('Nasal precision: %.4f' % nasalPrecision)
        print('Nasal recall: %.4f' % nasalRecall)
        print('Nasal F1: %.4f' % nasalF1)

        print('\n-------------------------------------')
        print('actualNasals: ', actualNasals)
        print('actualOrals: ', actualOrals)
        print('predNasals: ', predictNasals)
        print('predOrals: ', predictOrals)
        print('predNasalTrue: ', predictNasalsTrue)
        print('predNasalFalse: ', predictNasalsFalse)
        print('predOralsTrue: ', predictOralsTrue)
        print('predOralsFalse: ', predictOralsFalse)
        print('---------------------------------------\n')

        overallAccuracyList.append(overallAccuracy)

        oralPrecisionList.append(oralPrecision)
        oralRecallList.append(oralRecall)
        oralF1List.append(oralF1)

        nasalPrecisionList.append(nasalPrecision)
        nasalRecallList.append(nasalRecall)
        nasalF1List.append(nasalF1)


    vote_actualNasal = 0.0
    vote_actualOral = 0.0
    vote_predNasal = 0.0
    vote_predOral = 0.0
    vote_predNasalTrue = 0.0
    vote_predNasalFalse = 0.0
    vote_predOralTrue = 0.0
    vote_predOralFalse = 0.0

    nasalBestPredDict = {}
    for soundfile in nasal_file_pred_dict.keys():
        vote_actualNasal += 1
        pred = MFCCfeatures.majorityVote(nasal_file_pred_dict[soundfile])
        nasalBestPredDict[soundfile] = pred
        if pred == 'NASAL':
            vote_predNasal += 1
            vote_predNasalTrue += 1
        else:
            vote_predOral += 1
            vote_predNasalFalse += 1
    oralBestPredDict = {}
    for soundfile in oral_file_pred_dict.keys():
        vote_actualOral += 1
        pred = MFCCfeatures.majorityVote(oral_file_pred_dict[soundfile])
        oralBestPredDict[soundfile] = pred
        if pred == 'ORAL':
            vote_predOral += 1
            vote_predOralTrue += 1
        else:
            vote_predNasal += 1
            vote_predOralFalse += 1

    vote_overallAccuracy = round((vote_predNasalTrue+vote_predOralTrue)/(vote_actualNasal+vote_actualOral), 4)

    vote_oralPrecision = round(vote_predOralTrue/vote_predOral, 4)
    vote_oralRecall = round(vote_predOralTrue/vote_actualOral, 4)
    vote_oralF1 = round(MFCCfeatures.getF1score(vote_oralPrecision, vote_oralRecall), 4)

    vote_nasalPrecision = round(vote_predNasalTrue/vote_predNasal, 4)
    vote_nasalRecall = round(vote_predNasalTrue/vote_actualNasal, 4)
    vote_nasalF1 = round(MFCCfeatures.getF1score(vote_nasalPrecision, vote_nasalRecall), 4)
    print('\n-------------------------------------')
    print('vote_actualNasal: ', vote_actualNasal)
    print('vote_actualOral: ', vote_actualOral)
    print('vote_predNasal: ', vote_predNasal)
    print('vote_predOral: ', vote_predOral)
    print('vote_predNasalTrue: ', vote_predNasalTrue)
    print('vote_predNasalFalse: ', vote_predNasalFalse)
    print('vote_predOralTrue: ', vote_predOralTrue)
    print('vote_predOralFalse: ', vote_predOralFalse)
    print('---------------------------------------\n')

    return overallAccuracyList, \
           oralPrecisionList, oralRecallList, oralF1List, \
           nasalPrecisionList, nasalRecallList, nasalF1List, \
           vote_overallAccuracy, \
           vote_oralPrecision, vote_oralRecall, vote_oralF1, \
           vote_nasalPrecision, vote_nasalRecall, vote_nasalF1, \
           oral_file_pred_dict, nasal_file_pred_dict, \
           nasalBestPredDict, oralBestPredDict

def confusionMatrixPlot(oralBestPredDict, nasalBestPredDict, plotname, datatype):
    labels = ['NASAL', 'ORAL']
    goldlabels = ['ORAL']*len(oralBestPredDict)+['NASAL']*len(nasalBestPredDict)
    predlabels = [oralBestPredDict[k] for k in oralBestPredDict.keys()] + [nasalBestPredDict[k] for k in nasalBestPredDict.keys()]
    cm = confusion_matrix(goldlabels, predlabels, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of RNN classifier '+datatype)
    fig.colorbar(cax)
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(plotname)
    #plt.show()
    plt.close()
    return


if __name__ == '__main__':

    print("Preprocessing data ... ... ... ")

    cvn_traindata_nasal = 'data/all/diss_en/train/CVN/'
    cvn_testdata_nasal = 'data/all/diss_en/test/CVN/' #test data
    nvc_traindata_nasal = 'data/all/diss_en/train/NVC/'
    nvc_testdata_nasal = 'data/all/diss_en/test/NVC/'
    nvn_traindata_nasal = 'data/all/diss_en/train/NVN/'
    nvn_testdata_nasal = 'data/all/diss_en/test/NVN/'

    traindata_non = 'data/all/diss_en/train/CVC/'
    testdata_non = 'data/all/diss_en/test/CVC/' #test data

    cxt = sys.argv[1] # NVN, CVN, or NVC
    if cxt == 'NVN':
        traindata_nasal = nvn_traindata_nasal
        testdata_nasal = nvn_testdata_nasal
    if cxt == 'CVN':
        traindata_nasal = cvn_traindata_nasal
        testdata_nasal = cvn_testdata_nasal
    if cxt == 'NVC':
        traindata_nasal = nvc_traindata_nasal
        testdata_nasal = nvc_testdata_nasal

    nasal_trainfiles = MFCCfeatures.fileNamelist(traindata_nasal)
    nasal_testfiles = MFCCfeatures.fileNamelist(testdata_nasal)
    oral_trainfiles = MFCCfeatures.fileNamelist(traindata_non)
    oral_testfiles = MFCCfeatures.fileNamelist(testdata_non)

    trainfiles = nasal_trainfiles + oral_trainfiles
    testfiles = nasal_testfiles + oral_testfiles

    train_cfdict = MFCCfeatures.file2category(trainfiles, nasal_trainfiles)
    test_cfdict = MFCCfeatures.file2category(testfiles, nasal_testfiles)

    all_categories = ['NASAL', 'ORAL']
    category_lines = train_cfdict

    n_categories = len(all_categories) #i.e. n_categories = 2
    n_letters = 13 #n_letters is the size of the vector/tensor

    n_hidden = 300 # hidden size
    n_iters = 40000 # number of iterations
    learning_rate = 0.0005 # learning rate

    plotpath = 'plots/RNN/' # path for averaged loss plots
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)
    overallAccuracyList, \
    oralPrecisionList, oralRecallList, oralF1List, \
    nasalPrecisionList, nasalRecallList, nasalF1List, \
    vote_overallAccuracy, \
    vote_oralPrecision, vote_oralRecall, vote_oralF1, \
    vote_nasalPrecision, vote_nasalRecall, vote_nasalF1, \
    oral_file_pred_dict, nasal_file_pred_dict, \
    nasalBestPredDict, oralBestPredDict \
    = parameterTune(n_hidden, n_iters, learning_rate, plotpath, n_letters = n_letters, n_categories=n_categories, all_categories=all_categories, category_lines=category_lines, seedNum=10)

    #plot the result
    datatype = 'CVC-vs-'+cxt
    plotname = plotpath + 'testCM_RNN_'+datatype + '.pdf' # name of the confusion matrix plot
    confusionMatrixPlot(oralBestPredDict, nasalBestPredDict, plotname, datatype)

    print('---------------Nasal-------------------')
    for item in nasal_file_pred_dict.keys():
        print(item, '\t', nasal_file_pred_dict[item])
    print('----------------Oral------------------')
    for item in oral_file_pred_dict.keys():
        print(item, '\t', oral_file_pred_dict[item])
    print('--------------------------------------')

    print('Overall Accuracy: ', float(sum(overallAccuracyList))/len(overallAccuracyList))
    print(overallAccuracyList, '\n')

    print('Oral precision: ', float(sum(oralPrecisionList))/len(oralPrecisionList))
    print(oralPrecisionList)
    print('Oral recall: ', float(sum(oralRecallList))/len(oralRecallList))
    print(oralRecallList)
    print('Oral F1: ', float(sum(oralF1List))/len(oralF1List))
    print(oralF1List, '\n')

    print('Nasal precision: ', float(sum(nasalPrecisionList))/len(nasalPrecisionList))
    print(nasalPrecisionList)
    print('Nasal recall: ', float(sum(nasalRecallList))/len(nasalRecallList))
    print(nasalRecallList)
    print('Nasal F1: ', float(sum(nasalF1List))/len(nasalF1List))
    print(nasalF1List, '\n')

    print('Vote overall accuracy: ', vote_overallAccuracy, '\n')

    print('Vote oral precision: ', vote_oralPrecision)
    print('Vote oral recall: ', vote_oralRecall)
    print('Vote oral F1: ', vote_oralF1, '\n')

    print('Vote nasal precision: ', vote_nasalPrecision)
    print('Vote nasal recall: ', vote_nasalRecall)
    print('Vote nasal F1: ', vote_nasalF1)


