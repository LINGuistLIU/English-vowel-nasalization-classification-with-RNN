# Vowel-nasalization-classification-with-RNN

This repository is the scripts we used for classifying English vowels as to whether they are nasalized or not in 

[Liu, L.; Hulden, M.; Scarborough, R. (2019). "RNN Classification of English Vowels: Nasalized or Not", *Proceedings of the Society for Computation in Linguistics*: Vol. 2, Article 36] (https://scholarworks.umass.edu/scil/vol2/iss1/36/)

## Datasets

The data used for this work are private collected by [Styler (2015)] (http://wstyler.ucsd.edu/files/styler_dissertation_final.pdf).

Only 10 vowels singled out of the word for the train, dev, and test sets respectively are provided here as examples. 

## Scripts

The following examples are all based on the vanilla RNN classifier. To run the LSTM classifier, just change the classifier name.
-------------------------------------------------------------

To tune hyper-parameters,

$ python src/dev_RNN_CVN.py \[HIDDEN-SIE\] \[ITER-NUM\] \[LEARNING-RATE\]

We also tuned the architecture of model by changing the script in the process of architecture engineer.

--------------------------------------------------------------
To test the performance, 

$ python src/test_RNN.py \[CONTEXT\]

\[NOTE: CONTEXT can be CVN, NVC, or NVN.\]
