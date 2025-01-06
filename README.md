# Text Classification Based on the IMDb Dataset

## Structure

```
.
|-- BiRNN.py                        # Implementation by BiRNN 
|-- LR.py                           # Implementation by Logistic Regression
|-- MultinomialNB.py                # Implementation by CNN
|-- TextCNN.py                      # Implementation by Naive Bayes
|-- TokenEmbedding.py               # To embed tokens for BiRNN and CNN
|-- dataset.py                      # To deal with IMDb dataset
|-- download.py                     # Download IMDb
|-- model.py                        # Models like BiRNN, LR, CNN
|-- split_aclimdb_dataset.py        # split IMDb dataset into training set, validation set, test set
`-- tfidf.py                        # tfidf function
```

## Requirement

- python 3.8 or 3.9
- torch 2.5.1+cu121

## Run

You can run `python BiRNN.py`, `python LR.py`, `python MultinomialNB.py`, `python TextCNN.py` to train and test.