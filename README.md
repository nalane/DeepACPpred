# DeepACPpred
This project is the subject of the paper "DeepACPpred: A Novel Hybrid CNN-RNN Architecture for Predicting Anti-Cancer Peptides" by Nathaniel Lane and Indika Kahanda. As a brief summary, anti-cancer peptides are a promising alternative to chemotherapy. This project seeks to make it easier for researchers to identify good ACP candidates. Our contribution is a prediction system that uses recurrent neural networks (RNNs) and convolutional neural networks (CNNs) on the sequence of amino acids in the peptide. The thinking is that the RNN should be able to consider interactions amongst distant amino acids, while the CNN can consider interactions amongst nearby amino acids. In that way, these structures are complementary. Regardless of whether or not this is the case, we have found that the system works better than either technique alone.

## Usage
The file that contains the bulk of interesting code in this project is Predictor.py. There are two ways to run it. If you would like to perform 10-fold cross validation on a single dataset, you can run

    python Predictor.py <dataset>
    
If you would rather train on one dataset and test on another, you can run the following:

    python Predictor.py <train_set> <test_set>
   
