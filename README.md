# -NLP-Duplicate-Classification-on-Quora-Question-Pairs

Based on a Kaggle.com competition problem, the project is meant to develop an optimization algorithm to predict which of the pairs of questions have the same meaning on Quora.com. Since we have technical and daily problems all the time, one of the solutions is to find some relative questions in Quora and then find the best answer. However, there are lots of similar questions on the website come up with by different people and it is difficult to locate the best answer we want. As a result, finding the similar question pairs and putting them in the same category will make the answerers and other users easier to locate the most suitable questions and answers.

Reference: https://www.kaggle.com/c/quora-question-pairs/overview

We are planning to use techniques of natural language processing to extract features from the data, such as length of questions, difference of length, number of capital letters, question marks and indicators for question starting with "Are", "Can", "How", etc.

The features will be processed with a logistic regression model, to improve this, we may need to apply a more advanced algorithm such as building a neural network, adding different features to make the model more precise and accurate.
