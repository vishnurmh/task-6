K-Nearest Neighbors (KNN) Classification

In this task, I implemented the K-Nearest Neighbors (KNN) algorithm for classification using the dataset.

First, I loaded the dataset and separated the features and target variable. Since KNN is based on distance, I applied feature scaling using StandardScaler to normalize the data.

Then I split the dataset into training and testing sets. After that, I trained a KNN model with a chosen value of K.

I evaluated the model using accuracy score and confusion matrix. The confusion matrix helped in understanding how well the model is predicting each class.

I also experimented with different values of K such as 1, 3, 5, 7, and 9. By comparing their accuracy, I understood how the choice of K affects the performance of the model.

From this task, I learned how KNN works based on distance between data points and why normalization is important for this algorithm.

Tools used:
Python, Pandas, Scikit-learn
