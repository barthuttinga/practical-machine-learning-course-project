---
title: "Practical Machine Learning - Course Project"
author: "Bart Huttinga"
date: "27 december 2018"
output:
    html_document:
        keep_md: true
---

This GitHub Page is published at a custom domain, the source code can be found at [https://github.com/barthuttinga/practical-machine-learning-course-project](https://github.com/barthuttinga/practical-machine-learning-course-project).




## Loading and exploring the data

After first loading the data with `read.csv("pml-training.csv")` I noticed that there were many empty values in the data and quite a few error values (`#DIV/0!`). So, I reloaded the data using the command below, setting empty strings and `#DIV/0!` to NA.


```r
data <- read.csv("pml-training.csv", na.strings = c("", "#DIV/0!"))
```

Exploring the data I found that there were columns with lots of NA's. I decided to drop the columns that contain 95% or more NA's. This resulted in dropping 33 columns.


```r
# remove predictors that are > 95% NA
na_count <- apply(is.na(data), 2, sum)
na_features <- names(na_count[na_count / nrow(data) > .95])
data <- data %>% select(-c(na_features))
```

Variables with almost no variability are not useful for building a prediction model, so I removed all near zero variance predictors using the code below. This resulted in dropping another 68 columns.


```r
# remove near zero variance predictors
near_zero_table <- nearZeroVar(data, saveMetrics=TRUE)
near_zero_table$feature <- row.names(near_zero_table)
near_zero_vars <- near_zero_table %>% filter(nzv == TRUE)
data <- data %>% select(-c(near_zero_vars$feature))
```

To my judgement, the observation identifiers, usernames, and timestamps are of no value for construction the prediction model, so I removed these as well.


```r
# remove observation numbers, usernames, and timestamps
data <- data %>% select(-c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp"))
```

The dimensions of the dataset now were:


```r
dim(data)
```

```
## [1] 19622    54
```

## Partitioning and pre-processing the data

In order to be able to test the model against unseen data, I cut up the dataset in a training set and a testing set.


```r
# create train and test sets
inTrain <- createDataPartition(y = data$classe, p = 0.7, list = FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
```

I tried performing PCA with the code below, in order to compress the data. But this method seemed to result in models with lower accuracy, so I didn't use this method in the models presented below.


```r
# perform PCA on all predictors ("-54" because 54th = outcome)
preProc <- preProcess(training[, -54], method = "pca", thresh = .95)
training_PC <- predict(preProc, training)
```

## Cross-validation

To prevent over-fitting the model towards the training data I choose to use 10-fold cross-validation for training the data.


```r
# set train control to 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)
```

## Fitting the model

After exploring different options and methods, I fit four different models as candidates for the final model using linear discriminant analysis (LDA), decision tree algorithm, random forest algorithm, and a stacked model combining the three.


```r
fit_lda <- train(classe ~ ., data = training, method = "lda", trControl = train_control)
pred_lda <- predict(fit_lda, testing)
confusionMatrix(fit_lda, reference = pred_lda)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 23.4  2.7  1.7  1.0  0.8
##          B  0.8 12.4  1.6  0.7  2.9
##          C  1.8  2.6 11.6  1.9  1.7
##          D  2.3  0.7  2.1 12.3  1.8
##          E  0.1  0.8  0.4  0.6 11.3
##                             
##  Accuracy (average) : 0.7102
```

```r
fit_rpart <- train(classe ~ ., data = training, method = "rpart", trControl = train_control)
pred_rpart <- predict(fit_rpart, testing)
confusionMatrix(fit_rpart, reference = pred_rpart)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 23.8  4.3  1.9  2.6  1.4
##          B  0.5  6.7  0.6  2.9  2.4
##          C  4.1  8.4 15.0 10.8  6.1
##          D  0.0  0.0  0.0  0.0  0.0
##          E  0.1  0.0  0.0  0.1  8.5
##                             
##  Accuracy (average) : 0.5404
```

```r
fit_rf <- train(classe ~ ., data = training, method = "rf", trControl = train_control, ntree = 20)
pred_rf <- predict(fit_rf, testing)
confusionMatrix(fit_rf, reference = pred_rf)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.2  0.1  0.0  0.0
##          C  0.0  0.0 17.4  0.1  0.0
##          D  0.0  0.0  0.0 16.2  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9961
```

```r
training_stacked <- data.frame(classe = training$classe,
                      lda = predict(fit_lda, training),
                      rpart = predict(fit_rpart, training),
                      rf = predict(fit_rf, training))
fit_stacked <- train(classe ~ ., data = training_stacked, method = "rf",
                     trControl = train_control, ntree = 20)
testing_stacked <- data.frame(classe = testing$classe, lda = pred_lda,
                              rpart = pred_rpart, rf = pred_rf)
pred_stacked <- predict(fit_stacked, testing_stacked)
confusionMatrix(fit_stacked, reference = pred_stacked)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.0  0.0
##          D  0.0  0.0  0.0 16.4  0.0
##          E  0.0  0.0  0.0  0.0 18.4
##                        
##  Accuracy (average) : 1
```

## Final model

As we can see in the results above the model using a random forest algorithm has an expected out-of-sample accuracy of more than 99%. When combining the first, second, and third model in a stacked/ensembled model using a random forest algorithm an out-of-sample accuracy of 100% can be expected. Conclusion: I would select the stacked/ensembled model to predict with.
