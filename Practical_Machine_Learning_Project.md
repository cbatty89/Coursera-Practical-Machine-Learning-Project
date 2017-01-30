# Practical Machine Learning Project
Chris Batty  
January 30, 2017  



#Overview
Our goal for this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to attempt to identify the manner whether they were performing the exercise correctly or incorrectly.

# Data Processing

## Importing data
We were give two urls for obtaining the data.  The testing data was located at[Coursera: John Hopkins Practical Machine Learning Testing Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv), and the training data was located at [Coursera: John Hopkins Practical Machine Learning Training Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv).  After downloading the data the csv files were imported into R.





```r
training_model <- read.csv('pml-training.csv')
testing_for_prediction <-read.csv('pml-testing.csv')
```

## Cleaning Data
Since the data is unfamiliar the easiest way to get a basic understanding is to examine the structure.

```r
str(training_model)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "2/12/2011 13:32",..: 15 15 15 15 15 15 15 15 15 15 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.01685",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : Factor w/ 3 levels "","#DIV/0!","0": 1 1 1 1 1 1 1 1 1 1 ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

It can be seen from the structure that several of the columns have NA or blank values which we will address by eliminating columns that are 80% NA/blank, and also from the names of the variables that of the first 7 columns that user_name may be the only one to offer information in terms of what 'classe' the participants did the exercise and thus we will remove the others.

We will also create a training data set for the model creation, as well as a testing dataset to be able to examine accuracy.


```r
library(caret)
set.seed(2017)

training_model[training_model == ""] <- NA
training_model2 <- training_model[,colSums(is.na(training_model))/nrow(training_model) < .8]

training_modelfinal <-training_model2[,-c(1, 3:7)]

inTrain <- createDataPartition(y = training_modelfinal$classe,  p = 0.75, list = FALSE)

training <-training_modelfinal[inTrain,]
testing <-training_modelfinal[-inTrain,]
```


# Model Building

To attempt to find the best model possible we will construct a tree, linear discriminant analysis, random forest, and gradient boosted model and compare the accuracies of each.  We will also do 5-fold cross validation on all of the models except the random forest.


```r
library(randomForest)

mod_tree<- train(classe ~. , method = 'rpart', data = training, trControl = trainControl(method = 'cv', number= 5))

mod_lda <- train(classe ~., method = 'lda', data = training,  trControl = trainControl(method = 'cv', number= 5))

mod_forest <- randomForest(classe~., method = 'rf', data = training)

mod_boost <- train(classe~., method = 'gbm', data = training,  trControl = trainControl(method = 'cv', number= 5), verbose = FALSE)
```


Below we make predictions on our test set as well as create the confusion matrix for each model to examine it's accuracy.

```r
pred_t <- predict(mod_tree, testing)

pred_lda <- predict(mod_lda, testing)

pred_f <- predict(mod_forest, testing)

pred_b <- predict(mod_boost,testing)
```

###Tree Confusion Matrix

```r
confusionMatrix(pred_t, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1270  386  408  364  142
##          B   30  316   25  138  118
##          C   92  247  422  302  229
##          D    0    0    0    0    0
##          E    3    0    0    0  412
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4935          
##                  95% CI : (0.4794, 0.5076)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3376          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104  0.33298  0.49357   0.0000  0.45727
## Specificity            0.6295  0.92137  0.78513   1.0000  0.99925
## Pos Pred Value         0.4942  0.50399  0.32663      NaN  0.99277
## Neg Pred Value         0.9464  0.85200  0.88012   0.8361  0.89107
## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
## Detection Rate         0.2590  0.06444  0.08605   0.0000  0.08401
## Detection Prevalence   0.5241  0.12785  0.26346   0.0000  0.08462
## Balanced Accuracy      0.7700  0.62717  0.63935   0.5000  0.72826
```
### Linear Discriminant Analysis Confusion Matrix

```r
confusionMatrix(pred_lda, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1168  163   75   56   25
##          B   25  623   75   31  123
##          C  107   93  580   91   52
##          D   94   26  111  610   83
##          E    1   44   14   16  618
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7339          
##                  95% CI : (0.7213, 0.7462)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6629          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8373   0.6565   0.6784   0.7587   0.6859
## Specificity            0.9091   0.9358   0.9153   0.9234   0.9813
## Pos Pred Value         0.7855   0.7104   0.6284   0.6602   0.8918
## Neg Pred Value         0.9336   0.9190   0.9309   0.9513   0.9328
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2382   0.1270   0.1183   0.1244   0.1260
## Detection Prevalence   0.3032   0.1788   0.1882   0.1884   0.1413
## Balanced Accuracy      0.8732   0.7961   0.7968   0.8411   0.8336
```
### Random Forest Confusion Matrix

```r
confusionMatrix(pred_f, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    6    0    0    0
##          B    1  942    3    0    0
##          C    0    1  852    8    1
##          D    0    0    0  796    3
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9953        
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9941        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9926   0.9965   0.9900   0.9956
## Specificity            0.9983   0.9990   0.9975   0.9993   1.0000
## Pos Pred Value         0.9957   0.9958   0.9884   0.9962   1.0000
## Neg Pred Value         0.9997   0.9982   0.9993   0.9981   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1921   0.1737   0.1623   0.1829
## Detection Prevalence   0.2855   0.1929   0.1758   0.1629   0.1829
## Balanced Accuracy      0.9988   0.9958   0.9970   0.9947   0.9978
```
### Gradient Boosting Model Confusion Matrix

```r
confusionMatrix(pred_b, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1378   30    0    1    1
##          B   10  896   36    4    5
##          C    4   21  804   30    6
##          D    1    2   13  768   20
##          E    2    0    2    1  869
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9615          
##                  95% CI : (0.9557, 0.9667)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9512          
##  Mcnemar's Test P-Value : 2.834e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9878   0.9442   0.9404   0.9552   0.9645
## Specificity            0.9909   0.9861   0.9849   0.9912   0.9988
## Pos Pred Value         0.9773   0.9422   0.9295   0.9552   0.9943
## Neg Pred Value         0.9951   0.9866   0.9874   0.9912   0.9921
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2810   0.1827   0.1639   0.1566   0.1772
## Detection Prevalence   0.2875   0.1939   0.1764   0.1639   0.1782
## Balanced Accuracy      0.9893   0.9651   0.9626   0.9732   0.9816
```

As can be seen by looking at output of each confusion matrix, the most accurate model is the random forest with an accuracy of 99.53%.


#Prediction on Test Data
We conclude with predicting the classe of testing_for_prediction in our random tree model.


```r
pred_outcome <- predict(mod_forest, testing_for_prediction)

pred_outcome
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```



