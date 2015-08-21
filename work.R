## Working file. Final product will be available as Rmd. This is just to keep a history of what was done.
# libraries needed
library(caret)
library(dplyr)
library(randomForest)
library(rattle)
library(tree)
set.seed(1234)


# loading (change to url file load for project)
# note #DIV/0! is an NA String in the set.
lifts <- read.csv("./data/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
test <- read.csv("./data/pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))


# examine percent na by column


# after reading this post: https://class.coursera.org/predmachlearn-031/forum/thread?thread_id=97
# on handling NAs, I will be removing columns with greater than 95% NAs.

# get column names below threshold
threshold <- 0.95 * nrow(lifts)
keep <- colnames(lifts[,colSums(is.na(lifts)) <= threshold])
lifts <- lifts %>% select(one_of(keep))
nzv <- nearZeroVar(lifts, saveMetrics = TRUE)
keep <- rownames(nzv[nzv$nzv == FALSE,])
lifts <- lifts %>% select(one_of(keep))

# eyeballing the data, it appears that the first five columns have no rational predictive values
# col1: X - is the row number
# col2: user_name, self explanatory
# col3-5: form parts of a timestamp
# col6: num_window - there was no description of this variable but it appears to increase with rownum and offer no predictive ability as it doesn't appear to contain accellerometer data.
lifts <- lifts %>% select(7:59)

inTrain <- createDataPartition(y=lifts$classe, p=0.75, list=FALSE)
training <- lifts[inTrain,]
testing <- lifts[-inTrain,]

# first we'll examine a tree model
modelFit1 <- train(training$classe ~ ., method="rpart", data=training)
confusionMatrix(testing$classe, predict(modelFit1, testing))

fancyRpartPlot(modelFit1$finalModel)

# accuracy is pretty low at .049

#lets look at the data for correlations of predictors

# note the classe is the last column
# check for highly correlated variables
M <- abs(cor(training[,-53]))
diag(M) <- 0
which(M > 0.8, arr.ind=T)
# several of the variables are highly correlated supporting the use of PCA


modelFit2 <- train(training$classe ~ ., method="rf", preProcess="pca", data=training, allowParallel = TRUE)
confusionMatrix(testing$classe, predict(modelFit2, testing))



# finally I'll examine a random forest model without PCA to see how much accuracy is lost by 
# the variable reduction. I'll also do fewer than the default 25 folds to try to decrease the processing
# time a little as the last model took a significant amount of time to run
modelFit3 <- train(training$classe ~ ., method="rf", data=training, allowParallel = TRUE,
                   trControl = trainControl(method = "cv", number = 10))
confusionMatrix(testing$classe, predict(modelFit3, testing))
## improvement is negligible but we'll keep it as our final model.

varImpPlot(modelFit3$finalModel, sort=TRUE, main="Variable Importance: Final Model")

## predict the 20 test cases
pred_test <- predict(modelFit3, test)
pred_test

pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
answers <- c("B", "A", "B", "A", "A", "E", "D", "B", "A", "A", "B", "C", "B", "A", "E", "E",
             "A", "B", "B", "B")

pml_write_files(answers)
