##############################
### data setup and exploration
##############################

# suppress warnings
oldw <- getOption("warn")
options(warn = -1)

# install packages (if not already installed)
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kknn)) install.packages("kknn", repos = "http://cran.us.r-project.org")

# libraries
library(ggplot2)
library(tidyverse)
library(caret)
library(matrixStats)
library(nnet)          # multinom model
library(randomForest)  # rf model
library(kknn)          # kknn model

# global variables
numberOfDigits <- 5
options(digits = numberOfDigits)
proportionTestSet <- 0.20
control <- trainControl(method = "cv",   # 10-fold cross validation configuration
                        number = 10, 
                        p = .9, 
                        allowParallel = TRUE)

# global functions
plot_optimization_chart <- function(model, chart_title){
  m <- model %>% 
    ggplot(highlight = TRUE) +
    ggtitle(chart_title)
  print(m)
  model$bestTune
  model$finalModel
}

# read dataset from csv
# source of dataset: https://www.kaggle.com/yasserh/wine-quality-dataset
wineDF <- read_csv(file = "./dat/WineQT.csv") %>% as_tibble()
glimpse(wineDF)

# remove column "Id" (not used) and make outcome "quality" as first column
wineDF <- wineDF %>% select(-Id) %>% relocate(quality)
head(wineDF)

# on column names, replace spaces by underline "_"
names(wineDF) <- gsub(" ", "_", names(wineDF))

# check for NA's
sum(is.na(wineDF))

# display the cleaned dataset
head(wineDF)

# split validation set for the final step (20%)
set.seed(1, sample.kind="Rounding")
testIndex <- createDataPartition(y = wineDF$quality, 
                                 times = 1, 
                                 p = proportionTestSet, 
                                 list = FALSE)
validationSet <- wineDF[testIndex,]

# from the remaining dataset, split train and test sets (80 / 20%)
mainSet <- wineDF[-testIndex,]

set.seed(2, sample.kind="Rounding")
testIndex <- createDataPartition(y = mainSet$quality, 
                                 times = 1, 
                                 p = proportionTestSet, 
                                 list = FALSE)
testSet  <- mainSet[testIndex,]
trainSet <- mainSet[-testIndex,]

# check for the different wine quality classes on the trainSet
qualityClasses <- unique(trainSet$quality) %>% sort()
qualityClasses

# check for stratification of train / test split
p1 <- trainSet %>%
  group_by(quality) %>%
  summarize(qty = n()) %>%
  mutate(split = 'trainSet')

p2 <- testSet %>%
  group_by(quality) %>%
  summarize(qty = n()) %>%
  mutate(split = 'testSet')

p <- bind_rows(p1, p2) %>% group_by(split)
p %>% ggplot(aes(quality, qty, fill = split)) +
  geom_bar(stat="identity", position = "dodge") +
  ggtitle("Stratification of Testset / Trainset split") +
  scale_x_continuous(breaks = qualityClasses)

# variability analysis of predictors
X <- trainSet %>% select(-quality) %>% as.matrix()
sX <- colSds(X)
sX <- bind_cols(colnames(X), sX) %>% as_tibble()
colnames(sX) <- c("parameter", "stdev")

sX %>% 
  arrange(desc(stdev)) %>%    # First sort by val.
  mutate(parameter=factor(parameter, levels=parameter)) %>%   # This trick update the factor levels
  ggplot(aes(parameter, stdev)) +
  geom_bar(stat= "identity") +
  geom_text(aes(label=sprintf("%0.2f", stdev)), size=3, vjust=-0.8) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size=10)) +
  ggtitle("Ranking of predictors by variability")

# remove predictors with small variance
nzv <- trainSet %>%
  select(-quality) %>%
  nearZeroVar()
removedPredictors <- colnames(trainSet[,nzv])
removedPredictors

trainSet <- trainSet %>% select(-all_of(removedPredictors))

# clean for temporary data
rm(mainSet, testIndex, wineDF, X, p, p1, p2, sX)
rm(nzv, removedPredictors)

# change outcome to factor
trainSet$quality <- as_factor(trainSet$quality)
testSet$quality  <- as_factor(testSet$quality)
validationSet$quality <- as_factor(validationSet$quality)



#######################
### naive average model
#######################

# training
mu <- mean(as.numeric(trainSet$quality))
mu <- round(mu)

# predicting
N <- length(testSet$quality)
predicted <- replicate(N, mu) %>% factor(levels = qualityClasses)

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = testSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- tibble(model    = "naive",
                          accuracy = acc)
accuracyResults



##################################
# penalized multinomial regression
##################################

# training
set.seed(3, sample.kind="Rounding")
mNomFit <- trainSet %>% train(quality ~ ., 
                              method = "multinom", 
                              data = .,
                              trace = FALSE,   # suppress printing of iterations
                              tuneGrid = data.frame(decay = seq(0.01, 0.5, 0.01)),
                              trControl = control
)
plot_optimization_chart(mNomFit, "Optimization for multinom model")

# predicting
predicted <- testSet %>% predict(mNomFit, newdata = .)

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = testSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- bind_rows(accuracyResults, tibble(model ="multinom",
                                                     accuracy = acc))
accuracyResults



#####################
# random forest model
#####################

# training
set.seed(3, sample.kind="Rounding")
rfFit <- trainSet %>% train(quality ~ .,
                            method = "rf",
                            data = .,
                            tuneGrid = data.frame(mtry = seq(3, 7, 1)),
                            trControl = control)
plot_optimization_chart(rfFit, "Optimization for random forest")

# predicting
predicted <- testSet %>% predict(rfFit, newdata = ., type = "raw")

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = testSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- bind_rows(accuracyResults, tibble(model ="rf",
                                                     accuracy = acc))
accuracyResults



##################################
# knn
##################################

# training
set.seed(3, sample.kind="Rounding")
knnFit <- trainSet %>% train(quality ~ .,
                             method = "knn",
                             data = .,
                             tuneGrid = data.frame(k = seq(3, 71, 2)),
                             trControl = control
)
plot_optimization_chart(knnFit, "Optimization for knn")

# predicting
predicted <- testSet %>% predict(knnFit, newdata = ., type = "raw")

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = testSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- bind_rows(accuracyResults, tibble(model ="knn",
                                                     accuracy = acc))
accuracyResults



##########################
# kknn / all models search
##########################

# training
set.seed(3, sample.kind="Rounding")
kknnFit <- trainSet %>% train(quality ~ .,
                              method = "kknn",
                              data = .,
                              tuneGrid = data.frame(kmax = seq(3, 41, 2),
                                                    distance = c(1, 2),
                                                    kernel = c("rectangular", 
                                                               "triangular",
                                                               "epanechnikov",
                                                               "biweight",
                                                               "tri-weight",
                                                               "cos",
                                                               "inv",
                                                               "gaussian",
                                                               "rank",
                                                               "optimal"
                                                    )),
                              trControl = control)
plot_optimization_chart(kknnFit, "Optimization for kknn / models search")

# predicting
predicted <- testSet %>% predict(kknnFit, newdata = ., type = "raw")

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = testSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- bind_rows(accuracyResults, tibble(model ="kknn/all models",
                                                     accuracy = acc))
accuracyResults



##################
# kknn / inv model
##################

# training
set.seed(3, sample.kind="Rounding")
kknnInvFit <- trainSet %>% train(quality ~ .,
                                 method = "kknn",
                                 data = .,
                                 tuneGrid = data.frame(kmax = seq(3, 81, 2),
                                                       distance = seq(0.5, 2, 0.5),
                                                       kernel = "inv"),
                                 trControl = control)
plot_optimization_chart(kknnInvFit, "Optimization for kknn / inv model")

# predicting
predicted <- testSet %>% predict(kknnInvFit, newdata = ., type = "raw")

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = testSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- bind_rows(accuracyResults, tibble(model ="kknn/inv",
                                                     accuracy = acc))
accuracyResults



###############################
# validation of the final model
# final model: kknn inv
###############################

# predicting
predicted <- validationSet %>% predict(kknnInvFit, newdata = ., type = "raw")

# accuracy calculation
cm <- confusionMatrix(data = predicted, reference = validationSet$quality)
cm$table
acc <- cm$overall["Accuracy"]

accuracyResults <- bind_rows(accuracyResults, tibble(model ="kknn/inv/validation",
                                                     accuracy = acc))
accuracyResults



# reset warnings
options(warn = oldw)
