

#################################### Modelling Script #####################################

# Script which test different machine learning models, comparing the performance of
# these with a view to selecting best candidate model to take forward

# Load Packages -----------------------------------------------------------

library(mlr)
library(purrr)
library(tidytext)
library(spacyr)
library(text2vec)
library(tidyverse)
library(politeness)
library(keras)

# Load Data ---------------------------------------------------------------

train_raw <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/train.csv")
test_raw <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/test.csv")

# Feature Enginering  -----------------------------------------------------------

# take a sample of observations to explain
train <- head(train_raw, 10000)
# run punctuation and tidy data into correct format for modelling and combine with
# the punctuation features 
training_set <- train %>% 
  punctuation(id_col = id, text_col = comment_text) %>% 
  select(-comment_text) %>% cbind(politeness(train$comment_text, parser="none"))
# remove any logical features and replace with numerics
training_set <- map_if(training_set, is.logical, as.numeric)
# convert back to dataframe to capture list creation by map_if
if (!is.data.frame(training_set)) {
  training_set <- as.data.frame(training_set)
} 
# define correct target variable for the model.
training_set$target <- as.character(train$target < 0.5)



# Modelling ---------------------------------------------------------------

# Make Task
task <- makeClassifTask(data =training_set, target = "target")
# Make NB Learner and Assess Performance
xgb_learner <- makeLearner(cl =  "classif.ranger", predict.type = 'prob' )
# CV Score
xgb_model <- mlr::train(xgb_learner, task)

crossval(learner = xgb_learner, task = task, iters = 5, stratify = TRUE, measures =list(auc, f1))





