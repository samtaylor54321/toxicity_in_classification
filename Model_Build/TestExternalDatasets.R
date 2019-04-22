

#################################### Modelling Script #####################################

# Script which test different machine learning models, comparing the performance of
# these with a view to selecting best candidate model to take forward

# Load Packages -----------------------------------------------------------

library(mlr)
library(purrr)
library(lightgbm)
library(keras)
library(tidytext)
library(spacyr)
library(text2vec)

# Load Data ---------------------------------------------------------------

train <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/train.csv")
test <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/test.csv")

# Feature Enginering  -----------------------------------------------------------

# tidy data into correct format 

training_set <- train %>% 
  punctuation(id_col = id, text_col = comment_text) %>% 
  select(-comment_text) 

training_set$target <- as.character(train$target < 0.5)

training_set <- as.data.frame(training_set)

# Test to make sure that any columns are logical are convert to numeric for
# the purposes of modelling
  
training_set <-  as.data.frame(map_if(training_set, is.logical, as.numeric))

# Modelling ---------------------------------------------------------------

# Make Task

task <- makeClassifTask(data =training_set, target = "target")

# Make NB Learner and Assess Performance

nb_learner <- makeLearner(cl =  "classif.naiveBayes", predict.type = 'prob' )

# CV Score

crossval(learner = nb_learner, task = task, iters = 10, stratify = TRUE, measures =list(auc))

# predict on test set 

nb_preds <- predict(nb_model, task)

performance(nb_preds, measures = 'auc')


# getParamSet(nb_learner) tune these variables
# perform grid/random/bayesian optimisation search 
# 

cluster_example <- makeClusterTask(data = mtcars)
cluster_example

View(listLearners(cluster_example))

makeResampleDesc(method ='')
# classif.ranger
# classif.glmnet
# classif.glmnet


embeddings
politeness 
punctuation 
curse words
hate speech 


passed to xgboost 
or lightgbm - how does this impact model performance. 




