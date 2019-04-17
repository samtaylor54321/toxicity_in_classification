

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

# Load Data ---------------------------------------------------------------

train <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/train.csv")
test <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/test.csv")

# Feature Enginering  -----------------------------------------------------------

tokens <- train %>% tidytext::unnest_tokens(output=word,input=comment_text)


remove stop words
remove punctuation in


Sys.setenv('R_MAX_VSIZE'=32000000000)
Sys.getenv('R_MAX_VSIZE')


# tidy data into correct format 

training_set <- train %>% 
  punctuation(id_col = id, text_col = comment_text) %>% 
  select(-comment_text) 

training_set$target <- as.character(train$target < 0.5)

training_set <- as.data.frame(training_set)

# Test to make sure that any columns are logical are convert to numeric for
# the purposes of modelling
  
training_set <-  as.data.frame(map_if(training_set, is.logical, as.numeric))
  
# Make Task

task <- makeClassifTask(data =training_set, target = "target")

# Make NB Learner and Assess Performance

nb_learner <- makeLearner(cl =  "classif.naiveBayes", predict.type = 'response' )

nb_model<- train(nb_learner, task)

nb_preds <- predict(nb_model, task)

performance(nb_preds, measures = 'auc')



# classif.ranger
# classif.glmnet
# classif.glmnet





