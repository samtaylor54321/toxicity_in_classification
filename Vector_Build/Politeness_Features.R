
######################################  BUILD POLITENESS FEATURES ###########################################
 
# Function which uses the politness package to build a number of features relating to the politeness of 
# particular comments

# Load Packages -----------------------------------------------------------

library(tidyverse)
library(politeness)
library(spacyr)

# Load Datasets ----------------------------------------------------------

# read in CSVs and only include columns which are present in both training and test sets. 

train <- read_csv("/Users/samtaylor/Downloads/jigsaw-unintended-bias-in-toxicity-classification/train.csv") %>% 
  select(id, comment_text, target)

test <- read_csv("/Users/samtaylor/Downloads/jigsaw-unintended-bias-in-toxicity-classification/test.csv") 

# Build Politeness Features -----------------------------------------------

# initialise spacy wrapper to make sure that all possible features are created for modelling. NB this should 
# already be working correctly

spacyr::spacy_initialize()

# sets test to understand how long it takes to create this matrix 

start_time <- Sys.time()

polite_features <- politeness(train$comment_text, metric ='count', drop_blank =TRUE, parser='spacy')

end_time <- Sys.time()

# need to think of a way to output and getting this working in python - building a joint dataset?
# set up a script which takes in all CSV files and outputs a final dataset which can be read in by either R or Python. 
# RDS?


