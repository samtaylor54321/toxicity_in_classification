
######################################  BUILD POLITENESS FEATURES ################################
 
# Function which uses the politness package to build a number of features relating to the politeness of 
# particular comments

# Load Packages -----------------------------------------------------------

library(tidyverse)
library(politness)

# Load Datasets ----------------------------------------------------------

# read in CSVs and only include columns which are present in both training and test sets. 

train <- read_csv("/Users/samtaylor/Downloads/jigsaw-unintended-bias-in-toxicity-classification/train.csv") %>% 
  select(id, comment_text, target)

test <- read_csv("/Users/samtaylor/Downloads/jigsaw-unintended-bias-in-toxicity-classification/test.csv") 

# Build Politeness Features -----------------------------------------------









# building polite features 







install.packages('politeness')
library(politeness)