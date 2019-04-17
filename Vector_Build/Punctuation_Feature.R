
#################### Punctuation Feature #######################

# Function which extracts counts of different types of punctuation used in 
# text comments as well as a total count 

# Load Packages -----------------------------------------------------------

library(tidyverse)

# Load Data ---------------------------------------------------------------

train <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/train.csv")
test <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/test.csv")

# Build Function ----------------------------------------------------------

punctuation <- function (.data, id_col, text_col) {
  .data <- .data %>% 
    dplyr::select(!!ensym(id_col), !!ensym(text_col)) %>% 
    dplyr::mutate(exclamation = str_detect(!!ensym(text_col),'!'),
                  question = str_detect(!!ensym(text_col),'\\?'),
                  exclamation_count = str_count(!!ensym(text_col),'!'),
                  question_count = str_count(!!ensym(text_col),'\\?'),
                  semi_colon = str_detect(!!ensym(text_col), ';'),
                  semi_colon_count = str_count(!!ensym(text_col), ';'),
                  amp = str_detect(!!ensym(text_col), '&'),
                  amp_count = str_count(!!ensym(text_col), '&'))
  .data 
}

ggplot(punc, aes(x=exclamation_count, y=toxicity)) + geom_point() + geom_smooth(method='lm')



punc <- punctuation(train, id, comment_text)
punc$toxicity <- train$target

