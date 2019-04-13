
#################### Punctuation Feature #######################

# Function which extracts counts of different types of punctuation used in 
# text comments as well as a total count 

# Load Packages -----------------------------------------------------------

library(tidyverse)

# Load Data ---------------------------------------------------------------

train <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/train.csv")
test <- read_csv("/Users/samtaylor/Documents/Git/kaggle_jigsaw/kaggle_jigsaw/Data/test.csv")

# Build Function ----------------------------------------------------------

punctuation <- function (.data, id_var, text_column) {
  tbl <- .data %>% 
    select(id_var, text_column) %>% 
    mutate(exclamation = str_detect(text_column, 'a'))

    return(tbl)  
  #%>% 
    #mutate(exclamation = str_detect(text_column, "!"),
     #      question = str_count(text_column, "\\?"),
    #       )
}
# Run Function ------------------------------------------------------------


punctuation(train, id_var = "id", text_column = "comment_text")

punk <- punctuation(train, id_var ='id', text_column  = "comment_text")

punk$target <- train$target

head(punk, 20) %>% View()

punk %>% head(20) %>% mutate(exclamation = str_detect(comment_text,'!'),
                             exclamation_count = str_count(comment_text,'!'))

# Plot Function -----------------------------------------------------------

ggplot(punk, aes(x=as.factor(exclamation), y=target)) + geom_boxplot()

train %>% select(comment_text, target) %>% mutate(exclamation = str_detect(comment_text, "!")) %>% 
  ggplot(aes(x=exclamation, y=target)) + geom_boxplot()


# work out if differene between groups is statistically signifcant - is there
# any prediction power

train %>% select(comment_text, target) %>% mutate(exclamation = str_detect(comment_text, "!")) %>% 
  group_by(exclamation) %>% summarise(mean_target = mean(target),
                                      median_target = median(target),
                                      observations =n())
