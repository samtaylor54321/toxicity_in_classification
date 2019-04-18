
#################### Punctuation Feature #######################

# Function which extracts counts of different types of punctuation used in 
# text comments as well as a total count 

# Load Packages -----------------------------------------------------------

library(tidyverse)

# Build Function ----------------------------------------------------------

punctuation <- function (.data, id_col, text_col) {
  .data <- .data %>% 
    dplyr::select(!!ensym(id_col), !!ensym(text_col)) %>% 
    dplyr::mutate(id = !!ensym(id_col),
                  exclamation = str_detect(!!ensym(text_col),'!'),
                  question = str_detect(!!ensym(text_col),'\\?'),
                  exclamation_count = str_count(!!ensym(text_col),'!'),
                  question_count = str_count(!!ensym(text_col),'\\?'),
                  semi_colon = str_detect(!!ensym(text_col), ';'),
                  semi_colon_count = str_count(!!ensym(text_col), ';'),
                  amp = str_detect(!!ensym(text_col), '&'),
                  amp_count = str_count(!!ensym(text_col), '&'))
  .data 
}
