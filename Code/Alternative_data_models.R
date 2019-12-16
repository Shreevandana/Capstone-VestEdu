# Alternate Dataset Model Building 
# Simple Linear Regression and Random Forest Regression for 
# Basic idea for model ----> CDR ~ Rank + Degree + Major 

# 0. Install required libraries if not present
if (!require("readxl")) install.packages("readxl")
if (!require("dplyr")) install.packages("dplyr")
if (!require("magrittr")) install.packages("magrittr")
if (!require("reshape2")) install.packages("reshape2")
if (!require("stringr")) install.packages("stringr")
if (!require("randomForest")) install.packages("randomForest")
if (!require("caret")) install.packages("caret")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("ggthemes")) install.packages("ggthemes")
if (!require("RColorBrewer")) install.packages("RColorBrewer")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("gridExtra")) install.packages("RColorBrewer")
if (!require("graphics")) install.packages("graphics")
if (!require("raster")) install.packages("raster")

# 1. Activate libraries
suppressPackageStartupMessages(library(readxl))       # to r/w excel files
suppressPackageStartupMessages(library(dplyr))        # easier data manipulation 
suppressPackageStartupMessages(library(magrittr))     # for pipe operator
suppressPackageStartupMessages(library(reshape2))     # to pivot the data from wide to long format
suppressPackageStartupMessages(library(stringr))      # for sting manipulations

suppressPackageStartupMessages(library(randomForest)) # Random forest modelling package
suppressPackageStartupMessages(library(caret))        # classification and regression testing framework

suppressPackageStartupMessages(library(ggplot2))      # for beautiful plots
suppressPackageStartupMessages(library(ggthemes))     # for beautiful plot themes
suppressPackageStartupMessages(library(RColorBrewer)) # for beautiful color schemes
suppressPackageStartupMessages(library(gridExtra))    # for plot grid arrangements
suppressPackageStartupMessages(library(graphics))     # for graphics adjustment and grid arrangement
suppressPackageStartupMessages(library(raster))       # for map rastering

# 2. Read data from the excel file (Make sure that the required excel file is present in the current working directory)
cdr <- readxl::read_excel(path = "AlternateMethod-DOEMergedDataForModels-v1.2.xlsx", 
                          sheet = "final_data_pasted_instance")

# 3. Reshape the data (Convert wide format data to long format)
cdr_long <- reshape2::melt(data = cdr, 
                           id.vars=c("UNITID", "OPEID", "OPEID6", "INSTNM", "CITY", "STABBR",
                                     "ZIP", "CONTROL", "FORBES_UNIVERSITY_RANK", "TIER", "Inst-type-II"))

# Create degree, major and cdr_type variables
cdr_long %<>% dplyr::mutate(degree = str_remove(string = substr(x = variable, 
                                                                start = 1, 
                                                                stop = 2), 
                                                pattern = "_"), 
                            major = str_remove_all(string = str_extract(string = variable, pattern = "_[a-zA-Z]+_"),
                                                   pattern = "_"),
                            cdr_type = str_extract(string = variable, pattern = "CDR[2-3]")) %>%
  dplyr::select(-variable)

colnames(cdr_long) <- tolower(make.names(colnames(cdr_long)))

# pivot back to add cdr2 and cdr3 columns
cdr2df <- reshape2::dcast(data = cdr_long, 
                          formula = unitid+ opeid+ opeid6+ instnm+ city+ stabbr+
                            zip+ control+ forbes_university_rank+ tier+ inst.type.ii + major + degree ~ cdr_type)

# Correct field types
cdr2df %<>% dplyr::mutate(tier = as.factor(tier), 
                          inst.type.ii = dplyr::recode_factor(inst.type.ii, '3' = 'Non-Selective', '5' = 'Selective', .ordered = TRUE),
                          degree = dplyr::recode_factor(degree, '3' = "UG", '5' = "G", .ordered = TRUE),
                          major = as.factor(major))

# 4. EDA plots 
dgs <- unique(cdr2df$degree)
mjrs <- unique(cdr2df$major)
list_of_plots = list()
for (j in 1:length(dgs)){
  for (i in 1:length(mjrs)){
    sub_df <- cdr2df %>% dplyr::filter(degree == dgs[j] & major == mjrs[i])
    list_of_plots[[length(mjrs)*(j-1) + i]] = ggplot(data = sub_df, 
                                                     aes(x = forbes_university_rank, y = CDR2)) + 
      geom_point(aes(color = tier), shape = 16) + 
      geom_smooth(method=lm) + 
      ggtitle(paste("Rank vs. CDR-2 for ", dgs[j], mjrs[i])) + 
      theme_light()
    
    print(paste(j, dgs[j], i, mjrs[i], length(mjrs)*(j-1) + i))
  }
}
do.call(grid.arrange, c(list_of_plots, 
                        nrow = 2, 
                        top = "Scatter plots for all majors and degrees (CDR ~ Rank)"))


# 5. Data Pre-processing before modelling
# 5.1 Inverse calculation of CDR
cdr2df$cdr2inv <- 1/cdr2df$CDR2
cdr2df$cdr3inv <- 1/cdr2df$CDR3

# 5.2 multiply cdr by 100 (to make it percentages rather than decimals)
cdr2df$cdr2100 <- cdr2df$CDR2 * 100
cdr2df$cdr3100 <- cdr2df$CDR3 * 100

# 5.3 Remove all zero values in dependent variable
reg2_df <- cdr2df[which(cdr2df$cdr2100 != 0), ]
reg3_df <- cdr2df[which(cdr2df$cdr3100 != 0), ]

# 5.4 Train-test split
set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 80% of data as sample from total 'n' rows of the data  
sample2 <- sample.int(n = nrow(reg2_df), size = floor(.80*nrow(reg2_df)), replace = F)
train2 <- reg2_df[sample2, ]
test2  <- reg2_df[-sample2, ]

sample3 <- sample.int(n = nrow(reg3_df), size = floor(.80*nrow(reg3_df)), replace = F)
train3 <- reg3_df[sample3, ]
test3  <- reg3_df[-sample3, ]

# 6. Model Development and testing 
# 6.0 Function to calculate the R-squared 
r.sqrd <- function(actual, preds){
  rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
  tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
  rsq <- 1 - rss/tss
  return(list(mse = rss/length(actual), r.sq = rsq))
}

# 6.1 Linear model
# 6.1.1 for CDR2 ~ rank + major + degree
k10 <- caret::trainControl(method = "cv", number = 10)
ols_model2 <- caret::train(cdr2100 ~ forbes_university_rank + major + degree, 
                          data = train2, 
                          trControl = k10, 
                          method = "lm")
summary(ols_model2)
x$adj.r.squared
pred = predict(ols_model2, newdata = test2)
r.sqrd(actual = test2$cdr2100, preds = pred)
# Print performance metrics
cat('OLS Regression for CDR2 \nIn-Sample R.sq:', 
    r.sqrd(actual = train2$cdr2100, preds = predict(object = ols_model2, newdata = train2))$r.sq,
    '\nIn-Sample MSE:', 
    r.sqrd(actual = train2$cdr2100, preds = predict(object = ols_model2, newdata = train2))$mse, 
    '\nOut-of-sample R.sq:', 
    r.sqrd(actual = test2$cdr2100, preds = predict(object = ols_model2, newdata = test2))$r.sq,
    '\nOut-of-sample MSE:', r.sqrd(actual = test2$cdr2100, preds = predict(object = ols_model2, newdata = test2))$mse)

# 6.1.2 for CDR3 ~ rank + major + degree
k10 <- caret::trainControl(method = "cv", number = 10)
ols_model3 <- caret::train(cdr3100 ~ forbes_university_rank + major + degree, 
                          data = train3, 
                          trControl = k10, 
                          method = "lm")
summary(ols_model3)

pred = predict(ols_model3, newdata = test3)
r.sqrd(actual = test3$cdr3100, preds = pred)
# Print performance metrics
cat('OLS Regression for CDR3 \nIn-Sample R.sq:', 
    r.sqrd(actual = train3$cdr3100, preds = predict(object = ols_model3, newdata = train3))$r.sq,
    '\nIn-Sample MSE:', 
    r.sqrd(actual = train3$cdr3100, preds = predict(object = ols_model3, newdata = train3))$mse, 
    '\nOut-of-sample R.sq:', 
    r.sqrd(actual = test3$cdr3100, preds = predict(object = ols_model3, newdata = test3))$r.sq,
    '\nOut-of-sample MSE:', r.sqrd(actual = test3$cdr3100, preds = predict(object = ols_model3, newdata = test3))$mse)


# 6.2 Random forest model
# 6.2.2 for CDR2 ~ rank + major + degree
rf.fit2 <- randomForest::randomForest(cdr2100 ~ forbes_university_rank + major + degree, 
                                        data = train2,
                                        ntree = 500,
                                        importance = T)
rf.fit2
data.frame(Feature = row.names(rf.fit2$importance), rf.fit2$importance) %>% 
  mutate('%Importance' = IncNodePurity * 100/sum(rf.fit2$importance[, 'IncNodePurity']))

# Print performance metrics
cat('Random Forest Regression for CDR2 \nIn-Sample R.sq:', 
    r.sqrd(actual = train2$cdr2100, preds = predict(object = rf.fit2, newdata = train2))$r.sq,
    '\nIn-Sample MSE:', 
    r.sqrd(actual = train2$cdr2100, preds = predict(object = rf.fit2, newdata = train2))$mse, 
    '\nOut-of-sample R.sq:', 
    r.sqrd(actual = test2$cdr2100, preds = predict(object = rf.fit2, newdata = test2))$r.sq,
    '\nOut-of-sample MSE:', 
    r.sqrd(actual = test2$cdr2100, preds = predict(object = rf.fit2, newdata = test2))$mse)


# 6.2.3 for CDR3 ~ rank + major + degree
rf.fit3 <- randomForest::randomForest(cdr3100 ~ forbes_university_rank + major + degree, 
                                     data = train3,
                                     ntree = 500,
                                     importance = T)
rf.fit3
data.frame(Feature = row.names(rf.fit3$importance), rf.fit3$importance) %>% 
  mutate('%Importance' = IncNodePurity * 100/sum(rf.fit3$importance[, 'IncNodePurity']))

# Print performance metrics
cat('Random Forest Regression for CDR3 \nIn-Sample R.sq:', 
    r.sqrd(actual = train3$cdr3100, preds = predict(object = rf.fit3, newdata = train3))$r.sq,
    '\nIn-Sample MSE:', 
    r.sqrd(actual = train3$cdr3100, preds = predict(object = rf.fit3, newdata = train3))$mse, 
    '\nOut-of-sample R.sq:', 
    r.sqrd(actual = test3$cdr3100, preds = predict(object = rf.fit3, newdata = test3))$r.sq,
    '\nOut-of-sample MSE:', 
    r.sqrd(actual = test3$cdr3100, preds = predict(object = rf.fit3, newdata = test3))$mse)

# Random forest regression plots using CDR3
tdf <- reg3_df %>% mutate(line.type = factor(paste0(major, '-', degree), 
                                           levels = c("Arts-UG", "Business-UG", "STEM-UG", "Arts-G", "Business-G", "STEM-G")), 
                        pred = predict(object = rf.fit3, 
                                       newdata = .) ) %>% filter(forbes_university_rank <= 300)

p <- ggplot(data = tdf) +
  ggtitle(label = "Random Forest regression plot" , 
          subtitle = "Rankings vs CDR2") + 
  xlab(label = "Forbes University Rankings") + 
  ylab(label = "CDR-3yrs (in %)") + 
  geom_point(aes(x = forbes_university_rank,
                 y= cdr3100),
             color = "darkseagreen1") +
  geom_line(aes(x = forbes_university_rank,
                y = pred,
                color = line.type),
            size = 1.1) + 
  xlim(0, 300) + 
  ylim(0, 6) + 
  theme_gdocs() + scale_color_brewer(palette="Dark2")
p
