## ---- packages --------
# Load packages
library(tidymodels) # use tidymodels framework
library(ggplot2) # producing visual displays of data
library(dplyr) # manipulating and cleaning data
library(here) # making relative pathways
library(glmnet) # for LASSO regression engine
library(ranger) # making random forest model
library(doParallel) # for parallel processing
library(rsample) # for cross validation

# Set seed
rngseed <- 1234 # 1234 = rngseed 
set.seed(rngseed)

# Load data
data_location <- here::here("ml-models-exercise","cleandata.rds")
data <- readRDS(data_location)
str(data)

## ---- stratify --------
# Stratify RACE by AGE
# Define age groups of under 40 and over 40
data$AGEcat <- ifelse(data$AGE <= 40, "<=40", ">40")

# Use by() to stratify the race variable by age and find the summary
RaceAge_summary <- by(data$RACE, data$AGEcat, summary)

# Printing the summary statistics
RaceAge_summary

data <- subset(data, select = -AGEcat) # remove the AGEcat variable from the original data frame

# Stratify RACE by SEX
RaceSex_summary <- by(data$RACE, data$SEX, summary)
RaceSex_summary

## ---- combine --------
# Combining RACE levels 7 and 88
data$RACE <- factor(ifelse(data$RACE %in% c(7, 88), 3, data$RACE)) # combine levels 7 and 88 of the factor variable RACE into one level called 3 using the ifelse() function
str(data$RACE) # check the data structure to ensure that they were combined

## ---- correlation --------
# Correlation Plot
pairs(data[, c("WT", "HT", "AGE", "Y")], # use pairs() to find pairwise correlation for continuous variables
      labels = c("WT", "HT", "AGE", "Y"), # specify the labels for each variable
      main = "Correlation Plot", # Title of the plot
      row1attop = FALSE, # Default behavior for the direction of the diagonal
      gap = 1, # Distance between subplots
      cex.labels = NULL,# Size of the diagonal text (default)
      font.labels = 1) # Font style of the diagonal text

# correlation coefficients

matrix <- cor(data[, c("WT", "HT", "AGE", "Y")]) # use cor() function to make a correlation matrix for the specified continuous
print(matrix) # Print the correlation matrix


## ---- summary --------
# Summary of the data
summary(data)
# It looks like height is in meters and weight is in kg in this case

# Calculate BMI
data <- mutate(data, BMI = WT/(HT^2)) # use mutate() from dpylr to add a new variable to the data frame using preexisting columns
str(data$BMI) # check structure of the BMI variable

## ---- first-fit --------
set.seed(rngseed) # set seed for reproducibility

# Linear Regression
glm_recipe <- 
  recipe(Y ~ ., data = data) # specify the recipe by putting the formula for the linear model with all the predictors

glm_model <- linear_reg() %>% # make object for the model function
  set_engine("lm") %>% # Use linear regression engine lm
  set_mode("regression") # and also, set to regression mode

glm_workflow <- workflow() %>% # create a workflow 
  add_model(glm_model) %>% # apply linear regression model
  add_recipe(glm_recipe) # then, apply the recipe

glm_fit <- 
  glm_workflow %>% # add the workflow to the fit
  fit(data = data) # fit to the whole data set
tidy(glm_fit) # produce tibble to organize the resulting linear regression fit

# LASSO Regression
## I used a tidytuesday example and stack overflow to create this LASSO regression
lasso_rec <- recipe(Y ~ ., data = data) %>% # create recipe containing all predictors
  step_zv(all_numeric(), -all_outcomes()) %>% # removes non-zero variance variables of predictors (not outcomes)
  step_scale(all_numeric(), -all_outcomes(), -all_nominal()) %>% # regularizes numerical variables
  step_dummy(all_nominal()) # makes factor variables dummy variables

lasso_prep <- lasso_rec %>%
  prep() # ensures that strings are not converted into factors, call prep to prepare the recipe for fitting
lasso_spec <- linear_reg(penalty = 0.1, mixture = 1) %>% # gives a L1 penalty to the variables
  set_engine("glmnet") # set engine for LASSO regression

lasso_wf <- workflow() %>%
  add_recipe(lasso_rec) # apply the recipe to a lasso workflow

lasso_fit <- lasso_wf %>% # fit the model using the specs that were specified above
  add_model(lasso_spec) %>%
  fit(data = data) # fit to all data

lasso_fit %>%
  extract_fit_parsnip() %>% # extract the fit and use tidy() to produce a tibble
  tidy()

# Random Forest (RF) Model
rf_rec <- recipe( Y ~ ., data = data) # use full model as recipe for the random forest model

rf_model <- rand_forest()%>% # use rand_forest() to make a random forest model
  set_engine("ranger", seed = rngseed)%>% # set engine to ranger with the same seed for reproducibility
  set_mode("regression")%>% # set to regression mode
  translate()

rf_workflow <- workflow() %>% # create workflow for rf model
  add_recipe(rf_rec)%>% # apply recipe
  add_model(rf_model) # apply model

rf_fit <- rf_workflow%>%
  fit(data = data) # use the workflow to fit the rf model to the data
rf_fit # print fit (there is no tidy method for ranger)

## ---- predictions --------
# Making Predictions and Computing RMSE
## Linear regression
glm_aug <- augment(glm_fit, data) # add glm predictions to an augmented data frame
glm_rmse <- rmse(data = glm_aug, # compute rmse based on Y expected values and .pred as the estimate
                 truth = Y,
                 estimate = .pred)
print(glm_rmse)
## LASSO regression
lasso_aug <- augment(lasso_fit, data)
lasso_rmse <- rmse(data = lasso_aug,
                   truth = Y,
                   estimate = .pred)
print(lasso_rmse)
## Rf model
rf_aug <- augment(rf_fit, data)
rf_rmse <- rmse(data = rf_aug,
                truth = Y,
                estimate = .pred)
print(rf_rmse)

## ---- pred-plot --------
# Predicted versus Observed plot
## Add data to one data frame
plot_data <- data.frame(
  observed = c(data$Y), # Add observed value from original data frame
  glm_predictions = c(glm_aug$.pred), # add predicted values from each model
  lasso_predictions = c(lasso_aug$.pred), 
  rf_predictions = c(rf_aug$.pred), 
  model = rep(c("linear regression", "LASSO regression", "random forest model"), each = nrow(data))) # add labels to indicate the model
plot_data

## Create Plot
ggplot(plot_data, aes(x = observed)) +
  geom_point(aes(y = glm_predictions, color = "linear regression"), shape = 1) +
  geom_point(aes(y = lasso_predictions, color = "LASSO regression"), shape = 2) +
  geom_point(aes(y = rf_predictions, color = "random forest model"), shape = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") + #add the 45 degree line
  labs(x = "Observed Values", y = "Predicted Values", title = "Observed versus Predicted Values for Machine Learning Models")

## ---- tune1 --------
# LASSO tuning
lasso_grid <- 10^seq(-5, 2, length.out = 50) # create tuning grid
## I got help from ChatGPT for creating a tuning grid over that range from 1E-5 to 1E2 with 50 values on a log scale.

tune_lasso_wf <- workflow() %>%
  add_recipe(lasso_rec) %>%  # add the same recipe to the workflow
  add_model(
    linear_reg(penalty = tune()) %>%  # specify tuning penalty for LASSO model
      set_engine("glmnet"))

apparent_data <- apparent(data) # used to create an object with only tuning data

lasso_tune_grid <- tune_grid(
  object = tune_lasso_wf,  # Specify object as the workflow
  resamples = apparent_data, # use data for tuning from the apparent function
  grid = expand.grid(penalty = lasso_grid),  #  add tuning penalty grid
  control = control_grid(save_pred = TRUE)) # save the predictions

lasso_tune_grid %>% 
  autoplot()

# Stop parallel processing
stopImplicitCluster()

## ---- tune2 --------
# Tree tuning
tune_spec <- 
  rand_forest(
    mtry = tune(), # parameters of random forest model to tune are the mtry, trees, and min_n
    trees = 300,
    min_n = tune()) %>% 
  set_engine("ranger", seed = 1234) %>% # make sure to set seed again for the tree
  set_mode("regression")

apparent_data2 <- apparent(data)

## I used ChatGPT to help me define the tree grid for the parameters mtry and min_n
tree_grid <- grid_regular(mtry(range = c(1, 7)), 
                          min_n(range = c(1, 21)),
                          levels = 7) # make a grid with 7 levels with the specified                                          parameters from above

rf_tune_wf <- workflow() %>% # create workflow
  add_recipe(rf_rec) %>% # add recipe from earlier
  add_model(tune_spec) # add spec that is defined above

rf_res <- rf_tune_wf %>%
  tune_grid(
    resamples = apparent_data2, # use data from the apparent function
    grid = tree_grid) # use the specified grid above

rf_res %>%
  autoplot()

## ---- tune3 --------
# Cross validation folds
folds_data <- vfold_cv(data, v = 5, repeats = 5)
# Set the number of cores to use
num_cores <- detectCores() - 1
# Initialize parallel backend
doParallel::registerDoParallel(cores = num_cores) # add parallel processing to speed up the computations

set.seed(rngseed) # set seed for reproducibility

lasso_cv_grid <- tune_grid(
  object = tune_lasso_wf,  # Specify object as the workflow
  resamples = folds_data, # use data for tuning from the CV folds
  grid = expand.grid(penalty = lasso_grid),  #  add lasso tuning penalty grid
  control = control_grid(save_pred = TRUE)) # save the predictions

lasso_cv_grid %>% 
  autoplot()

# Stop parallel processing
stopImplicitCluster()

# RF Tree Model
##I looked for how to do parallel processing with ChatGPT
# Set the number of cores to use
num_cores <- detectCores() - 1

# Initialize parallel backend
doParallel::registerDoParallel(cores = num_cores)

set.seed(rngseed) # set seed for reproducibility

rf_res <- rf_tune_wf %>%
  tune_grid(
    resamples = folds_data, # use data from the CV folds
    grid = tree_grid)
rf_res %>%
  autoplot()

# Stop parallel processing
stopImplicitCluster()