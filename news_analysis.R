#################### DATA PREPROCESSING ####################

# Initiate a connection to H2O
library(h2o)
h2o.init(nthreads = -1)
h2o.removeAll()

# Load data and remove all non-predictive attributes
news <- read.csv("OnlineNewsPopularity.csv")
news <- subset(news, select = -c(url, timedelta))

# Convert target variable into 2 classes using the median as the threshold
news$shares <- ifelse(news$shares >= 1400, 1, 0)

# Factorize the dummy variables and apply feature scaling
cols <- c(12:17, 30:37, 59)
news[cols] <- lapply(news[cols], factor)
news[, !cols] <- as.data.frame(scale(news[, !cols]))
summary(news)

# Create train/test partitions using a 75/25 split
news <- as.h2o(news)
split <- h2o.splitFrame(news, ratios=0.75, seed=123)
train <- split[[1]]
test <- split[[2]]



#################### TRAINING MODELS #################### 

##### Naive Bayes #####

# Retrieving cross-validation results
nb_perf <- h2o.naiveBayes(training_frame = train,
                          y = "shares", 
                          model_id = "nb",
                          nfolds = 5, seed = 123)
print(nb_perf)


##### Logistic Regression #####

# Selecting the hyperparameters
log_hype <- list(alpha = c(0.005, 0.01, 0.02), 
                 lambda = c(0, 0.000005, 0.00001))

# Applying Grid Search
log_grid <- h2o.grid("glm", y = "shares", 
                     grid_id = "log",
                     family = "binomial",
                     hyper_params = log_hype, 
                     training_frame = train,
                     nfolds = 5, seed = 123)

# Retrieving cross-validation results
log_perf <- h2o.getGrid(grid_id = "log", sort_by = "accuracy", decreasing = T)
print(log_perf)

# Storing the best model
log_best <- h2o.getModel(log_perf@model_ids[[1]])


##### Random Forests #####

# Selecting the hyperparameters
rf_hype <- list(ntrees = c(50, 100, 200), 
                max_depth = c(5, 10, 20))

# Applying Grid Search
rf_grid <- h2o.grid("randomForest", y = "shares", 
                    grid_id = "rf",
                    hyper_params = rf_hype, 
                    training_frame = train, 
                    nfolds = 5, seed = 123)

# Retrieving cross-validation results
rf_perf <- h2o.getGrid(grid_id = "rf", sort_by = "accuracy", decreasing = T)
print(rf_perf)

# Storing the best model
rf_best <- h2o.getModel(rf_perf@model_ids[[1]])


##### Artificial Neural Network #####

# Selecting the hyperparameters
ann_hype <- list(activation = c("Rectifier", "Maxout"),
                 hidden=list(c(50,50), c(100,100), c(75,75,75)))

# Applying Grid Search
ann_grid <- h2o.grid("deeplearning", y = "shares", 
                     grid_id = "ann",
                     loss = "CrossEntropy",
                     epochs = 10,
                     hyper_params = ann_hype, 
                     training_frame = train, 
                     nfolds = 5, seed = 123)

# Retrieving cross-validation results
ann_perf <- h2o.getGrid(grid_id = "ann", sort_by = "accuracy", decreasing = T)
print(ann_perf)

# Storing the best model
ann_best <- h2o.getModel(ann_perf@model_ids[[1]])


##### Gradient Boosted Trees #####

# Selecting the hyperparameters
gbm_hype <- list(ntrees = c(100, 200), 
                 max_depth = c(10, 20), 
                 learn_rate = c(0.005, 0.01))

# Applying Grid Search
gbm_grid <- h2o.grid("gbm", y = "shares", 
                     grid_id = "gbm",
                     hyper_params = gbm_hype, 
                     training_frame = train, 
                     nfolds = 5, seed = 123)

# Retrieving cross-validation results
gbm_perf <- h2o.getGrid(grid_id = "gbm", sort_by = "accuracy", decreasing = T)
print(gbm_perf)

# Storing the best model
gbm_best <- h2o.getModel(gbm_perf@model_ids[[1]])



#################### MODEL SELECTION #################### 

# Creating a dataframe to store training results
results <- as.data.frame(c("Naive Bayes",
                           "Logistic Regression",
                           "Random Forests",
                           "Artificial Neural Network", 
                           "Gradient Boosted Trees"))

colnames(results)[1] <- "Model"
results$AUC <- 0
results$Accuracy <- 0

# Retrieving the results for the best model for each algorithm
results[1, "AUC"] <- nb_perf@model$cross_validation_metrics_summary[2, 1]
results[2, "AUC"] <- log_best@model$cross_validation_metrics_summary[2, 1]
results[3, "AUC"] <- rf_best@model$cross_validation_metrics_summary[2, 1]
results[4, "AUC"] <- ann_best@model$cross_validation_metrics_summary[2, 1]
results[5, "AUC"] <- gbm_best@model$cross_validation_metrics_summary[2, 1]
results[1, "Accuracy"] <- nb_perf@model$cross_validation_metrics_summary[1, 1]
results[2, "Accuracy"] <- log_best@model$cross_validation_metrics_summary[1, 1]
results[3, "Accuracy"] <- rf_best@model$cross_validation_metrics_summary[1, 1]
results[4, "Accuracy"] <- ann_best@model$cross_validation_metrics_summary[1, 1]
results[5, "Accuracy"] <- gbm_best@model$cross_validation_metrics_summary[1, 1]
results[5, "Accuracy"] <- nb_cm$overall[1]
print(results)



#################### TESTING FINAL MODEL ####################

base_pred <- h2o.predict(nb_perf, test)
base_perf <- h2o.performance(nb_perf, test)
test_pred <- h2o.predict(rf_best, test)
test_perf <- h2o.performance(rf_best, test)

print(base_perf@metrics$max_criteria_and_metric_scores)
print(base_perf@metrics$AUC)
plot(base_perf, main="Baseline Model - ROC Curve (AUC = 0.6447122)")

print(test_perf@metrics$max_criteria_and_metric_scores)
test_auc <- h2o.auc(test_perf, valid = T)
plot(test_perf, main="Final Model - ROC Curve (AUC = 0.729215)")