library(faraway)
library(readr)
library(mlbench)
library(caret)
library(corrplot)

housing <- read_csv("Housing.csv")
spec(housing)
head(housing, n = 50)

table(housing$mainroad)

housing$street[housing$mainroad == "yes"] <- 1
housing$street[housing$mainroad != "yes"] <- 0

table(housing$guestroom)

housing$guest[housing$guestroom == "yes"] <- 1
housing$guest[housing$guestroom != "yes"] <- 0

table(housing$basement)

housing$base[housing$basement == "yes"] <- 1
housing$base[housing$basement != "yes"] <- 0

table(housing$hotwaterheating)

housing$heat[housing$hotwaterheating == "yes"] <- 1
housing$heat[housing$hotwaterheating != "yes"] <- 0

table(housing$airconditioning)

housing$ac[housing$airconditioning == "yes"] <- 1
housing$ac[housing$airconditioning != "yes"] <- 0

table(housing$prefarea)

housing$area[housing$prefarea == "yes"] <- 1
housing$area[housing$prefarea != "yes"] <- 0

table(housing$furnishingstatus)

housing$furnished[housing$furnishingstatus == "furnished"] <- 1
housing$furnished[housing$furnishingstatus == "semi-furnished"] <- 0.5
housing$furnished[housing$furnishingstatus == "unfurnished"] <- 0

housing <- housing[,-6:-10]
View(housing)
housing <- housing[,-7:-8]
View(housing)
spec(housing)

validationIndex <- createDataPartition(housing$price, p = .8, list = FALSE)
validation <- housing[-validationIndex,]
house <- housing[validationIndex,]

dim(house)
sapply(house, class)

#jittered scatter plot matrix
housejitter <- sapply(house[,2:12], jitter)
pairs(housejitter, names(house[,2:12]), col = house$price)

#Correlation Plot#
correlations <- cor(house[,2:12])
corrplot(correlations, method = "circle")

#Run the algorithms using 10-fold cross-validation#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

#Linear Model#
set.seed(7)
fit.lm <- train(price ~ ., data = house, method = "lm", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Generalized Linear Model#
set.seed(7)
fit.glm <- train(price ~ ., data = house, method = "glm", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Cubist#
set.seed(7)
fit.cubist <- train(price ~ ., data = house, method = "cubist", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Partial Least Squares#
set.seed(7)
fit.pls <- train(price ~ ., data = house, method = "pls", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Random Forest#
set.seed(7)
fit.rf <- train(price ~ ., data = house, method = "rf", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

outcome <- resamples(list(LM = fit.lm, GLM = fit.glm, CUBIST = fit.cubist, PLS = fit.pls, RF = fit.rf))
summary(outcome)
dotplot(outcome)

#Test with the original Linear Model#
library(faraway)

g <- lm(price ~ ., data = house)
summary(g)

prediction <- predict(g, house, type = "response")
model_output <- cbind(house, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_price <- log(model_output$price)

rmse <- sqrt(mean((model_output$log_prediction - model_output$log_price)^2))
r2 <- cor(model_output$log_price, model_output$log_prediction)^2
print(rmse) #0.2388802#
print(r2) #0.5931977#

#Test with the General Linear Model#
g2 <- glm(price ~ ., family = gaussian, data = house)
summary(g2)

prediction2 <- predict(g2, house, type = "response")
model_output2 <- cbind(house, prediction2)

model_output2$log_prediction2 <- log(model_output2$prediction2)
model_output2$log_price <- log(model_output2$price)

rmse <- sqrt(mean((model_output2$log_prediction2 - model_output2$log_price)^2))
r2 <- cor(model_output2$log_price, model_output2$log_prediction2)^2
print(rmse) #0.2388802#
print(r2) #0.5931977#

#Test with Random Forest#
library(randomForest)

g3 <- randomForest(price ~ ., data = house)
summary(g3)

prediction3 <- predict(g3, house, type = "response")
model_output3 <- cbind(house, prediction3)

model_output3$log_prediction3 <- log(model_output3$prediction3)
model_output3$log_price <- log(model_output3$price)

rmse <- sqrt(mean((model_output3$log_prediction3 - model_output3$log_price)^2))
r2 <- cor(model_output3$log_price, model_output3$log_prediction3)^2
print(rmse) #0.1758118#
print(r2) #0.7971312#
