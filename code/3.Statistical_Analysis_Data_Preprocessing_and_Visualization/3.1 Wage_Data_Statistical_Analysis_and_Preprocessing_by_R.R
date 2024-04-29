# Install necessary packages
install.packages("lmtest")
install.packages("car")
install.packages("bootstrap")

# Import required packages
library(lmtest)
library(car)
library(bootstrap)

# Data analysis begins
setwd("/Users/a1234/Desktop/workspace/Employment_Analysis_and_Recommendation_System_Based_on_NLP_and_Data_Modeling/data/")
wage <- read.csv("Wage_sample.csv", header = TRUE)
attach(wage)

# Construct data frame
y <- wage$Salary
x1 <- wage$Sex
x2 <- wage$Race
x3 <- wage$Age
x4 <- wage$Edlevel
x5 <- wage$Work
x6 <- wage$Union
x7 <- wage$Time
x8 <- wage$Marr
x9 <- wage$Jobcat
data1 <- data.frame(y = y, x1, x2, x3, x4, x5, x6, x7, x8, x9)

# QQ plot
op = par(mfrow = c(3, 4))
qqnorm(x1, main = "x1")
qqline(x1, col = "red")
qqnorm(x2, main = "x2")
qqline(x2, col = "red")
qqnorm(x3, main = "x3")
qqline(x3, col = "red")
qqnorm(x4, main = "x4")
qqline(x4, col = "red")
qqnorm(x5 * x5, main = "x5")
qqline(x5 * x5, col = "red")
qqnorm(x6, main = "x6")
qqline(x6, col = "red")
qqnorm(x7, main = "x7")
qqline(x7, col = "red")
qqnorm(x8, main = "x8")
qqline(x8, col = "red")
qqnorm(x9, main = "x9")
qqline(x9, col = "red")
qqnorm(y, main = "y")
qqline(y, col = "red")
par(op)

# Scatterplot matrix
scatterplotMatrix(data1, spread = FALSE, lty.smooth = 2, main = "Scatter Plot Matrix")

# Regression analysis
lm1 <- lm(y ~ ., data = data1)
reg1 <- summary(lm1)
reg1

# Breusch-Pagan test
ncvTest(lm1)

# Durbin-Watson test
dwtest(lm1)

# Residual analysis
par(mfrow = c(2, 2))
plot(lm1)

# Log transformation of the dependent variable
trans1 <- lm(log(y) ~ ., data = data1)
lm2 <- trans1
summary(lm2)

par(mfrow = c(2, 2))
plot(lm2)

# Breusch-Pagan test
ncvTest(lm2)

# Stepwise regression
step(trans1, direction = c("backward"))
step <- lm(log(y) ~ x2 + x3 + x4 + x5 + x6 + x7 + x9, data = data1)
lm3 <- step
summary(lm3)

# Breusch-Pagan test
ncvTest(lm3)

# Outlier test
outlierTest(lm3)

# Remove outliers
data_out1 <- data1[-c(550), ]
lm_out1 <- lm(log(y) ~ x2 + x3 + x4 + x5 + x6 + x7 + x9, data = data_out1)
summary(lm_out1)

# Influence measures
influence.measures(lm_out1)

# Influence plot
op <- par(mfrow = c(2, 2), mar = 0.4 + c(4, 4, 1, 1), oma = c(0, 0, 2, 0))
plot(step, 1:4)
par(op)

# Regression analysis after removing outliers
data_out2 <- data1[-c(128, 234, 550, 630), ]
lm_out2 <- lm(log(y) ~ x2 + x3 + x4 + x5 + x6 + x7 + x9, data = data_out2)
lm4 <- lm_out2
summary(lm4)

# Rename columns of processed data frame to original column names
colnames(data_out2) <- c("Salary", "Sex", "Race", "Age", "Edlevel", 
                         "Work", "Union", "Time", "Marr", "Jobcat")

# Save processed data to CSV file
write.csv(data_out2, file = "processed_data.csv", row.names = FALSE)

# Breusch-Pagan test
ncvTest(lm4)

# Correlation analysis
cor(data_out2)


lm5 <- lm(log(y) ~ x2 + x3 + x4 + x5 + x6 + x7 + x9 + x2 * x6 + x3 * x5 + x4 * x9 + x1 * x2 + x7 * x9, data = data_out2)
summary(lm5)

# VIF test
library(car)
vif(lm5)

# Prediction error calculation
MMAE <- vector(length = 100)

test <- function(n, model) {
  for (i in 1:n) {
    tr <- sample(nrow(wage), 0.7 * nrow(wage))
    te <- sample(nrow(wage), 0.3 * nrow(wage))
    
    train. <- wage[tr, ]
    test. <- wage[te, ]
    
    y <- train.$Salary
    x1 <- train.$Sex
    x2 <- train.$Race
    x3 <- train.$Age
    x4 <- train.$Edlevel
    x5 <- train.$Work
    x6 <- train.$Union
    x7 <- train.$Time
    x8 <- train.$Marr
    x9 <- train.$Jobcat
    data1 <- data.frame(y = y, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    
    y_te <- test.$Salary
    x1_te <- test.$Sex
    x2_te <- test.$Race
    x3_te <- test.$Age
    x4_te <- test.$Edlevel
    x5_te <- test.$Work
    x6_te <- test.$Union
    x7_te <- test.$Time
    x8_te <- test.$Marr
    x9_te <- test.$Jobcat
    data_te <- data.frame(x1_te, x2_te, x3_te, x4_te, x5_te, x6_te, x7_te, x8_te, x9_te)
    colnames(data_te) <- c("x1", "x2", "x3", "x4", 'x5', "x6", "x7", "x8", "x9")
    
    predict(model, data_te)
    test <- predict(model, data_te)
    
    y <- log(y_te)
    n <- length(y)
    
    MMAE[i] <- sum(abs(y - exp(test))) / (700 - n)
  }
  mean(MMAE)
}

test(1000, lm5)

# Press statistic
Press <- function(themodel) {
  theResiduals <- themodel$residuals
  X <- model.matrix(themodel)
  return(sum((theResiduals / (1 - hat(X)))^2))
}
Press(reg1)

# Model assessment
Press_stat <- function(themodel) {
  theResiduals <- themodel$residuals
  X <- model.matrix(themodel)
  return(sum((theResiduals / (1 - hat(X)))^2))
}
CP_stat <- function(themodel, fullmodel) {
  p <- length(themodel$coefficients)
  n <- nrow(themodel$model)
  sp <- summary(themodel)$sigma
  s <- summary(fullmodel)$sigma
  bias <- (sp^2 - s^2) * (n - p) / (s^2)
  return(p + bias)
}
level <- function(T) {
  a <- summary(T)$r.squared
  b <- summary(T)$adj.r.squared
  c <- summary(T)$fstatistic[1] 
  d <- summary(T)$coefficients[1, 4]
  if (d == 0 ) {
    d = "< 2.2e-16"
  }
  e <- Press_stat(T)
  f <- CP_stat(T, lm1)
  cat("r.squared:", a,
      "adj.r.squared:", b,
      "F:", c,
      "P-value:", d,
      "PRESS statistic:", e,
      "Cp statistic:", f)
}
level(lm5)  # Here you need to pass the model you want to assess, here I passed lm5


# QQ plot
qqplot_path <- "/Users/a1234/Desktop/workspace/Employment_Analysis_and_Recommendation_System_Based_on_NLP_and_Data_Modeling/Image/qqplot.png"
png(qqplot_path, res = 50)  # Increase resolution to 300 ppi
op = par(mfrow = c(6, 2))  # Reduce the number of plots in each row
qqnorm(x1, main = "x1")
qqline(x1, col = "red")
qqnorm(x2, main = "x2")
qqline(x2, col = "red")
qqnorm(x3, main = "x3")
qqline(x3, col = "red")
qqnorm(x4, main = "x4")
qqline(x4, col = "red")
qqnorm(x5 * x5, main = "x5")
qqline(x5 * x5, col = "red")
qqnorm(x6, main = "x6")
qqline(x6, col = "red")
qqnorm(x7, main = "x7")
qqline(x7, col = "red")
qqnorm(x8, main = "x8")
qqline(x8, col = "red")
qqnorm(x9, main = "x9")
qqline(x9, col = "red")
qqnorm(y, main = "y")
qqline(y, col = "red")
par(op)
dev.off()

# Scatterplot matrix
scatterplot_path <- "/Users/a1234/Desktop/workspace/Employment_Analysis_and_Recommendation_System_Based_on_NLP_and_Data_Modeling/Image/scatterplot.png"
png(scatterplot_path, res = 100)  # Increase resolution to 300 ppi
scatterplotMatrix(data1, spread = FALSE, lty.smooth = 2, main = "Scatter Plot Matrix")
dev.off()

# Residual plots
residual_path <- "/Users/a1234/Desktop/workspace/Employment_Analysis_and_Recommendation_System_Based_on_NLP_and_Data_Modeling/Image/residual_plots.png"
png(residual_path, res = 100)  # Increase resolution to 300 ppi
par(mfrow = c(2, 2))
plot(lm1)
dev.off()

# Influence plot
influence_path <- "/Users/a1234/Desktop/workspace/Employment_Analysis_and_Recommendation_System_Based_on_NLP_and_Data_Modeling/Image/influence_plot.png"
png(influence_path, res = 100)  # Increase resolution to 300 ppi
op <- par(mfrow = c(2, 2), mar = 0.4 + c(4, 4, 1, 1), oma = c(0, 0, 2, 0))
plot(step, 1:4)
par(op)
dev.off()

