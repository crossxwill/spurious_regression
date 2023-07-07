set.seed(1)

x <- cumsum(rnorm(100))
y <- cumsum(rnorm(100))

df <- data.frame(x=x,
                 y=y)

train_df <- df[1:80,]
test_df <- df[81:100,]

# spurious regression and r-squared

spurious = lm(y ~x, data=train_df)

summary(spurious)

# helper function

f_rsquared <- function(mod, train, test, response_var_name='y'){
  
  constant <- mean(train[,response_var_name])
  actuals <- test[,response_var_name]
  preds <- predict(mod, newdata=test)
  
  MSE_null <- mean((actuals - constant)^2)  # variance
  MSE_mod <- mean((actuals - preds)^2)
  
  rsquared = (MSE_null - MSE_mod) / MSE_null # % of variance explained
  
  return(rsquared)
  
}

# training r-squared
f_rsquared(spurious, train_df, train_df)

# test r-squared
f_rsquared(spurious, train_df, test_df)
