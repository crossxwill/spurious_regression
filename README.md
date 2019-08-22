# Summary

Using the `forecast` package, I propose an approach to detect spurious regressions using time-series cross-validation. The method is as follows:

1. For each rolling subset of time-series data (i.e., rolling training data), fit a linear regression model and a competing naive model. A naive model uses only the last data point of the training data.

2. Measure the MSE of each model (i.e, regression and naive) using a rolling test data set that is `h=1` period ahead of the training data.

3. If the CV MSE of linear regression exceeds that of the naive model, then we can conclude that the regression is spurious. In other words, a regression is spurious if it fails to defeat a naive model.

It is recommended to use `h=1`. A large `h` may favor the regression model if both the response and the predictors share a strong deterministic trend.

# Helper Functions

The following function wraps `lm` into a function that can be passed into `tsCV`. The function was copied from [Rob Hyndman](https://stackoverflow.com/questions/50255912/timeseries-crossvalidation-in-r-using-tscv-with-tslm-models) on Stack Overflow.


```r
linRegCV <- function(y, h, xreg){
  if(NROW(xreg) < length(y) + h)
    stop("Not enough xreg data for forecasting")
  
  X <- xreg[seq_along(y),]
  fit <- tslm(y ~ X)
  X <- xreg[length(y)+seq(h),]
  fc <- forecast(fit, newdata=X, h=h)
  
  return(fc)
}
```


The following function runs `tsCV` on both a naive model and a linear regression model. Then the function computes the CV mean-squared error (CVMSE) of each model. It is recommended to use `h=1`.


```r
detectSpuriousRegCV <- function(y, h=1, xreg, initial=20){

  if(NROW(y) != NROW(xreg)) {
    warning('y and X are of different lengths. Data will be truncated. \n')
      
    minLength <- min(NROW(y), NROW(xreg))
    
    y <- head(y, minLength)
    xreg <- head(xreg, minLength)
    }
  
  linRegCVResiduals<- tsCV(y=y, linRegCV, h=h, xreg=xreg, initial=initial)
  naiveCVResiduals <- tsCV(y=y, naive, h=h, initial=initial)
  
  naiveCVResiduals[is.na(linRegCVResiduals)] <- NA # leads and lags in the regression may introduce NA
  
  MSE_lm <- mean(linRegCVResiduals^2, na.rm=TRUE)
  MSE_naive <- mean(naiveCVResiduals^2, na.rm=TRUE)
      
  out <- list(CV_MSE_lm=MSE_lm, CV_MSE_naive=MSE_naive, SpuriousRegression=MSE_lm>MSE_naive)
  
  return(out)
}
```


# An example of a spurious regression

Is the number of Air Transport Passengers in Australia related to rice production in the country of Guinea?


```r
data(ausair)
data(guinearice)

detectSpuriousRegCV(y=ausair, h=1, xreg=guinearice)
```

```
## Warning in detectSpuriousRegCV(y = ausair, h = 1, xreg = guinearice): y and X are of different lengths. Data will be truncated.
```

```
## $CV_MSE_lm
## [1] 21.53863
## 
## $CV_MSE_naive
## [1] 10.77681
## 
## $SpuriousRegression
## [1] TRUE
```


# An example of a non-spurious regression

Is consumption related to income?


```r
data(uschange)

consumption <- uschange[,'Consumption']
income <- uschange[,'Income']

detectSpuriousRegCV(y=consumption, h=1, xreg=income)
```

```
## $CV_MSE_lm
## [1] 0.3625108
## 
## $CV_MSE_naive
## [1] 0.4800927
## 
## $SpuriousRegression
## [1] FALSE
```

# Simulating spurious regressions

Simulate 100 pairs of non-stationary time series with drift and trend.


```r
simulateSpuriousData <- function(seed, ndrift=0.01, ntrend=0.02){
  nobs <- 60
  set.seed(seed)
  
  Xts <- ts(cumsum(ndrift + ntrend*seq_len(nobs) + rnorm(nobs))) # Random-walk with drift and trend
  Yts <- ts(cumsum(ndrift + ntrend*seq_len(nobs) + rnorm(nobs))) # Random-walk with drift and trend
  
  return(list(Xts=Xts, Yts=Yts))
}

manySpuriousData <- map(1:100, simulateSpuriousData)
```

For each pair, run `detectSpuriousRegCV` with `h=3`.


```r
simulateSpuriousRegression <- map(manySpuriousData, function(lstData){
  out <- detectSpuriousRegCV(y=lstData$Yts, xreg=lstData$Xts, h=3)
  
  return(out$SpuriousRegression)
})

mean(unlist(simulateSpuriousRegression))
```

```
## [1] 0.95
```

Based on the simulation, `detectSpuriousRegCV` correctly found the spurious regression 95% of the time with `h=3`.

Since there is a strong deterministic trend in both pairs of time series, the `h=3` parameter is poorly chosen. Use `h=1` instead.


```r
simulateSpuriousRegressionh1 <- map(manySpuriousData, function(lstData){
  out <- detectSpuriousRegCV(y=lstData$Yts, xreg=lstData$Xts, h=1)
  
  return(out$SpuriousRegression)
})

mean(unlist(simulateSpuriousRegressionh1))
```

```
## [1] 1
```

Based on the simulation, `detectSpuriousRegCV` correctly found the spurious regression 100% of the time with `h=1`.

Simulate another 100 pairs of non-stationary time series with very strong drift and trend.


```r
manySpuriousDataStrongTrend <- map(1:100, simulateSpuriousData, ndrift=0.02, ntrend=0.04)
```

For each pair, run `detectSpuriousRegCV` with `h=3`.


```r
simulateSpuriousRegressionStrongTrend_h3 <- map(manySpuriousDataStrongTrend, function(lstData){
  out <- detectSpuriousRegCV(y=lstData$Yts, xreg=lstData$Xts, h=3)
  
  return(out$SpuriousRegression)
})

mean(unlist(simulateSpuriousRegressionStrongTrend_h3))
```

```
## [1] 0.45
```

With `h=3`, strong deterministic trends reduce the effectiveness of `detectSpuriousRegCV`.

For each pair, run `detectSpuriousRegCV` with `h=1`.


```r
simulateSpuriousRegressionStrongTrend_h1 <- map(manySpuriousDataStrongTrend, function(lstData){
  out <- detectSpuriousRegCV(y=lstData$Yts, xreg=lstData$Xts, h=1)
  
  return(out$SpuriousRegression)
})

mean(unlist(simulateSpuriousRegressionStrongTrend_h1))
```

```
## [1] 0.95
```

Based on the simulation, `detectSpuriousRegCV` correctly found the spurious regression 95% of the time with `h=1`. When detecting spurious regression with cross-validation, always set `h=1`.

# Further work

1. How does `detectSpuriousRegCV` perform on very long non-stationary time series (n > 1000)?

2. Could we modify the naive model to include drift and trend?

3. How does `detectSpuriousRegCV`perform on co-integrated time series?

4. How does `detectSpuriousRegCV` benchmark against popular co-integration tests like Engle-Granger and Phillips–Ouliaris?
