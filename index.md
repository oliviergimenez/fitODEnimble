---
title: "ODEs and Nimble"
output: 
  html_document:
    keep_md: true
---



## Motivation

I got recently interested in fitting dynamic models specified as a system of ODEs with some observation error on top of it, see e.g. our recent paper [here](https://www.sciencedirect.com/science/article/abs/pii/S2211675320300221) on the colonization of wolf in France (pdf [there](https://oliviergimenez.github.io/pubs/Louvrier2020SpatialStatistics.pdf)).

Also, there has been much activity about dynamic models in relation to the covid-19 pandemic (e.g. [this]( https://statmodeling.stat.columbia.edu/2020/04/02/more-coronavirus-research-using-stan-to-fit-differential-equation-models-in-epidemiology/) or [that](https://www.medrxiv.org/content/10.1101/2020.03.22.20040915v1.article-info)), and I'd like to better understand how these models work, and how parameters are estimated, just out of curiosity.

Now there are several software options to implement these models, such as `OpenBUGS` (see example [here](https://github.com/oliviergimenez/fitODEswithOpenBUGS)), `Jags` pending some tweaks (see [mecastat](https://gitlab.paca.inra.fr/jfrey/jags-module/tree/19b5aefe1ff1b4cf494461650820f802eadc7ff5/mecastat)), `Stan` (see example [here](https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html)) or other packages like [deBInfer](https://github.com/pboesu/debinfer), [pomp](http://kingaa.github.io/pomp/), [fitR](http://sbfnk.github.io/mfiidd/index.html) or [fitode](https://rdrr.io/github/parksw3/fitode/man/fitode.html) and probably others I do not know of. 

There are also nice introductions to the topic, like [the awesome short course *Model-based Inference in Ecology and Epidemiology* by Aaron King](http://kingaa.github.io/short-course/) and the [book *Epidemics: Models and Data in R* by Ottar N. Bjornstad](https://www.springer.com/gp/book/9783319974866) which comes with [R codes](https://github.com/objornstad/epimdr). 

Here I show how to fit ODE-type models to noisy data using `Nimble`, *a system for building and sharing analysis methods for statistical models, especially for hierarchical models and computationally-intensive methods*, see [here](https://r-nimble.org/) for more details. We begin by writing a `Nimble` function to wrap the `R` ODE solver `deSolve::ode()` that is often used. Thanks Daniel Turek for his help! We first consider a 1D ODE example, then we move to >1D ODE examples for fun. Last, we fit models that are specified with a system of ODEs to noisy data.

Let's load the `Nimble` package:

```r
library(nimble)
```

## 1D example

Our first example is a simple logistic growth model $dy/dt= r y (1 - y/K)$. We wrap the `deSolve::ode()` function in a function `R_ode` so that we do not have to parse the name of a function (here `logistic`) as an argument:

```r
R_ode <- function(y, times, params) {
    logistic <- function(t, y, parms) return(list(parms[1] * y * (1 - y / parms[2])))
    result <- deSolve::ode(y, times, logistic, params)
    x <- result[,-1]
    return(x)
}
```

Now we write the `Nimble` function with `nimbleRcall`:

```r
nimble_ode <- nimbleRcall(
    prototype = function(
        y = double(), # y is a scalar
        times = double(1), # times is a vector
        params = double(1) # params is a vector
    ) {},
    returnType = double(1), # outcome is a vector
    Rfun = 'R_ode'
)
```

Set some values:

```r
y <- 0.1 # initial value
times <- seq(from = 0, to = 10, by = 0.01) # time sequence
params <- c(1.5, 10) # r and K
```

Write the `Nimble` code:

```r
code <- nimbleCode({
    xOde[1:1001, 1] <- nimble_ode(y, times[1:1001], params[1:2])
})
```

Build and compile le model:

```r
constants <- list()
data <- list()
inits <- list(y = y, times = times, params = params)
Rmodel <- nimbleModel(code, constants, data, inits)
Cmodel <- compileNimble(Rmodel)
#Rmodel$calculate()   ## not NA
#Cmodel$calculate()   ## not NA
```

Solve ODE using our brand new `Nimble` function:

```r
ode_Nimble <- Rmodel$xOde
head(ode_Nimble)
```

```
##           [,1]
## [1,] 0.1000000
## [2,] 0.1014960
## [3,] 0.1030141
## [4,] 0.1045547
## [5,] 0.1061181
## [6,] 0.1077046
```

Solve the ODE with native `R` function `deSolve::ode()`:

```r
logistic <- function(t, y, parms) return(list(parms[1] * y * (1 - y / parms[2])))
ode_nativeR <- deSolve::ode(y, times, logistic, params)[, -1]
head(ode_nativeR)
```

```
## [1] 0.1000000 0.1014960 0.1030141 0.1045547 0.1061181 0.1077046
```
Compare the two:

```r
sum(ode_Nimble - ode_nativeR)
```

```
## [1] 0
```

```r
sum(ode_Nimble - ode_nativeR)
```

```
## [1] 0
```

OK, we're all good. Now let's have a look to a more complex example. 

## 4D example

I use data from a paper by Witkowski and Brais entitled [Bayesian Analysis of Epidemics - Zombies, Influenza, and other Diseases](https://arxiv.org/abs/1311.6376). The authors provide a Python notebook [here](https://gist.github.com/bblais/181abd99f878282666b98a29588dda41). The authors propose the following SEIR epidemic model (more [here](https://github.com/oliviergimenez/SIRcovid19)) to capture the zombie apocalypse:
$$
dS/dt = - \beta S Z \\
dE/dt = \beta S Z - \zeta E \\
dZ/dt = \zeta E - \alpha S Z \\
dR/dt = \alpha S Z
$$
Let us solve this system of ODEs. First, set up some values:

```r
y <- c(508.2, 0, 1, 0) # initial values
times <- seq(from = 0, to = 50, by = 1) # time sequence
params <- c(0.2, 6, 0, 508.2) # beta, zeta, alpha, Sinit
```

Write the system of ODEs:

```r
shaun <- function(t, y, parms){
  dy1 <- - parms[1] * y[1] * y[3] / parms[4] # S
  dy2 <- parms[1] * y[1] * y[3] / parms[4] - parms[2] * y[2] # E
  dy3 <- parms[2] * y[2] - parms[3] * y[1] * y[3] # Z
  dy4 <- parms[3] * y[1] * y[3] # R
  return(list(c(dy1, dy2, dy3, dy4)))
} 
ode_nativeR <- deSolve::ode(y, times, shaun, params)[, -1]
head(ode_nativeR)
```

```
##             1          2        3 4
## [1,] 508.2000 0.00000000 1.000000 0
## [2,] 507.9851 0.03792698 1.176998 0
## [3,] 507.7255 0.04608553 1.428391 0
## [4,] 507.4107 0.05589356 1.733388 0
## [5,] 507.0290 0.06777190 2.103251 0
## [6,] 506.5662 0.08214955 2.551646 0
```

Now on to `Nimble`, with the wraper first:

```r
R_ode <- function(y, times, params) {
  shaun <- function(t, y, parms){
    dy1 <- - parms[1] * y[1] * y[3] / parms[4] # S
    dy2 <- parms[1] * y[1] * y[3] / parms[4] - parms[2] * y[2] # E
    dy3 <- parms[2] * y[2] - parms[3] * y[1] * y[3] # Z
    dy4 <- parms[3] * y[1] * y[3] # R
    return(list(c(dy1, dy2, dy3, dy4)))
}    
  result <- deSolve::ode(y, times, shaun, params)
  x <- result[,-1]
  return(x)
}
```

Now we write the `Nimble` function with `nimbleRcall`:

```r
nimble_ode <- nimbleRcall(
    prototype = function(
        y = double(1), # y is a vector
        times = double(1), # times is a vector
        params = double(1) # params is a vector
    ) {},
    returnType = double(2), # outcome is a matrix
    Rfun = 'R_ode'
)
```

Write the `Nimble` code:

```r
code <- nimbleCode({
    xOde[1:51, 1:4] <- nimble_ode(y[1:4], times[1:51], params[1:4])
})
```

Build and compile le model:

```r
constants <- list()
data <- list()
inits <- list(y = y, times = times, params = params)
Rmodel <- nimbleModel(code, constants, data, inits)
Cmodel <- compileNimble(Rmodel)
#Rmodel$calculate()   ## not NA
#Cmodel$calculate()   ## not NA
```

Solve ODE using our brand new `Nimble` function:

```r
ode_Nimble <- Rmodel$xOde
head(ode_Nimble)
```

```
##          [,1]       [,2]     [,3] [,4]
## [1,] 508.2000 0.00000000 1.000000    0
## [2,] 507.9851 0.03792698 1.176998    0
## [3,] 507.7255 0.04608553 1.428391    0
## [4,] 507.4107 0.05589356 1.733388    0
## [5,] 507.0290 0.06777190 2.103251    0
## [6,] 506.5662 0.08214955 2.551646    0
```

Compare with native `R`:

```r
sum(ode_Nimble - ode_nativeR)
```

```
## [1] 0
```

```r
sum(ode_Nimble - ode_nativeR)
```

```
## [1] 0
```

It works, awesome ! 

## Add gaussian noise

Now let us have a look to a system with some observation error, which we assume is gaussian. We go on with the zombie example. Briefly speaking, the authors counted the number of living deads in several famous zombie movies. I use the data from [Shaun of the Dead](https://www.imdb.com/title/tt0365748/).

First, we read in the data:

```r
tgrid <- c(0.00, 3.00, 5.00, 6.00, 8.00, 10.00, 22.00, 22.20, 22.50, 24.00, 25.50, 
           26.00, 26.50, 27.50, 27.75, 28.50, 29.00, 29.50, 31.50) 
zombies <- c(0, 1, 2, 2, 3, 3, 4, 6, 2, 3, 5, 12, 15, 25, 37, 25, 65, 80, 100)
```

Now the code:

```r
code <- nimbleCode({
  
  # system of ODEs
  xOde[1:ngrid, 1:ndim] <- nimble_ode(y[1:ndim], 
                                      times[1:ngrid], 
                                      params[1:ndim])
  # priors on parameters
  params[1] ~ dunif(0, 10) # beta
  params[2] ~ dunif(0, 1) # zeta
  params[3] ~ dunif(0, 0.01) # alpha
  params[4] ~ dunif(300, 600) # Sinit
  
  # observation error
  for (i in 1:ngrid){
    obs_x[i] ~ dnorm(xOde[i, 3], tau.x)
  }
  
  # prior on error sd
  tau.x <- 1 / var.x
  var.x <- 1 / (sd.x * sd.x)
  sd.x ~ dunif(0, 5)
})
```

Specify the constants, data and initial values:

```r
# constants
constants <- list(ngrid = 19, 
                  ndim = 4)

# data (pass times and y in constants?)
data <- list(times = tgrid,
             obs_x = zombies,
             y = c(508.2, 0, 1, 0))

# initial values
inits <- list(sdx = 2)
```

Get ready:

```r
Rmodel <- nimbleModel(code, constants, data, inits)
# Rmodel$calculate()   ## NA...
conf <- configureMCMC(Rmodel)
```

```
## ===== Monitors =====
## thin = 1: params, sd.x
## ===== Samplers =====
## RW sampler (5)
##   - params[]  (4 elements)
##   - sd.x
```

```r
conf$printMonitors()
```

```
## thin = 1: params, sd.x
```


```r
conf$printSamplers(byType = TRUE)
```

```
## RW sampler (5)
##   - params[]  (4 elements)
##   - sd.x
```


```r
Rmcmc <- buildMCMC(conf)
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)
```

Unleash the beast:

```r
samplesList <- runMCMC(Cmcmc, 5000, 
                       nburnin = 1000,
                       nchains = 2,
                       samplesAsCodaMCMC = TRUE)
```

```
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
```

Check out convergence:

```r
library(coda)
gelman.diag(samplesList)
```

```
## Potential scale reduction factors:
## 
##           Point est. Upper C.I.
## params[1]       3.75       8.21
## params[2]       4.11      14.78
## params[3]       1.23       1.38
## params[4]       6.38      14.06
## sd.x            1.35       2.12
## 
## Multivariate psrf
## 
## 5.53
```

Visualize traceplots and posterior distributions:

```r
library(basicMCMCplots)
chainsPlot(samplesList)
```

![](index_files/figure-html/unnamed-chunk-26-1.png)<!-- -->

Apart from the standard deviation of the observation error, the mixing is poor. This is what I had with `OpenBUGS` too, see [here](https://github.com/oliviergimenez/fitODEswithOpenBUGS/blob/master/README.md). 

Can we do something about that? Hopefully yes, and this is what's great with `Nimble`, you have full control of the underlying MCMC machinery. There are useful advices [here](https://r-nimble.org/better-block-sampling-in-mcmc-with-the-automated-factor-slice-sampler). One reason for poor mixing is correlation in parameters. Let's have a look then. 

```r
cor(samplesList$chain1)
```

```
##             params[1]  params[2]   params[3]  params[4]        sd.x
## params[1]  1.00000000  0.2565799  0.44972423  0.5901333 -0.05426149
## params[2]  0.25657987  1.0000000  0.72664647  0.5657761 -0.23919060
## params[3]  0.44972423  0.7266465  1.00000000  0.1436804 -0.06174826
## params[4]  0.59013326  0.5657761  0.14368042  1.0000000 -0.25934127
## sd.x      -0.05426149 -0.2391906 -0.06174826 -0.2593413  1.00000000
```

```r
cor(samplesList$chain2)
```

```
##            params[1]  params[2]  params[3]  params[4]       sd.x
## params[1]  1.0000000 -0.4693463  0.3545649 -0.2484769  0.1997804
## params[2] -0.4693463  1.0000000 -0.1613304  0.7991534 -0.3698920
## params[3]  0.3545649 -0.1613304  1.0000000 -0.5939054  0.2332505
## params[4] -0.2484769  0.7991534 -0.5939054  1.0000000 -0.4136865
## sd.x       0.1997804 -0.3698920  0.2332505 -0.4136865  1.0000000
```

Well, we don't have strong correlations, but let's pretend we do for the sake of illustration. Say $\beta$ (params[1]) and $\zeta$ (params[2]) are correlated for example, and let's use block sampling to try and improve mixing. 

First, ask what are the samples used currently: 

```r
conf$printSamplers()
```

```
## [1] RW sampler: params[1]
## [2] RW sampler: params[2]
## [3] RW sampler: params[3]
## [4] RW sampler: params[4]
## [5] RW sampler: sd.x
```

OK, now remove the default samplers for params[1] and params[3] and use a block random walk instead:

```r
conf$removeSamplers(c('params[1]', 'params[2]'))
conf$printSamplers()
```

```
## [1] RW sampler: params[3]
## [2] RW sampler: params[4]
## [3] RW sampler: sd.x
```

```r
conf$addSampler(target = c('params[1]', 'params[2]'),
                type = 'RW_block')
conf$printSamplers()
```

```
## [1] RW sampler: params[3]
## [2] RW sampler: params[4]
## [3] RW sampler: sd.x
## [4] RW_block sampler: params[1], params[2]
```

Now rebuild, recompile and rerun:

```r
Rmcmc <- buildMCMC(conf)
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)
samplesList <- runMCMC(Cmcmc, 2500, 
                       nburnin = 1000,
                       nchains = 2,
                       samplesAsCodaMCMC = TRUE)
```

```
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
```

Visualize traceplots and posterior distributions:

```r
chainsPlot(samplesList)
```

![](index_files/figure-html/unnamed-chunk-31-1.png)<!-- -->

OK, that's disappointing, again. What if we use another sampler?

```r
Rmodel <- nimbleModel(code, constants, data, inits)
conf <- configureMCMC(Rmodel)
```

```
## ===== Monitors =====
## thin = 1: params, sd.x
## ===== Samplers =====
## RW sampler (5)
##   - params[]  (4 elements)
##   - sd.x
```

```r
conf$printSamplers()
```

```
## [1] RW sampler: params[1]
## [2] RW sampler: params[2]
## [3] RW sampler: params[3]
## [4] RW sampler: params[4]
## [5] RW sampler: sd.x
```

```r
conf <- configureMCMC(Rmodel, onlySlice = TRUE)
```

```
## ===== Monitors =====
## thin = 1: params, sd.x
## ===== Samplers =====
## slice sampler (5)
##   - params[]  (4 elements)
##   - sd.x
```

```r
conf$printSamplers()
```

```
## [1] slice sampler: params[1]
## [2] slice sampler: params[2]
## [3] slice sampler: params[3]
## [4] slice sampler: params[4]
## [5] slice sampler: sd.x
```

Now rebuild, recompile and rerun:

```r
Rmcmc <- buildMCMC(conf)
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)
samplesList <- runMCMC(Cmcmc, 2000, 
                       nburnin = 1000,
                       nchains = 2,
                       samplesAsCodaMCMC = TRUE)
```

```
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
```

Visualize traceplots and posterior distributions:

```r
chainsPlot(samplesList)
```

![](index_files/figure-html/unnamed-chunk-34-1.png)<!-- -->

Much slower, and doesn't improve mixing.

Last chance:

```r
conf$removeSamplers(c('params[1]', 'params[2]'))
conf$printSamplers()
```

```
## [1] slice sampler: params[3]
## [2] slice sampler: params[4]
## [3] slice sampler: sd.x
```

```r
conf$addSampler(target = c('params[1]', 'params[2]'),
                type = 'AF_slice')
conf$printSamplers()
```

```
## [1] slice sampler: params[3]
## [2] slice sampler: params[4]
## [3] slice sampler: sd.x
## [4] AF_slice sampler: params[1], params[2]
```

```r
Rmcmc <- buildMCMC(conf)
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)
samplesList <- runMCMC(Cmcmc, 2500, 
                       nburnin = 1000,
                       nchains = 2,
                       samplesAsCodaMCMC = TRUE)
```

```
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
```

```r
chainsPlot(samplesList)
```

![](index_files/figure-html/unnamed-chunk-35-1.png)<!-- -->

## Can we call external code to make the ODE solver faster?

Note that in my attempts above, I call an ODE solver through `R`, which is the slow option. Going full `C/C++` would speed things up. And indeed we can call external code, some `C` code for example, as explained in the help file of function `nimbleExternalCall`. More soon on that.  

```r
?nimbleExternalCall
## Not run: 
sink('add1.h')
cat('
 extern "C" {
 void my_internal_function(double *p, double*ans, int n);
 }
')
sink()
sink('add1.cpp') 
cat('
 #include <cstdio>
 #include "add1.h"
 void my_internal_function(double *p, double *ans, int n) {
   printf("In my_internal_function\\n");
     /* cat reduces the double slash to single slash */ 
   for(int i = 0; i < n; i++) 
     ans[i] = p[i] + 1.0;
 }
')
sink()
system('g++ add1.cpp -c -o add1.o')
Radd1 <- nimbleExternalCall(function(x = double(1), ans = double(1),
n = integer()){}, Cfun =  'my_internal_function',
headerFile = file.path(getwd(), 'add1.h'), returnType = void(),
oFile = file.path(getwd(), 'add1.o'))
## If you need to use a function with non-scalar return object in model code,
## you can wrap it  in another nimbleFunction like this:
model_add1 <- nimbleFunction(
     run = function(x = double(1)) {
         ans <- numeric(length(x))
         Radd1(x, ans, length(x))
         return(ans)
         returnType(double(1))
     })
demoCode <- nimbleCode({
     for(i in 1:4) {x[i] ~ dnorm(0,1)} ## just to get a vector
     y[1:4] <- model_add1(x[1:4])
})
demoModel <- nimbleModel(demoCode, inits = list(x = rnorm(4)),
check = FALSE, calculate = FALSE)
CdemoModel <- compileNimble(demoModel, showCompilerOutput = TRUE)

## End(Not run)
```

## Last one for the road

In a very nice case study, Bob Carpenter shows how to estimate predator-prey population dynamics using Stan, check out [here](https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html). Let's try and reproduce his results. 

Let's get the data first:

```r
library(tidyverse)
dat <- read_csv('https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/lotka-volterra/hudson-bay-lynx-hare.csv', comment = '#')
dat
```

```
## # A tibble: 21 × 3
##     Year  Lynx  Hare
##    <dbl> <dbl> <dbl>
##  1  1900   4    30  
##  2  1901   6.1  47.2
##  3  1902   9.8  70.2
##  4  1903  35.2  77.4
##  5  1904  59.4  36.3
##  6  1905  41.7  20.6
##  7  1906  19    18.1
##  8  1907  13    21.4
##  9  1908   8.3  22  
## 10  1909   9.1  25.4
## # ℹ 11 more rows
```

Visualize:

```r
dat %>%
  pivot_longer(cols = c('Lynx','Hare'),
               names_to = 'species',
               values_to = 'counts') %>%
ggplot(aes(x = Year, y = counts, color = species)) +
geom_line(size = 0.75) +
geom_point(size = 1.5) +
ylab("Counts (thousands)")
```

![](index_files/figure-html/unnamed-chunk-38-1.png)<!-- -->

Now on to `Nimble`, with the wraper first:

```r
# parms = (alpha, beta, gamma, delta)
R_ode <- function(y, times, params) {
  lotka <- function(t, y, parms){
    dy1 <- (parms[1] - parms[2] * y[2]) * y[1] # prey
    dy2 <- (-parms[3] + parms[4] * y[1]) * y[2] # predator
    return(list(c(dy1, dy2)))
}    
  result <- deSolve::ode(y, times, lotka, params, method = "rk4")
  x <- result[,-1]
  return(x)
}
```

Now we write the `Nimble` function with `nimbleRcall`:

```r
nimble_ode <- nimbleRcall(
    prototype = function(
        y = double(1), # y is a vector
        times = double(1), # times is a vector
        params = double(1) # params is a vector
    ) {},
    returnType = double(2), # outcome is a matrix
    Rfun = 'R_ode'
)
```

Now the code:

```r
code <- nimbleCode({
  
  # system of ODEs
  xOde[1:ngrid, 1:ndim] <- nimble_ode(z_init[1:ndim], 
                                      times[1:ngrid], 
                                      params[1:4])
  # priors on parameters
  params[1] ~ dnorm(1, sd = 0.5) # alpha
  params[2] ~ dnorm(0.05, sd = 0.05) # beta
  params[3] ~ dnorm(1, sd = 0.5) # gamma
  params[4] ~ dnorm(0.05, sd = 0.05) # delta
  
  # observation error
  for (i in 1:ngrid){
    obs_prey[i] ~ dnorm(xOde[i, 1], sd = sdy[1])
    obs_pred[i] ~ dnorm(xOde[i, 2], sd = sdy[2])
  }
  for (k in 1:ndim){
    yinit[k] ~ dnorm(z_init[k], sdy[k])
  }
  # prior on error sd
  for (j in 1:ndim){
    sdy[j] ~ dunif(0,3)
    z_init[j] ~ dnorm(log(10), sd = 1)
  }
})
```

Specify the constants, data and initial values:

```r
N <- length(dat$Year) - 1
ts <- 1:N
y_init <- c(dat$Hare[1], dat$Lynx[1])
y <- as.matrix(dat[2:(N + 1), 2:3]) # lynx, hare

# constants
constants <- list(ngrid = N, 
                  ndim = 2)

# data (pass times and y in constants?)
data <- list(times = ts,
             obs_pred = log(y[,1]),
             obs_prey = log(y[,2]),
             yinit = log(y_init))

# initial values
inits <- list(params = c(1, 0.5, 1, 0.5),
              sdy = c(0.2, 0.2),
              z_init = log(c(10, 10)))
```

Get ready:

```r
Rmodel <- nimbleModel(code, constants, data, inits)
Rmodel$calculate() # -905...
```

```
## [1] -904.6105
```

```r
conf <- configureMCMC(Rmodel)
```

```
## ===== Monitors =====
## thin = 1: params, sdy, z_init
## ===== Samplers =====
## RW sampler (8)
##   - params[]  (4 elements)
##   - sdy[]  (2 elements)
##   - z_init[]  (2 elements)
```

```r
conf$printMonitors()
```

```
## thin = 1: params, sdy, z_init
```


```r
conf$printSamplers(byType = TRUE)
```

```
## RW sampler (8)
##   - params[]  (4 elements)
##   - sdy[]  (2 elements)
##   - z_init[]  (2 elements)
```


```r
Rmcmc <- buildMCMC(conf)
Cmodel <- compileNimble(Rmodel)
Cmcmc <- compileNimble(Rmcmc, project = Rmodel)
```

Unleash the beast:

```r
samplesList <- runMCMC(Cmcmc, 10000, 
                       nburnin = 5000,
                       nchains = 2,
                       samplesAsCodaMCMC = TRUE)
```

```
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
## |-------------|-------------|-------------|-------------|
## |-------------------------------------------------------|
```

Check out convergence:

```r
library(coda)
gelman.diag(samplesList)
```

```
## Potential scale reduction factors:
## 
##           Point est. Upper C.I.
## params[1]       1.10       1.36
## params[2]       1.09       1.35
## params[3]       1.11       1.41
## params[4]       1.13       1.46
## sdy[1]          1.02       1.08
## sdy[2]          1.00       1.00
## z_init[1]       1.04       1.15
## z_init[2]       1.02       1.08
## 
## Multivariate psrf
## 
## 1.09
```

Visualize traceplots and posterior distributions:

```r
library(basicMCMCplots)
chainsPlot(samplesList)
```

![](index_files/figure-html/unnamed-chunk-48-1.png)<!-- -->

Summary estimates:

```r
summary(samplesList)
```

```
## 
## Iterations = 1:5000
## Thinning interval = 1 
## Number of chains = 2 
## Sample size per chain = 5000 
## 
## 1. Empirical mean and standard deviation for each variable,
##    plus standard error of the mean:
## 
##             Mean      SD  Naive SE Time-series SE
## params[1] 0.4939 0.04674 0.0004674       0.008640
## params[2] 0.1786 0.01644 0.0001644       0.003004
## params[3] 0.7820 0.07234 0.0007234       0.015589
## params[4] 0.2337 0.02151 0.0002151       0.004243
## sdy[1]    0.2950 0.05415 0.0005415       0.001406
## sdy[2]    0.2456 0.04574 0.0004574       0.001350
## z_init[1] 4.0412 0.12562 0.0012562       0.006846
## z_init[2] 2.1685 0.09235 0.0009235       0.004976
## 
## 2. Quantiles for each variable:
## 
##             2.5%    25%    50%    75%  97.5%
## params[1] 0.4062 0.4600 0.4943 0.5253 0.5870
## params[2] 0.1462 0.1668 0.1791 0.1899 0.2097
## params[3] 0.6605 0.7282 0.7728 0.8349 0.9337
## params[4] 0.1979 0.2168 0.2309 0.2499 0.2778
## sdy[1]    0.2102 0.2571 0.2873 0.3257 0.4194
## sdy[2]    0.1728 0.2142 0.2397 0.2717 0.3485
## z_init[1] 3.8083 3.9525 4.0360 4.1236 4.3078
## z_init[2] 1.9978 2.1069 2.1645 2.2278 2.3601
```

The estimates we get are not exactly the same as Stan's estimates. Not sure why. 

## R version used


```r
sessionInfo()
```

```
## R version 4.2.3 (2023-03-15)
## Platform: aarch64-apple-darwin20 (64-bit)
## Running under: macOS Monterey 12.6
## 
## Matrix products: default
## BLAS:   /Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib/libRblas.0.dylib
## LAPACK: /Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib/libRlapack.dylib
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] lubridate_1.9.2      forcats_1.0.0        stringr_1.5.0       
##  [4] dplyr_1.1.2          purrr_1.0.1          readr_2.1.4         
##  [7] tidyr_1.3.0          tibble_3.2.1         ggplot2_3.4.2       
## [10] tidyverse_2.0.0      basicMCMCplots_0.2.7 coda_0.19-4         
## [13] nimble_0.13.1       
## 
## loaded via a namespace (and not attached):
##  [1] deSolve_1.36     tidyselect_1.2.0 xfun_0.39        bslib_0.5.0     
##  [5] lattice_0.21-8   colorspace_2.1-0 vctrs_0.6.3      generics_0.1.3  
##  [9] htmltools_0.5.5  yaml_2.3.7       utf8_1.2.3       rlang_1.1.1     
## [13] jquerylib_0.1.4  pillar_1.9.0     glue_1.6.2       withr_2.5.0     
## [17] bit64_4.0.5      lifecycle_1.0.3  munsell_0.5.0    gtable_0.3.3    
## [21] codetools_0.2-19 evaluate_0.21    labeling_0.4.2   knitr_1.43      
## [25] tzdb_0.4.0       fastmap_1.1.1    curl_5.0.1       parallel_4.2.3  
## [29] fansi_1.0.4      highr_0.10       scales_1.2.1     cachem_1.0.8    
## [33] vroom_1.6.3      jsonlite_1.8.5   farver_2.1.1     bit_4.0.5       
## [37] hms_1.1.3        digest_0.6.31    stringi_1.7.12   grid_4.2.3      
## [41] cli_3.6.1        tools_4.2.3      magrittr_2.0.3   sass_0.4.6      
## [45] crayon_1.5.2     pkgconfig_2.0.3  timechange_0.2.0 rmarkdown_2.22  
## [49] rstudioapi_0.14  R6_2.5.1         igraph_1.4.2     compiler_4.2.3
```

