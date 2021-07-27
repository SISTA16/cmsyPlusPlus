#THIS CODE TRAINS THREE FEED-FORWARD ARTIFICIAL NEURAL NETWORKS TO ESTIMATE START/INTERMEDIATE/END Bk PRIOR RANGES FROM CATCH STATISTICS

rm(list=ls(all=FALSE)) # clear previous variables etc
options(digits=3) # displays all numbers with three significant digits as default
graphics.off() # close graphics windows from previous sessions

#load library
require(neuralnet)

# functions
scl <- function(x){ (x - min(x))/(max(x) - min(x)) } # scale variables such that lowest value=0 and highest=1

# set hidden layers and neurons
hn            <- c(71) # max = number of predictors, e.g. 7,6; 5,4 #the problem can be solved with one layer, and in this way the network is more stable both on the training set and the k-fold validation
rp            <- 5      # number of repetitions for the training - #20 gives more stability to the network and more independence of the initial training step
thld          <- 0.01   # threshold for minimum decrease in overall error, default 0.01 = 1%
stp           <- 1e+06  # the maximum steps for the training of the neural network, default 1e+05
alg           <- "rprop-"  # possible: backprop, rprop+, rprop-, sag, or slr
act.fct       <- "logistic" # possible: "logistic" or "tanh"; linear.output must be FALSE # logistic works better than tanh with the data at hand
crossvalidate <- T  # Do cross-validation (T/F)? Takes 10 times longer
k             <- 20 # k-fold cross validation
ct_MSY.lim    <- 0.99 # threshold above which B/k variance is assumed same as at MSY

# Load data and set variables names
# use reference stocks as input
Bk_file      <- "Out_Train_ID_9e.csv"
catch_file   <- "Train_Catch_9e.csv"

cdat         <- read.csv(catch_file, header=T, dec=".", stringsAsFactors = FALSE)
Bkdat        <- read.csv(Bk_file, header=T, dec=".", stringsAsFactors = FALSE)

##INPUT PREPARATION
# get properties of catch time series for every stock
ct_MSY.int <- vector() # catch in intermediate year relative to MSY
mean.ct_MSY.end <- vector() # mean catch in last years relative to MSY
mean.ct_MSY.start <- vector() # mean catch in first years relative to MSY
min_max    <- vector() # minimum catch relative to maximum
max.ct.i   <- vector() # relative position of max catch
int.ct.i   <- vector() # relative position of interim year
min.ct.i   <- vector() # relative position of min catch
Bk.end     <- vector() # dependend variable, Bk in last year
Bk.start   <- vector() # dependend variable, Bk in first year
Bk.int     <- vector() # dependend variable, Bk in intermediate year
yr.i       <- vector() # number of years
slope.last <- vector() # slope of catch in last 10 years
slope.first<- vector() # slope of catch in first 10 years
start.i    <- vector() # index of stocks where ct[1]/MSY.pr < 0.7
int.i      <- vector() # index of stocks where ct[1]/MSY.pr < 0.7
end.i      <- vector() # index of stocks where ct[1]/MSY.pr < 0.7

stocks     <- sort(unique(Bkdat$Stock)) # unique list of stocks to process
stocks<-stocks[-which(nchar(stocks)==0)]
Flat <- rep(0, length(stocks))
LH <- rep(0, length(stocks))
LHL <- rep(0, length(stocks))
HL <- rep(0, length(stocks))
HLH <- rep(0, length(stocks))
OTH <- rep(0, length(stocks))


cat("Evaluating altogether",length(stocks),"stocks..\n")
cat("Threshold for inclusion is <",ct_MSY.lim,"MSY \n")
j <- 0 # counter
for(stock in stocks) {
  j=j+1
  # get absolute catch properties
  start.yr        <- as.integer(Bkdat$start.yr[Bkdat$Stock==stock])
  int.yr          <- as.integer(Bkdat$int.yr[Bkdat$Stock==stock])
  end.yr          <- as.integer(Bkdat$end.yr[Bkdat$Stock==stock])
  ct.raw          <- as.numeric(cdat$ct[cdat$Stock==stock & cdat$yr>=start.yr & cdat$yr<=end.yr])
  yr              <- as.integer(cdat$yr[cdat$Stock==stock & cdat$yr>=start.yr & cdat$yr<=end.yr])
  ct              <- ksmooth(x=yr,y=ct.raw,kernel="normal",n.points=length(yr),bandwidth=3)$y
  bt              <- as.numeric(cdat$TB[cdat$Stock==stock & cdat$yr>=start.yr & cdat$yr<=end.yr])
  nyr             <- length(ct) # number of years
  max.ct          <- max(ct,na.rm = T)
  min.ct          <- min(ct,na.rm = T)
  mean.ct.end     <- mean(ct.raw[(nyr-4):nyr])
  mean.ct.start   <- mean(ct.raw[1:5])
  min.yr.i        <- which.min(ct)
  max.yr.i        <- which.max(ct)
  yr.min.ct       <- yr[min.yr.i]
  yr.max.ct       <- yr[max.yr.i]
  min_max.stock   <- min.ct/max.ct
  mean.ct         <- mean(ct)
  sd.ct           <- sd(ct)
  
  # determine prior for MSY
  # if max catch is reached in last 5 years or catch is flat, assume MSY=max catch
  if(max.yr.i>(nyr-4) || ((sd.ct/mean.ct) < 0.1 && min_max.stock > 0.66)) {
    log.msy.pr <- log(max(ct)) } else {
      # else determine proxy for MSY from max smoothed catch based on regression with 104 ICES stocks
      log.msy.pr  <- -0.237 + 0.977 * log(max.ct) } # SD residuals = 0.236, r2=0.989
  MSY.pr       <- exp(log.msy.pr)
  
  # get and store normalized (0-1) catch properties
  ct_MSY.int      <- append(ct_MSY.int,ct[which(yr==int.yr)]/MSY.pr)
  mean.ct_MSY.end <- append(mean.ct_MSY.end,mean.ct.end/MSY.pr)
  mean.ct_MSY.start <- append(mean.ct_MSY.start,mean.ct.start/MSY.pr)
  min_max         <- append(min_max,min_max.stock)
  max.ct.i        <- append(max.ct.i,max.yr.i/nyr)
  int.ct.i        <- append(int.ct.i,which(yr==int.yr)/nyr)
  min.ct.i        <- append(min.ct.i,min.yr.i/nyr)
  yr.i            <- append(yr.i,nyr)
  
#  index of stocks where ct[1]/MSY.pr < ct_MSY.lim 
  if((ct[1]/MSY.pr)          < ct_MSY.lim) start.i <- append(start.i,j) 
  if((ct[yr==int.yr]/MSY.pr) < ct_MSY.lim) int.i   <- append(int.i,j)
  if((ct[nyr]/MSY.pr)        < ct_MSY.lim) end.i   <- append(end.i,j)
  
  # Get slope of catch in last 10 years
  ct.last           <- ct[(nyr-9):nyr]/mean.ct # last catch standardized by mean catch
  yrs.last          <- seq(1:10) 
  fit.last          <- lm(ct.last ~ yrs.last)
  slope.last        <- append(slope.last,as.numeric(coefficients(fit.last)[2]))
  
  # Get slope of catch in first 10 years
  ct.first          <- ct[1:10]/mean.ct # catch standardized by mean catch
  yrs.first         <- seq(1:10) 
  fit.first         <- lm(ct.first ~ yrs.first)
  slope.first       <- append(slope.first,as.numeric(coefficients(fit.first)[2]))
  
  # catch curve pattern
  if(min_max.stock>=0.45 & (ct[1]/max.ct) >= 0.45 & (ct[nyr]/max.ct) >= 0.45) { 
    Flat[j] <- 1 } else Flat[j] <- 0
  if(min_max.stock<0.25 & (ct[1]/max.ct)<0.45 & (ct[nyr]/max.ct)>0.45) { 
    LH[j] <- 1 } else LH[j] <- 0
  if(min_max.stock<0.25 & (ct[1]/max.ct) < 0.45 & (ct[nyr]/max.ct) < 0.25) { 
    LHL[j] <- 1 } else LHL[j] <- 0
  if(min_max.stock<0.25 & (ct[1]/max.ct) > 0.5 & (ct[nyr]/max.ct) < 0.25) { 
    HL[j] <- 1 } else HL[j] <- 0
  if(min_max.stock<0.25 & (ct[1]/max.ct) >= 0.45 & (ct[nyr]/max.ct) >= 0.45) { 
    HLH[j] <- 1 } else HLH[j] <- 0
  if(sum(c(Flat[j],LHL[j],LH[j],HL[j],HLH[j]))<1) { 
    OTH[j] <- 1 } else OTH[j] <- 0
  
  # get true B/k in last year
  Bk.end     <- append(Bk.end,Bkdat$rel_B_BSM[Bkdat$Stock==stock])
  # get true B/k in start year
  Bk.start   <- append(Bk.start,Bkdat$rel_start_B_BSM[Bkdat$Stock==stock])
  # get true B/k in intermediate year
  Bk.int     <- append(Bk.int,Bkdat$rel_int_B_BSM[Bkdat$Stock==stock])
} # end of stocks loop

# normalize number of years and slope
  yr.norm.max<-max(yr.i)
  yr.norm.min<-min(yr.i)
  yr.norm    <- scl(yr.i)
  
  slope.last.max <-max(slope.last)
  slope.last.min <-min(slope.last)
  slope.last.nrm <- scl(slope.last)
  
  slope.first.max <-max(slope.first)
  slope.first.min <-min(slope.first)
  slope.first.nrm <- scl(slope.first)
  
  
##END OF INPUT PREPARATION - COMMON TO START, INTERMEDIATE AND END NEURAL NETWORKS

#prepare input data
train <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH,ct_MSY.int,mean.ct_MSY.end,min_max,
                             max.ct.i,int.ct.i,min.ct.i,yr.norm,slope.last.nrm,slope.first.nrm,mean.ct_MSY.start))

train.start <- train[start.i,]
Bk.start    <- Bk.start[start.i]
train.int   <- train[int.i,]
Bk.int      <- Bk.int[int.i] 
train.end   <- train[end.i,]
Bk.end      <- Bk.end[end.i]

#----------------------------------------------------------
# ANN training function
#----------------------------------------------------------
trainNN<-function(Bk.type,Bk.ref,train.raw,crossvalidation){

  # Bk classes below.Bmsy and above.Bmsy here declared and filled
  nBk        <- length(Bk.ref)
  below.Bmsy <- vector()
  above.Bmsy <- vector()

  for(i in 1:nBk) {
   if(Bk.ref[i] < 0.5) {
     below.Bmsy[i] <- 1
     above.Bmsy[i] <- 0} else {
       below.Bmsy[i] <- 0
       above.Bmsy[i] <- 1 } 
   }
 
#---------------------------------------------------------     
# run NN to check whether B/k is above or below Bmsy
#---------------------------------------------------------  
  cat("Check whether B is below or above Bmsy for",nBk,"training stocks\n")
  targetcolumnset <- c("below.Bmsy","above.Bmsy")

  if(Bk.type=="start") {
    traincolumnset <- c("Flat","LH","LHL","HL","HLH","OTH","min_max","max.ct.i","min.ct.i","yr.norm", # shapes, general   
                        "mean.ct_MSY.start","slope.first.nrm","mean.ct_MSY.end","slope.last.nrm")     # start, end
                          
  } else if(Bk.type=="int") {
    traincolumnset <- c("Flat","LH","LHL","HL","HLH","OTH",      # shapes
                      "min_max","max.ct.i","min.ct.i","yr.norm", # general  
                      "int.ct.i","ct_MSY.int",                   # int
                      "mean.ct_MSY.end","slope.last.nrm",        # end
                      "mean.ct_MSY.start","slope.first.nrm")     # start
  } else {
    traincolumnset <- c("Flat","LH","LHL","HL","HLH","OTH","ct_MSY.int","min_max","max.ct.i",  # arbitrary best sequence
                        "int.ct.i","min.ct.i","yr.norm",
                        "mean.ct_MSY.start","slope.first.nrm","mean.ct_MSY.end","slope.last.nrm")
    }                  
  
  train.1<-cbind(train.raw,below.Bmsy,above.Bmsy)
  train.1<-train.1[c(traincolumnset,targetcolumnset)]
  f <- as.formula(paste(paste(targetcolumnset, collapse = " + "), "~", paste(traincolumnset, collapse = " + ") ))
  set.seed(20) #50000
  # run neuralnet
  nn <- neuralnet(
      f,
      data = train.1,
      hidden = hn, # use e.g 1 or 2 hidden layers with specified neurons
      threshold = thld, # stop if error decreases less than indicated %
      stepmax = stp,
      rep = rp,  # more repetitions reduce variability in mean accuracy
      act.fct = act.fct,  # set the activation function, logistic is default; other option 'tanh' 
      linear.output = FALSE, # activates the activation function if FALSE 
      lifesign = "minimal", # specify the amount of printing during running (full, minimal, none)
      algorithm = alg) # option to use different algorithms
  
  # Compute predictions
  pr.nn1     <- compute(nn, train.1[,1:(which(colnames(train.1)=="below.Bmsy")-1)]) 
  # Extract results
  pr.nn1_    <- pr.nn1$net.result
  # Accuracy (training set)
  original_values <- max.col(train.1[, which(colnames(train.1)=="below.Bmsy"):length(colnames(train.1))]) 
  pr.nn1_2   <- max.col(pr.nn1_)
  accuracy   <- mean(pr.nn1_2 == original_values)
  cat("Accuracy selftest =", accuracy,"\n")
 
  if (crossvalidation==T){
    # Results from cv
    cat("Cross-validating..\n")
    outs <- NULL
    # Train test split proportions
    proportion <- 0.95 # Set to 0.995 for LOOCV
    #Crossvalidation
    for(i in 1:k) {
      cat(i," ")
      index    <- sample(1:nrow(train.1), round(proportion*nrow(train.1)))
      train_cv <- train.1[index, ]
      test_cv  <- train.1[-index, ]
      train_cv <-train_cv[c(traincolumnset,targetcolumnset)]
      test_cv  <-test_cv[c(traincolumnset,targetcolumnset)]
      f        <- as.formula(paste(paste(targetcolumnset, collapse = " + "), "~", paste(traincolumnset, collapse = " + ") ))
      nn_cv    <- neuralnet(f,data = train_cv,hidden = hn,threshold = thld,stepmax = stp,rep = rp,
                         act.fct = act.fct,linear.output = FALSE,algorithm = alg)
      pr.nn3   <- compute(nn_cv, test_cv[, 1:(which(colnames(test_cv)=="below.Bmsy")-1)]) 
      pr.nn3_  <- pr.nn3$net.result
      original_values3 <- max.col(test_cv[, which(colnames(test_cv)=="below.Bmsy"):length(colnames(test_cv))]) 
      pr.nn3_2 <- max.col(pr.nn3_)
      outs[i]  <- mean(pr.nn3_2 == original_values3)
    }
    accuracy3 <- mean(outs)
    cat("\nCrossvalidation accuracy =",accuracy3,"(min",min(outs),", max",max(outs),")\n\n")
  }
  return(nn)
}

cat("Training NN for START Bk range\n")
nn.startbio   <- trainNN("start",Bk.start,train.start,crossvalidate)

cat("Training NN for INTERMEDIATE Bk range\n")
nn.intbio   <- trainNN("int",Bk.int,train.int,crossvalidate)  

cat("Training NN for END Bk range\n")
nn.endbio  <- trainNN("end",Bk.end,train.end,crossvalidate)

save(file = "ffnn.bin",list = c("nn.endbio","nn.startbio","nn.intbio","yr.norm.max","yr.norm.min","slope.last.max","slope.last.min","slope.first.max","slope.first.min"))