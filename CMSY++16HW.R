##---------------------------------------------------------------------------------------------
## CMSY and BSM analysis ----
## Developed by Rainer Froese, Gianpaolo Coro and Henning Winker in 2016, version of January 2021
## PDF creation added by Gordon Tsui and Gianpaolo Coro
## Time series within 1950-2030 are stored in csv file
## Correction for effort creep added by RF
## Multivariate normal r-k priors added to CMSY by HW, RF and GP in October 2019
## Multivariate normal plus observation error on catch added to BSM by HW in November 2019
## Retrospective analysis added by GP in November 2019
## Bayesian implementation of CMSY added by RF and HW in May 2020
## Slight improvements to NA rules for prior B/k done by RF in June 2020
## RF added on-screen proposal to set start.year to medium catch if high or low biomass is unclear at low catch
## Alling notation and posterior compuations between CMSY++ and BSM done by HW in June 2020
## RF fixed a bug where some CMSY instead of BSM results were wrongly reported for management, October 2020
## RF updated cor.log.rk to -0.76 based and MSY.prior based om max.ct, based on a analysis of 240+ global stocks
## HW added use of MSY.prior to predict k.prior in JAGS
## RF and GP reviewed and improved B/k default priors, adding neural network
## HW added beta distribution for B/k priors
## GP added ellipse estimation (lower right focus) of most likely r-k pair for CMSY
##---------------------------------------------------------------------------------------------

# Automatic installation of missing packages
list.of.packages <- c("R2jags","coda","parallel","foreach","doParallel","gplots","mvtnorm","snpar","neuralnet","conicfit")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
library(R2jags)  # Interface with JAGS
library(coda)
library(gplots)
library(mvtnorm)
library(snpar)
library(neuralnet)
library(conicfit)

#-----------------------------------------
# Some general settings ----
#-----------------------------------------
# set.seed(999) # use for comparing results between runs
rm(list=ls(all=FALSE)) # clear previous variables etc
options(digits=3) # displays all numbers with three significant digits as default
graphics.off() # close graphics windows from previous sessions
FullSchaefer <- F    # initialize variable; automatically set to TRUE if enough abundance data are available
n.chains     <- 2 # number of chains to be used in JAGS, default = 2
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set working directory to source file location

#-----------------------------------------
# Required settings, File names ----
#-----------------------------------------
catch_file  <- "Train_Catch_9e.csv" #"Stocks_Catch_2020CMSYrun2_v5_RS - Copy.csv"  #"CombStocks_Catch_2020CMSYrun3_v4.csv"  # "SAUP_Catch_1.csv"  #"SimCatchCPUE_4.csv"  # "Stocks_Catch_Aust_2.csv" #"STECF_Catch_2020_2.csv" #"tRFMO_Catch_2020.csv" #"ICES_Catch_2020.csv" #"Global_Stocks_Catch.csv" #"SimCatchCPUE_4.csv" #"Stocks_Catch_Test.csv"  #"Stock_Catch_forRainer.csv" # "SimCatchCPUE_3.csv" #  name of file containing "Stock", "yr", "ct", and optional "bt"
id_file     <- "Train_ID_9e.csv" #"Stocks_ID_forDeng_RSb - Copy.csv" # "CombStocks_ID_2020CMSYrun3_v3_RF3.csv"   #    "SAUP_ID_2.csv"    #"SimSpecCPUE_4_NA_int_end.csv"  #"Train_ID_7j.csv" # "Stocks_ID_Aust_4.csv"  #"STECF_ID_2020_2.csv" #tRFMO_ID_2020_2.csv" #"ICES_ID_2020_4.csv" #"Robust_Stocks_ID_10_allNA.csv" #"SimSpecCPUE_4.csv"#"Stocks_ID_R_7.csv"  #"Stock_ID_forRainer.csv"  #  "NCod_ID_4.csv" #"SimSpecCPUE_3.csv" #  name of file containing stock-specific info and settings for the analysis
nn_file     <-  "ffnn.bin" # file containing neural networks trained to estimate B/k priors
outfile     <- paste("Out_",format(Sys.Date(),format="%B%d%Y_"),id_file,sep="") # default name for output file

#----------------------------------------
# Select stock to be analyzed ----
#----------------------------------------
stocks      <-NA
# If the input files contain more than one stock, specify below the stock to be analyzed
# If the line below is commented out (#), all stocks in the input file will be analyzed
 stocks <-"ple.27.7d" #"Blacknose shark - Atlantic"  #"Alop_sup_Indian"  #"Acadian redfish - Gulf of Maine / Georges Bank"# "Hogfish - Florida Keys / East Florida"  #"Splitnose rockfish - Pacific Coast"  #"Acadian_redfish"  # "ple.27.7d" #"cod.27.1-2coast"#"cod.27.7e-k" # "Acadian redfish - Gulf of Maine / Georges Bank"  #c("Greenspotted rockfish - Pacific Coast")

#-----------------------------------------
# General settings for the analysis ----
#-----------------------------------------
CV.C         <- 0.15  #><>MSY: Add Catch CV
CV.cpue      <- 0.2 #><>MSY: Add minimum realistic cpue CV
sigmaR       <- 0.1 # overall process error for CMSY; SD=0.1 is the default
sd.log.msy.pr <- 0.3 # rounded upward to account for reduced variability in selected stocks
nbk          <- 3 # Number of B/k priors to be used by BSM, with options 1 (first year), 2 (first & intermediate), 3 (first, intermediate & final bk priors)
bt4pr        <- F # if TRUE, available abundance data are used for B/k prior settings
auto.start   <- F # if TRUE, start year will be set to first year with intermediate catch to avoid ambiguity between low and high bimass if catches are very low
ct_MSY.lim   <- 1.21  # ct/MSY.pr ratio above which B/k prior is assumed constant
q.biomass.pr <- c(0.9,1.1) # if btype=="biomass" this is the prior range for q
n            <- 5000 # number of points in multivariate cloud in graph panel (b)
ni           <- 3 # iterations for r-k-startbiomass combinations, to test different variability patterns; no improvement seen above 3
nab          <- 3 # recommended=5; minimum number of years with abundance data to run BSM
bw           <- 3 # default bandwidth to be used by ksmooth() for catch data
mgraphs      <- F # set to TRUE to produce additional graphs for management
e.creep.line <- T # set to TRUE to display uncorrected CPUE in biomass graph
kobe.plot    <- F # set to TRUE to produce additional kobe status plot; management graph needs to be TRUE for Kobe to work
BSMfits.plot <- F # set to TRUE to plot fit diagnostics for BSM
pp.plot      <- F # set to TRUE to plot Posterior and Prior distributions for CMSY and BSM
retros       <- F # set to TRUE to enable retrospective analysis (1-3 years less in the time series)
save.plots   <- T # set to TRUE to save graphs to JPEG files
close.plots  <- T # set to TRUE to close on-screen plots after they are saved, to avoid "too many open devices" error in batch-processing
write.output <- T # set to TRUE if table with results in output file is wanted; expects years 2004-2014 to be available
write.pdf    <- F # set to TRUE if PDF output of results is wanted. See more instructions at end of code.
select.yr    <- NA # option to display F, B, F/Fmsy and B/Bmsy for a certain year; default NA
write.rdata  <- F #><>HW write R data file

#----------------------------------------------
#  FUNCTIONS ----
#----------------------------------------------


#-------------------------------------------------------------
# Function to run Bayesian Schaefer Model (BSM)
#-------------------------------------------------------------
bsm   <- function(ct,btj,nyr,prior.r,prior.k,startbio,q.priorj,
                  init.q,init.r,init.k,pen.bk,pen.F,b.yrs,b.prior,CV.C,CV.cpue,nbk,rk.cor,cmsyjags) {
  #><> convert b.prior ranges into beta priors
  bk.beta = beta.prior(b.prior)

  if(cmsyjags==TRUE ){ nbks=3 } else {nbks = nbk} # Switch between CMSY + BSM

  # Data to be passed on to JAGS
  jags.data        <- c('ct','btj','nyr', 'prior.r', 'prior.k', 'startbio', 'q.priorj',
                        'init.q','init.r','init.k','pen.bk','pen.F','b.yrs','bk.beta','CV.C','CV.cpue','nbks','rk.cor')
  # Parameters to be returned by JAGS #><> HW add key quantaties
  jags.save.params <- c('r','k','q', 'P','ct.jags','cpuem','proc.logB','B','F','BBmsy','FFmsy','ppd.logrk')

  # JAGS model ----
  Model = "model{
    # to reduce chance of non-convergence, Pmean[t] values are forced >= eps
    eps<-0.01
    #><> Add Catch.CV
    for(t in 1:nyr){
      ct.jags[t] ~ dlnorm(log(ct[t]),pow(CV.C,-2))
    }

    penm[1]  <- 0 # no penalty for first biomass
    Pmean[1] <- log(alpha)
    P[1]     ~ dlnorm(Pmean[1],itau2)

    for (t in 2:nyr) {
      Pmean[t] <- ifelse(P[t-1] > 0.25,
        log(max(P[t-1] + r*P[t-1]*(1-P[t-1]) - ct.jags[t-1]/k,eps)),  # Process equation
        log(max(P[t-1] + 4*P[t-1]*r*P[t-1]*(1-P[t-1]) - ct.jags[t-1]/k,eps))) # linear decline of r at B/k < 0.25
      P[t]     ~ dlnorm(Pmean[t],itau2) # Introduce process error
      penm[t]  <- ifelse(P[t]<(eps+0.001),log(q*k*P[t])-log(q*k*(eps+0.001)),
                   # ifelse(P[t]>1,ifelse((ct[t]/max(ct))>0.2,log(q*k*P[t])-log(q*k*(0.99)),0),0)) # penalty if Pmean is outside viable biomass
                    ifelse(P[t]>1.1,log(q*k*P[t])-log(q*k*(0.99)),0))
    }

    # Get Process error deviation
    for(t in 1:nyr){
      proc.logB[t] <- log(P[t]*k)-log(exp(Pmean[t])*k)}

    # ><> b.priors with penalties
    # Biomass priors/penalties are enforced as follows
    for(i in 1:nbks){
    bk.mu[i] ~ dbeta(bk.beta[1,i],bk.beta[2,i])
    bk.beta[3,i] ~ dnorm(bk.mu[i]-P[b.yrs[i]],10000)
    }

    for (t in 1:nyr){
      Fpen[t]   <- ifelse(ct[t]>(0.9*k*P[t]),ct[t]-(0.9*k*P[t]),0) # Penalty term on F > 1, i.e. ct>B
      pen.F[t]  ~ dnorm(Fpen[t],1000)
      pen.bk[t] ~ dnorm(penm[t],10000)
      cpuem[t]  <- log(q*P[t]*k);
      btj[t]     ~ dlnorm(cpuem[t],pow(sigma2,-1));
    }

  # priors
  log.alpha               <- log((startbio[1]+startbio[2])/2) # needed for fit of first biomass
  sd.log.alpha            <- (log.alpha-log(startbio[1]))/4
  tau.log.alpha           <- pow(sd.log.alpha,-2)
  alpha                   ~  dlnorm(log.alpha,tau.log.alpha)

  # set realistic prior for q
  log.qm              <- mean(log(q.priorj))
  sd.log.q            <- (log.qm-log(q.priorj[1]))/2
  tau.log.q           <- pow(sd.log.q,-2)
  q                   ~  dlnorm(log.qm,tau.log.q)

  # define process (tau) and observation (sigma) variances as inversegamma priors
  itau2 ~ dgamma(4,0.01)
  tau2  <- 1/itau2
  tau   <- pow(tau2,0.5)

  isigma2 ~ dgamma(2,0.01)
  sigma2 <- 1/isigma2+pow(CV.cpue,2) # Add minimum realistic CPUE CV
  sigma  <- pow(sigma2,0.5)

  log.rm              <- mean(log(prior.r))
  sd.log.r            <- abs(log.rm - log(prior.r[1]))/2
  tau.log.r           <- pow(sd.log.r,-2)

  # bias-correct lognormal for k
  log.km              <- mean(log(prior.k))
  sd.log.k            <- abs(log.km-log(prior.k[1]))/2
  tau.log.k           <- pow(sd.log.k,-2)

  # Construct Multivariate lognormal (MVLN) prior
  mu.rk[1] <- log.rm
  mu.rk[2] <- log.km

  # Prior for correlation log(r) vs log(k)
  #><>MSY: now directly taken from mvn of ki = 4*msyi/ri
  rho <- rk.cor

  # Construct Covariance matrix
  cov.rk[1,1] <- sd.log.r * sd.log.r
  cov.rk[1,2] <- rho
  cov.rk[2,1] <- rho
  cov.rk[2,2] <- sd.log.k * sd.log.k

  # MVLN prior for r-k
  log.rk[1:2] ~ dmnorm(mu.rk[],inverse(cov.rk[,]))
  r <- exp(log.rk[1])
  k <- exp(log.rk[2])

  #><>MSY get posterior predictive distribution for rk
  ppd.logrk[1:2] ~ dmnorm(mu.rk[],inverse(cov.rk[,]))

  # ><>HW: Get B/Bmsy and F/Fmsy directly from JAGS
  Bmsy <- k/2
  Fmsy <- r/2
  for (t in 1:nyr){
  B[t] <- P[t]*k # biomass
  F[t] <- ct.jags[t]/B[t]
  BBmsy[t] <- P[t]*2 #true for Schaefer
  FFmsy[t] <- ifelse(BBmsy[t]<0.5,F[t]/(Fmsy*2*BBmsy[t]),F[t]/Fmsy)
  }
} "    # end of JAGS model

  # Write JAGS model to file ----
  cat(Model, file="r2jags.bug")

  #><>MSY: change to lognormal inits (better)
  j.inits <- function(){list("log.rk"=c(rnorm(1,mean=log(init.r),sd=0.2),rnorm(1,mean=log(init.k),sd=0.1)),
                             "q"=rlnorm(1,mean=log(init.q),sd=0.2),"itau2"=1000,"isigma2"=1000)}
  # run model ----
  jags_outputs <- jags.parallel(data=jags.data,
                                working.directory=NULL, inits=j.inits,
                                parameters.to.save=jags.save.params,
                                model.file="r2jags.bug", n.chains = n.chains,
                                n.burnin = 30000, n.thin = 10,
                                n.iter = 60000)
  return(jags_outputs)
}

#><> beta.prior function
get_beta <- function(mu,CV,Min=0,Prior="x",Plot=FALSE){
  a = seq(0.0001,1000,0.001)
  b= (a-mu*a)/mu
  s2 = a*b/((a+b)^2*(a+b+1))
  sdev = sqrt(s2)
  # find beta parameter a
  CV.check = (sdev/mu-CV)^2
  a = a[CV.check==min(CV.check)]
  # find beta parameter b
  b = (a-mu*a)/mu
  x = seq(Min,1,0.001)
  pdf = dbeta(x,a,b)
  if(Plot==TRUE){
    plot(x,pdf,type="l",xlim=range(x[pdf>0.01]),xlab=paste(Prior),ylab="",yaxt="n")
    polygon(c(x,rev(x)),c(rep(0,length(x)),rev(ifelse(pdf==Inf,100000,pdf))),col="grey")
  }
  return(c(a,b))
}

#><> convert b.prior ranges into beta priors
beta.prior = function(b.prior){
  bk.beta = matrix(0,nrow = 3,ncol=3)
  for(i in 1:3){
    sd.bk = (b.prior[2,i]-b.prior[1,i])/(4*0.98)
    mu.bk = mean(b.prior[1:2,i])
    cv.bk = sd.bk/mu.bk
    bk.beta[1:2,i] = get_beta(mu.bk,cv.bk)
  }
  return(bk.beta)
}

#Fits an ellipse around the CMSY r-k cloud and estimates the rightmost focus
traceEllipse<-function(rs,ks,prior.r,prior.k){
  log.rs<-log(rs)
  log.ks<-log(ks)

#  #select data within the bounding box
#  log.rs<-log.rs[which(rs>prior.r[1] & rs<prior.r[2] &
#                         ks>prior.k[1] & ks<prior.k[2]
#  )]
#  log.ks<-log.ks[which(rs>prior.r[1] & rs<prior.r[2] &
#                         ks>prior.k[1] & ks<prior.k[2]
#  )]

  #prepare data for ellipse fitting
  cloud.data <- as.matrix(data.frame(x = log.rs, y = log.ks))
  ellip <- EllipseDirectFit(cloud.data)
  #estimate ellipse characteristics
  atog<-AtoG(ellip)
  ellipG <- atog$ParG
  ell.center.x<-ellipG[1]
  ell.center.y<-ellipG[2]
  ell.axis.a<-ellipG[3]
  ell.axis.b<-ellipG[4]
  ell.tilt.angle.deg<-180/pi*ellipG[5]
  ell.slope<-tan(ellipG[5])
  xy.ell<-calculateEllipse(ell.center.x,
                           ell.center.y,
                           ell.axis.a,
                           ell.axis.b,
                           ell.tilt.angle.deg)
  #draw ellipse
  #points(x=xy.ell[,1],y=xy.ell[,2],col='red',type='l')
  ell.intercept.1 = ell.center.y-ell.center.x*ell.slope
  #draw ellipse main axis
  #abline(a =ell.intercept.1, b=ell.slope,col='red')
  #calculate focus from demi-axes
  ell.demiaxis.c.sqr<-(0.25*ell.axis.a*ell.axis.a)-(0.25*ell.axis.b*ell.axis.b)
  if (ell.demiaxis.c.sqr<0)
    ell.demiaxis.c.sqr<-ell.axis.a/2
  else
    ell.demiaxis.c<-sqrt(ell.demiaxis.c.sqr)
  sin.c<-ell.demiaxis.c*sin(ellipG[5])
  cos.c<-ell.demiaxis.c*cos(ellipG[5])
  ell.foc.y<-ell.center.y-sin.c
  ell.foc.x<-ell.center.x-cos.c
  #draw focus
  #points(x=ell.foc.x,y=ell.foc.y,
  #      pch = 16, cex = 1.2,
  #     col='green',bty='l')

  return (c(exp(ell.foc.x),exp(ell.foc.y)))
}
#---------------------------------------------
# END OF FUNCTIONS
#---------------------------------------------

#-----------------------------------------
# Start output to screen
#-----------------------------------------
cat("-------------------------------------------\n")
cat("CMSY++ Analysis,", date(),"\n")
cat("-------------------------------------------\n")

#------------------------------------------
# Read data and assign to vectors
#------------------------------------------
# create headers for data table file
if(write.output==T){
  outheaders = data.frame("Group","Region", "Subregion","Name","SciName","Stock",
                          "start.yr","end.yr","start.yr.new","btype",
                          "N bt","start.yr.cpue","end.yr.cpue","min.cpue","max.cpue","min.yr.cpue","max.yr.cpue",
                          "endbio.low","endbio.hi","q.prior.low","q.prior.hi",
                          "MaxCatch","MSY_prior","MeanLast5RawCatch","SDLast5RawCatch","LastCatch",
                          "MinSmoothCatch","MaxSmoothCatch","MeanSmoothCatch","gMeanPrior_r",
                          "MSY_BSM","lcl.MSY_BSM","ucl.MSY_BSM","r_BSM","lcl.r_BSM","ucl.r_BSM","log.r_var",
                          "k_BSM","lcl.k_BSM","ucl.k_BSM","log.k_var","log.kr_cor","log.kr_cov","q_BSM","lcl.q_BSM","ucl.q_BSM",
                          "rel_B_BSM","lcl.rel_B_BSM","ucl.rel_B_BSM","rel_start_B_BSM","lcl.rel_start_B_BSM","ucl.rel_start_B_BSM",
                          "rel_int_B_BSM","lcl.rel_int_B_BSM","ucl.rel_int_B_BSM","int.yr","rel_F_BSM",
                          "r_CMSY","lcl.r_CMSY","ucl.r_CMSY","k_CMSY","lcl.k_CMSY","ucl.k_CMSY","MSY_CMSY","lcl.MSY_CMSY","ucl.MSY_CMSY",
                          "rel_B_CMSY","2.5th.rel_B_CMSY","97.5th.rel_B_CMSY","rel_start_B_CMSY","2.5th.rel_start_B_CMSY","97.5th.rel_start_B_CMSY",
                          "rel_int_B_CMSY","2.5th.rel_int_B_CMSY","97.5th.rel_int_B_CMSY",
                          "rel_F_CMSY","2.5th.rel_F_CMSY","97.5th.rel_F_CMSY",
                          "F_msy","lcl.F_msy","ucl.F_msy","curF_msy","lcl.curF_msy","ucl.curF_msy",
                          "MSY","lcl.MSY","ucl.MSY","Bmsy","lcl.Bmsy","ucl.Bmsy",
                          "last.B","lcl.last.B","ucl.last.B","last.B_Bmsy","lcl.last.B_Bmsy","ucl.last.B_Bmsy",
                          "last.F","lcl.last.F","ucl.last.F","last.F_Fmsy","lcl.last.F_Fmsy","ucl.last.F_Fmsy",
                          "sel_B","sel_B_Bmsy","sel_F","sel_F_Fmsy",
                          # create columns for catch, F/Fmsy and Biomass for 1950 to 2020
                          "c50","c51","c52","c53","c54","c55","c56","c57","c58","c59",
                          "c60","c61","c62","c63","c64","c65","c66","c67","c68","c69",
                          "c70","c71","c72","c73","c74","c75","c76","c77","c78","c79",
                          "c80","c81","c82","c83","c84","c85","c86","c87","c88","c89",
                          "c90","c91","c92","c93","c94","c95","c96","c97","c98","c99",
                          "c00","c01","c02","c03","c04","c05","c06","c07","c08","c09",
                          "c10","c11","c12","c13","c14","c15","c16","c17","c18","c19",
                          "c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30",
                          "F.Fmsy50","F.Fmsy51","F.Fmsy52","F.Fmsy53","F.Fmsy54","F.Fmsy55","F.Fmsy56","F.Fmsy57","F.Fmsy58","F.Fmsy59",
                          "F.Fmsy60","F.Fmsy61","F.Fmsy62","F.Fmsy63","F.Fmsy64","F.Fmsy65","F.Fmsy66","F.Fmsy67","F.Fmsy68","F.Fmsy69",
                          "F.Fmsy70","F.Fmsy71","F.Fmsy72","F.Fmsy73","F.Fmsy74","F.Fmsy75","F.Fmsy76","F.Fmsy77","F.Fmsy78","F.Fmsy79",
                          "F.Fmsy80","F.Fmsy81","F.Fmsy82","F.Fmsy83","F.Fmsy84","F.Fmsy85","F.Fmsy86","F.Fmsy87","F.Fmsy88","F.Fmsy89",
                          "F.Fmsy90","F.Fmsy91","F.Fmsy92","F.Fmsy93","F.Fmsy94","F.Fmsy95","F.Fmsy96","F.Fmsy97","F.Fmsy98","F.Fmsy99",
                          "F.Fmsy00","F.Fmsy01","F.Fmsy02","F.Fmsy03","F.Fmsy04","F.Fmsy05","F.Fmsy06","F.Fmsy07","F.Fmsy08","F.Fmsy09",
                          "F.Fmsy10","F.Fmsy11","F.Fmsy12","F.Fmsy13","F.Fmsy14","F.Fmsy15","F.Fmsy16","F.Fmsy17","F.Fmsy18","F.Fmsy19",
                          "F.Fmsy20","F.Fmsy21","F.Fmsy22","F.Fmsy23","F.Fmsy24","F.Fmsy25","F.Fmsy26","F.Fmsy27","F.Fmsy28","F.Fmsy29","F.Fmsy30",
                          "B50","B51","B52","B53","B54","B55","B56","B57","B58","B59",
                          "B60","B61","B62","B63","B64","B65","B66","B67","B68","B69",
                          "B70","B71","B72","B73","B74","B75","B76","B77","B78","B79",
                          "B80","B81","B82","B83","B84","B85","B86","B87","B88","B89",
                          "B90","B91","B92","B93","B94","B95","B96","B97","B98","B99",
                          "B00","B01","B02","B03","B04","B05","B06","B07","B08","B09",
                          "B10","B11","B12","B13","B14","B15","B16","B17","B18","B19",
                          "B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30")
  write.table(outheaders,file=outfile, append = T, sep=",",row.names=F,col.names=F)
}

# Read data
cdat         <- read.csv(catch_file, header=T, dec=".", stringsAsFactors = FALSE)
cinfo        <- read.csv(id_file, header=T, dec=".", stringsAsFactors = FALSE)
load(file = nn_file) # load neural network file
cat("Files", catch_file, ",", id_file,",",nn_file,"read successfully","\n")

#---------------------------------
# Analyze stock(s)
#---------------------------------
if(is.na(stocks[1])==TRUE){
  # stocks         <- as.character(cinfo$Stock) # Analyze stocks in sequence of ID file
  # stocks         <- sort(as.character(cinfo$Stock[cinfo$Stock>="Cras_vir_Virginian"])) # Analyze in alphabetic order after a certain stock
   stocks         <- sort(as.character(cinfo$Stock)) # Analyze stocks in alphabetic order
  # stocks         <- as.character(cinfo$Stock[cinfo$btype!="None" & cinfo$Stock>"Squa_aca_BlackSea"]) # Analyze stocks by criteria in ID file
}

# analyze one stock after the other...
for(stock in stocks) {

  cat("Processing",stock,",", as.character(cinfo$ScientificName[cinfo$Stock==stock]),"\n")

  #retrospective analysis
  retros.nyears<-ifelse(retros==T,3,0) #retrospective analysis
  FFmsy.retrospective<-list() #retrospective analysis
  BBmsy.retrospective<-list() #retrospective analysis
  years.retrospective<-list() #retrospective analysis

  retrosp.step =0
  for (retrosp.step in 0:retros.nyears){ #retrospective analysis loop

  # Declare conditional Objects that feature with ifelse clauses
    B.sel        <- NULL
    B.Bmsy.sel   <- NULL
    F.sel        <- NULL
    F.Fmsy.sel   <- NULL
    true.MSY     <- NULL
    true.r       <- NULL
    true.k       <- NULL
    true.Bk      <- NULL
    true.F_Fmsy  <- NULL
    true.q       <- NULL

  # assign data from cinfo to vectors
    btype        <- as.character(cinfo$btype[cinfo$Stock==stock])
    res          <- as.character(cinfo$Resilience[cinfo$Stock==stock])
    start.yr     <- as.numeric(cinfo$StartYear[cinfo$Stock==stock])
    end.yr       <- as.numeric(cinfo$EndYear[cinfo$Stock==stock])
    end.yr.orig  <- end.yr
    end.yr 	     <- end.yr-retrosp.step #retrospective analysis
    yr           <- as.numeric(cdat$yr[cdat$Stock==stock & cdat$yr >= start.yr & cdat$yr <= end.yr])
    if(length(yr)==0){
      cat("ERROR: Could not find the stock in the Catch file -
      check that the stock names match in ID and Catch files and that commas are used (not semi-colon)")
      return (NA) }

    # code to change start year to avoid ambiguity in biomass prior -----------------------------------------
    ct.raw       <- as.numeric(cdat$ct[cdat$Stock==stock & cdat$yr >= start.yr & cdat$yr <= end.yr])/1000  ## assumes that catch is given in tonnes, transforms to '000 tonnes
    ct           <- ksmooth(x=yr,y=ct.raw,kernel="normal",n.points=length(yr),bandwidth=bw)$y
    ct.3         <- mean(ct[1:3])
    max.ct       <- max(ct)

    if(btype=="biomass" | btype=="CPUE" ) {
      bt.raw1 <- as.numeric(cdat$bt[cdat$Stock==stock & cdat$yr >= start.yr & cdat$yr <= end.yr])
      # if bt.raw is zero, change to NA
      bt.raw1[bt.raw1==0] <- NA
      if(btype=="biomass") { # make sure both catch and biomass are divided by 1000
        bt <- bt.raw1/1000 } else { # get number of integer digits for bt.raw (because sometimes they give numbers of eggs!)
          bt.digits <- floor(log10(mean(bt.raw1,na.rm=T)))+1
          if(bt.digits>3) {bt.raw <- bt.raw1/10^(bt.digits-1)} else {bt.raw <- bt.raw1}
          bt     <- bt.raw #ksmooth(x=yr,y=bt.raw,kernel="normal",n.points=length(yr),bandwidth=3)$y
        } # end of bt==CPUE loop
      if(length(bt[is.na(bt)==F])==0) {
        cat("ERROR: No CPUE or biomass data in the Catch input file")
        return (NA) }
    } else {bt <- NA; bt.raw <- NA} # if there is no biomass or CPUE, set bt to NA


    # code to change start year to avoid ambiguity in biomass prior -----------------------------------------
    start.yr.new <- NA # initialize / reset start.yr.new with NA
    if(is.na(cinfo$stb.low[cinfo$Stock==stock]) & ct.3 < (0.33*max.ct) & start.yr < 2000 & (btype=="None" || yr[is.na(bt)==F][1]>yr[3])) { # it is unlikely that a fishery started on an unexploited stock after 2000
      start.yr.new <- yr[which(ct >= (0.4*max.ct))][1]
      cat("\n          *****************************************************************************************
          Attention: Low catch in",start.yr,"may indicate either depleted or unexploited biomass.
          Set startbio in ID file to 0.01-0.2 or 0.8-1.0 to indicate depleted or unexploited biomass.\n")
      if(auto.start==T) { # change start year automatically if auto.start is TRUE
        start.yr  <- start.yr.new
        cat("          Meanwhile start year was set to",start.yr,"to avoid ambiguity.\n")
      } else {
        cat("          Else, set start year in ID file to",start.yr.new,"to avoid uncertainty\n")
      }
      cat("          ******************************************************************************************\n\n") }
    # end of code for start biomass prior ambiguity

    ename        <- cinfo$Name[cinfo$Stock==stock]
    r.low        <- as.numeric(cinfo$r.low[cinfo$Stock==stock])
    r.hi         <- as.numeric(cinfo$r.hi[cinfo$Stock==stock])
    stb.low      <- as.numeric(cinfo$stb.low[cinfo$Stock==stock])
    stb.hi       <- as.numeric(cinfo$stb.hi[cinfo$Stock==stock])
    int.yr       <- as.numeric(cinfo$int.yr[cinfo$Stock==stock])
    intb.low     <- as.numeric(cinfo$intb.low[cinfo$Stock==stock])
    intb.hi      <- as.numeric(cinfo$intb.hi[cinfo$Stock==stock])
    endb.low     <- as.numeric(cinfo$endb.low[cinfo$Stock==stock])
    endb.hi      <- as.numeric(cinfo$endb.hi[cinfo$Stock==stock])
    e.creep      <- as.numeric(cinfo$e.creep[cinfo$Stock==stock])
    force.cmsy   <- cinfo$force.cmsy[cinfo$Stock==stock]
    comment      <- as.character(cinfo$Comment[cinfo$Stock==stock])
    source       <- as.character(cinfo$Source[cinfo$Stock==stock])
    # set global defaults for uncertainty
    sigR         <- sigmaR
    # for simulated data only
    if(substr(id_file,1,3)=="Sim") {
      true.MSY     <- cinfo$true.MSY[cinfo$Stock==stock]/1000
      true.r       <- cinfo$true.r[cinfo$Stock==stock]
      true.k       <- cinfo$true.k[cinfo$Stock==stock]/1000
      true.Bk      <- (cinfo$last.TB[cinfo$Stock==stock]/1000)/true.k
      true.F_Fmsy  <- cinfo$last.F_Fmsy[cinfo$Stock==stock]
      true.q       <- cinfo$last.cpue[cinfo$Stock==stock]/cinfo$last.TB[cinfo$Stock==stock]
    }
    # do retrospective analysis
    if (retros==T && retrosp.step==0){
      cat("* ",ifelse(btype!="None","BSM","CMSY")," retrospective analysis for ",
          stock," has been enabled\n",sep="") #retrospective analysis
    }
    if (retros==T){
      cat("* Retrospective analysis: step n. ",(retrosp.step+1),"/",(retros.nyears+1),
          ". Range of years: [",start.yr ," - ",end.yr,"]\n",sep="") #retrospective analysis
    }

    # -------------------------------------------------------------
    # check for common errors
    #--------------------------------------------------------------
    if(length(btype)==0){
      cat("ERROR: Could not find the stock in the ID input file - check that the stock names match in ID and Catch files and that commas are used (not semi-colon)")
      return (NA) }
    if(start.yr < cdat$yr[cdat$Stock==stock][1]){
      cat("ERROR: start year in ID file before first year in catch file\n")
      return (NA)
      break}
    if(length(yr)==0){
      cat("ERROR: Could not find the stock in the Catch input files - Please check that the code is written correctly")
      return (NA) }
    if(btype %in% c("None","CPUE","biomass")==FALSE){
      cat("ERROR: In ID file, btype must be None, CPUE, or biomass.")
      return (NA) }
    if(retros==F & length(yr) != (end.yr-start.yr+1)) {
      cat("ERROR: indicated year range is of different length than years in catch file\n")
      return (NA)}
    if(length(ct.raw[ct.raw>0])==0) {
      cat("ERROR: No catch data in the Catch input file")
      #return (NA)
      next }
    if(is.na(int.yr)==F & (int.yr < start.yr | int.yr > end.yr)) {
      cat("ERROR: year for intermediate B/k prior outside range of years")
      return (NA)}
    if(is.na(int.yr)==T & (is.na(intb.low)==F | is.na(intb.hi)==F)) {
        cat("ERROR: intermediate B/k prior given without year")
        return (NA)}

    # apply correction for effort-creep to commercial(!) CPUE
    if(btype=="CPUE" && is.na(e.creep)==FALSE) {
      cpue.first  <- min(which(is.na(bt)==F))
      cpue.last   <- max(which(is.na(bt)==F))
      cpue.length <- cpue.last - cpue.first
      bt.cor      <- bt
      for(i in 1:(cpue.length)) {
        bt.cor[cpue.first+i]  <- bt[cpue.first+i]*(1-e.creep/100)^i # equation for decay in %
      }
      bt <- bt.cor
    }

    if(retros==T && force.cmsy == F && (btype !="None" & length(bt[is.na(bt)==F])<nab) ) { #stop retrospective analysis if cpue is < nab
      cat("Warning: Cannot run retrospective analysis for ",end.yr,", number of remaining ",btype," values is too low (<",nab,")\n",sep="")
      #retrosp.step<-retros.nyears
      break }

    if(is.na(mean(ct.raw))){
      cat("ERROR: Missing value in Catch data; fill or interpolate\n")
    }
    nyr          <- length(yr) # number of years in the time series


    # initialize vectors for viable r, k, bt, and all in a matrix
    mdat.all    <- matrix(data=vector(),ncol=2+nyr+1)

    # initialize other vectors anew for each stock
    current.attempts <- NA

    # use start.yr if larger than select year
    if(is.na(select.yr)==F) {
      sel.yr <- ifelse(start.yr > select.yr,start.yr,select.yr)
    } else sel.yr <- NA

    #----------------------------------------------------
    # Determine initial ranges for parameters and biomass
    #----------------------------------------------------
    if(!(res %in% c("High","Medium","Low","Very low"))) {
      cat("ERROR: Resilience not High, Medium, Low, or Very low in ID input file")
      return (NA)} else {
    # initial range of r from input file
    if(is.na(r.low)==F & is.na(r.hi)==F) {
      prior.r <- c(r.low,r.hi)
    } else
      # initial range of r based on resilience
      if(res == "High") {
        prior.r <- c(0.6,1.5)} else if(res == "Medium") {
          prior.r <- c(0.2,0.8)}    else if(res == "Low") {
            prior.r <- c(0.05,0.5)}  else { # i.e. res== "Very low"
              prior.r <- c(0.015,0.1)}
    }
    gm.prior.r      <- exp(mean(log(prior.r))) # get geometric mean of prior r range

    #-----------------------------------------
    # determine MSY prior
    #-----------------------------------------
    # get index of years with lowest and highest catch
    min.yr.i     <- which.min(ct)
    max.yr.i     <- which.max(ct)
    yr.min.ct    <- yr[min.yr.i]
    yr.max.ct    <- yr[max.yr.i]
    min.ct       <- ct[min.yr.i]
    max.ct       <- ct[max.yr.i]
    min_max      <- min.ct/max.ct
    mean.ct      <- mean(ct)
    sd.ct        <- sd(ct)

    ct.sort     <- sort(ct.raw)
    # if max catch is reached in last 5 years or catch is flat, assume MSY=max catch
    if(max.yr.i>(nyr-4) || ((sd.ct/mean.ct) < 0.1 && min_max > 0.66)) {
        MSY.pr <- mean(ct.sort[(nyr-2):nyr]) } else {
          MSY.pr <- 0.75*mean(ct.sort[(nyr-4):nyr]) } # else, use fraction of mean of 5 highest catches as MSY prior

    #><>MSY: MSY prior
    log.msy.pr    <- log(MSY.pr)
    prior.msy     <- c(exp(log.msy.pr-1.96*sd.log.msy.pr),exp(log.msy.pr+1.96*sd.log.msy.pr))


    #----------------------------------------------------------------
    # Multivariate normal sampling of r-k log space
    #----------------------------------------------------------------
    # turn numerical ranges into log-normal distributions
    mean.log.r=mean(log(prior.r))
    sd.log.r=(log(prior.r[2])-log(prior.r[1]))/(2*1.96)  # assume range covers 4 SD

    #><>MSY: new k = r-msy space
    # generate msy and r independently
    ri1     <- rlnorm(n,mean.log.r,sd.log.r)
    msyi1  <- rlnorm(n,log.msy.pr,sd.log.msy.pr)
    ki1     <- msyi1*4/ri1
    
    #><>MSY: get log median and covariance
    cov_rk <- cov(cbind(log(ri1),log(ki1)))
    mu_rk <-  apply(cbind(log(ri1),log(ki1)),2,median)
    rk.cor <- cov_rk[2,1] #MSY: correlation rho input to JAGS
    #><>MSY: mvn prior for k = 4*msy/r
    mvn.log.rk <- rmvnorm(n,mean=mu_rk,cov_rk)

    ri2    <- exp(mvn.log.rk[,1])
    ki2    <- exp(mvn.log.rk[,2])

    mean.log.k <- median(log(ki1))
    sd.log.k.pr <- sd(log(ki1))
    # quick check must be the same
    sd.log.k = sqrt(cov_rk[2,2])
    sd.log.k.pr
    sd.log.k
    #><>MSY: k.prior - Transform to range 
    prior.k     <- exp(mean.log.k-1.96*sd.log.k.pr) # declare variable and set prior.k[1] in one step
    prior.k[2]  <- exp(mean.log.k+1.96*sd.log.k.pr)
  

    #-----------------------------------------
    # determine prior B/k ranges
    #-------------------------------------------------
    # determine intermediate year int.yr for prior B/k
    if(is.na(cinfo$int.yr[cinfo$Stock==stock])==F) {
      int.yr <- cinfo$int.yr[cinfo$Stock==stock]     # use int.yr give by user
    } else {if(min_max > 0.7) { # if catch is about flat, use middle year as int.yr
      int.yr    <- as.integer(mean(c(start.yr, end.yr)))
      } else { # only consider catch 5 years away from end points and within last 30 years # 50
      yrs.int       <- yr[yr>(yr[nyr]-30) & yr>yr[4] & yr<yr[nyr-4]]
      ct.int        <- ct[yr>(yr[nyr]-30) & yr>yr[4] & yr<yr[nyr-4]]
      min.ct.int    <- min(ct.int)
      min.ct.int.yr <- yrs.int[which.min(ct.int)]
      max.ct.int    <- max(ct.int)
      max.ct.int.yr <- yrs.int[which.max(ct.int)]
      #if min year is after max year, use min year for int year
      if(min.ct.int.yr > max.ct.int.yr) { int.yr <- min.ct.int.yr } else {
        # if min.ct/max.ct after max.ct < 0.7, use that year for int.yr
        min.ct.after.max <- min(ct.int[yrs.int >= max.ct.int.yr])
        if((min.ct.after.max/max.ct.int) < 0.75) {
          int.yr <- yrs.int[yrs.int > max.ct.int.yr & ct.int==min.ct.after.max]
        } else {int.yr <- min.ct.int.yr}
      }
      # get latest year where ct < 1.2 min ct
      # int.yr        <- max(yrs.int[ct.int<=(1.2*min.ct.int)])
     }
    }# end of int.yr loop

    # get additional properties of catch time series
    mean.ct.end       <- mean(ct.raw[(nyr-4):nyr]) # mean of catch in last 5 years
    mean.ct_MSY.end   <- mean.ct.end/MSY.pr
    # Get slope of catch in last 10 years
    ct.last           <- ct[(nyr-9):nyr]/mean(ct) # last catch standardized by mean catch
    yrs.last          <- seq(1:10)
    fit.last          <- lm(ct.last ~ yrs.last)
    slope.last        <- as.numeric(coefficients(fit.last)[2])
    slope.last.nrm    <- (slope.last - slope.last.min)/(slope.last.max - slope.last.min) # normalized slope 0-1
    # Get slope of catch in first 10 years
    ct.first          <- ct[1:10]/mean.ct # catch standardized by mean catch
    yrs.first         <- seq(1:10)
    fit.first         <- lm(ct.first ~ yrs.first)
    slope.first       <- as.numeric(coefficients(fit.first)[2])
    slope.first.nrm   <- (slope.first - slope.first.min)/(slope.first.max - slope.first.min) # normalized slope 0-1

    ct_max.1          <- ct[1]/max.ct
    ct_MSY.1          <- ct[1]/MSY.pr
    mean.ct_MSY.start <- mean(ct.raw[1:5])/MSY.pr
    ct_MSY.int        <- ct[which(yr==int.yr)]/MSY.pr
    ct_max.end        <- ct[nyr]/max.ct
    ct_MSY.end        <- ct[nyr]/MSY.pr
    max.ct.i          <- which.max(ct)/nyr
    int.ct.i          <- which(yr==int.yr)/nyr
    min.ct.i          <- which.min(ct)/nyr
    yr.norm           <- (nyr - yr.norm.min)/(yr.norm.max - yr.norm.min) # normalize nyr 0-1

    # classify catch patterns as Flat, LH, LHL, HL, HLH or OTH
    if(min_max >=0.45 & ct_max.1 >= 0.45 & ct_max.end >= 0.45) { Flat <- 1 } else Flat <- 0
    if(min_max<0.25 & ct_max.1<0.45 & ct_max.end>0.45) { LH <- 1 } else LH <- 0
    if(min_max<0.25 & ct_max.1 < 0.45 & ct_max.end < 0.25) { LHL <- 1 } else LHL <- 0
    if(min_max<0.25 & ct_max.1 > 0.5 & ct_max.end < 0.25) { HL <- 1 } else HL <- 0
    if(min_max<0.25 & ct_max.1 >= 0.45 & ct_max.end >= 0.45) { HLH <- 1 } else HLH <- 0
    if(sum(c(Flat,LHL,LH,HL,HLH))<1) { OTH <- 1 } else OTH <- 0

    # Compute predictions for start, end, and int Bk with trained neural networks
    # B/k range that contains 90% of the data points if ct/MSY.pr >= 1
    bk.MSY <- c(0.256 , 0.721 ) # based on all ct/MSY.pr data for 400 stocks # data copied from Plot_ct_MSY_13.R output
    CL.1   <- c( 0.01 , 0.203 )
    CL.2   <- c( 0.2 , 0.431 )
    CL.3   <- c( 0.8 , -0.45 )
    CL.4   <- c( 1.02 , -0.247 )

    # estimate startbio
    # if ct/MSY.pr >= ct_MSY.lim use bk.MSY range
    if(mean.ct_MSY.start >= ct_MSY.lim) {
      startbio    <- bk.MSY
    } else { # else run neural network to determine whether B/k is above or below 0.5
        nninput.start  <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH,min_max,max.ct.i,min.ct.i,yr.norm, #ct_MSY.1,
                                          mean.ct_MSY.start,slope.first.nrm,mean.ct_MSY.end,slope.last.nrm)) #gm.prior.r
        pr.nn.startbio <- compute(nn.startbio, nninput.start)
        pr.nn_indices.startbio <- max.col(pr.nn.startbio$net.result)
        ct_MSY.use     <- ifelse(ct_MSY.1 < mean.ct_MSY.start,ct_MSY.1,mean.ct_MSY.start)
        if(pr.nn_indices.startbio==1) { # if nn predicts B/k below 0.5
          startbio      <- c(CL.1[1]+CL.1[2]*mean.ct_MSY.start,CL.2[1]+CL.2[2]*mean.ct_MSY.start) } else {
            startbio      <- c(CL.3[1]+CL.3[2]*mean.ct_MSY.start,CL.4[1]+CL.4[2]*mean.ct_MSY.start) }
    } # end of neural network loop

    # estimate intbio
    if(ct_MSY.int >= ct_MSY.lim) {
      intbio    <- bk.MSY
    } else { # else run neural network to determine whether B/k is above or below 0.5
    nninput.int    <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH, # shapes
                                          min_max,max.ct.i,min.ct.i,yr.norm, # general
                                          int.ct.i,ct_MSY.int,                   # int
                                          mean.ct_MSY.end,slope.last.nrm,        # end
                                          mean.ct_MSY.start,slope.first.nrm))     # start

    pr.nn.intbio   <- compute(nn.intbio, nninput.int)
    pr.nn_indices.intbio <- max.col(pr.nn.intbio$net.result)
    if(pr.nn_indices.intbio==1){ # if nn predicts B/k below 0.5
      intbio      <- c(CL.1[1]+CL.1[2]*ct_MSY.int,CL.2[1]+CL.2[2]*ct_MSY.int) } else {
          intbio    <- c(CL.3[1]+CL.3[2]*ct_MSY.int,CL.4[1]+CL.4[2]*ct_MSY.int)}
    } # end of nn loop

    # estimate endbio
      # if ct/MSY.pr >= ct_MSY.lim use bk.MSY range
      if(mean.ct_MSY.end >= ct_MSY.lim) {
        endbio    <- bk.MSY
      } else { # else run neural network to determine whether B/k is above or below 0.5
         nninput.end    <- as.data.frame(cbind(Flat,LH,LHL,HL,HLH,OTH,ct_MSY.int,min_max,max.ct.i,  # arbitrary best sequence
                                               int.ct.i,min.ct.i,yr.norm,
                                               mean.ct_MSY.start,slope.first.nrm,mean.ct_MSY.end,slope.last.nrm))
         pr.nn.endbio   <- compute(nn.endbio, nninput.end)
         pr.nn_indices.endbio <- max.col(pr.nn.endbio$net.result)
         ct_MSY.use    <- ifelse(ct_MSY.end < mean.ct_MSY.end,ct_MSY.end,mean.ct_MSY.end)
         if(pr.nn_indices.endbio==1){ # if nn predicts B/k below 0.5
          endbio      <- c(CL.1[1]+CL.1[2]*ct_MSY.use,CL.2[1]+CL.2[2]*ct_MSY.use) } else {
            endbio      <- c(CL.3[1]+CL.3[2]*ct_MSY.use,CL.4[1]+CL.4[2]*ct_MSY.use)}

  } # end of nn loop

    # -------------------------------------------------------
    # if abundance data are available, use to set B/k priors
    #--------------------------------------------------------
    # The following assumes that max smoothed cpue will not exceed carrying capacity and will
    # not be less than a quarter of carrying capacity

    if(btype != "None") {
      # get length, min, max, min/max ratio of smoothed bt data
      start.bt      <- yr[which(bt>0)[1]]
      end.bt        <- yr[max(which(bt>0))]
      yr.bt         <- seq(from=start.bt,to=end.bt,by=1) #range of years with bt data
      bt.no.na      <- approx(bt[yr>=start.bt & yr<=end.bt],n=length(yr.bt))$y
      bt.sm         <- ksmooth(x=yr.bt,y=bt.no.na,kernel="normal",n.points=length(yr.bt),bandwidth=bw)$y
      min.bt.sm     <- min(bt.sm,na.rm=T)
      max.bt.sm     <- max(bt.sm,na.rm=T)
      yr.min.bt.sm  <- yr.bt[which.min(bt.sm)]
      yr.max.bt.sm  <- yr.bt[bt.sm==max.bt.sm]

    # The prior B/k bounds derived from cpue are Bk.cpue.pr.low = 0.25 * cpue/max.cpue
    # and Bk.cpue.pr.hi = 1.0 * cpue/max.cpue
      if(bt4pr == T) { # if B/k priors shall be estimated from CPUE...
      # if cpue is available in first 3 years, use to set startbio
      if(is.na(stb.low)==T & is.na(stb.hi)==T & start.bt <= yr[3]) {
          startbio.bt <- c(0.25*bt.sm[1]/max.bt.sm,bt.sm[1]/max.bt.sm)
          # if first catch is low and cpue close to max, assume unexploited stock
          if(ct[1]/max.ct < 0.2 & bt.sm[1]/max.bt.sm > 0.8) {startbio.bt <- c(0.8,1)}

        # use startbio estimated from bt only if it is narrower or similar to startbio estimated by the neural network
        if((1.25*(startbio[2]-startbio[1])) >  (startbio.bt[2]-startbio.bt[1])) {
          startbio <- startbio.bt }

      } # end of startbio loop

      # use min cpue to set intbio (ignore years close to start or end)
      if(is.na(intb.low)==T & is.na(intb.hi)==T) {
        st.33     <- ifelse(start.bt<(start.yr+3),start.yr+3,start.bt) # first year eligible for intbio
        end.33    <- ifelse(end.bt>(end.yr-3),end.yr-3,end.bt) # last year eligible for intbio
        bt.33     <- bt.sm[yr.bt>=st.33 & yr.bt<=end.33] # CPUE values relevant for intbio
        yr.bt.33  <- seq(from=st.33,to=end.33,by=1) # range of years with relevant bt data
        min.bt.33 <- min(bt.33,na.rm=T) # mimimum of relevant bt
        int.yr.bt <- yr.bt.33[bt.33==min.bt.33] # year with min bt
        intbio.bt <- c(0.25*min.bt.33/max.bt.sm,min.bt.33/max.bt.sm) # intbio prior predicted for int.yr.bt

        # if mean catch/MSY before int.yr is high (> 0.8), use narrower range
        ct.MSY.prev  <- mean(ct[yr>=(int.yr-4) & yr<=int.yr])/MSY.pr
        if(ct.MSY.prev > 0.8) { intbio.bt <- c(1.2*intbio.bt[1],0.8*intbio.bt[2]) }
        # if cpue range is narrow, use lower intbio
        if(min.bt.sm/max.bt.sm>0.3) {intbio.bt <- c(0.8*intbio.bt[1],0.8*intbio.bt[2]) }

        # use intbio estimated from bt only if it is narrower or similar to intbio estimated by the neural network
        if((1.25*(intbio[2]-intbio[1])) >=  (intbio.bt[2]-intbio.bt[1])) {
          int.yr   <- int.yr.bt
          intbio   <- intbio.bt }

      } # end of intbio loop

      # if cpue is within last 3 years of time series, use to set endbio
      if(is.na(endb.low)==T & is.na(endb.hi)==T) {
        if(end.bt >= yr[nyr-2]) {
          endbio.bt  <- c(0.25*bt.sm[yr.bt==end.bt]/max.bt.sm,bt.sm[yr.bt==end.bt]/max.bt.sm)
          # if mean catch/MSY before end.yr is high (> 0.8), use narrower range,
          # because with high previous catch, biomass can neither be very low nor near k
          ct.MSY.prev  <- mean(ct[yr>=(end.yr-4) & yr<=end.yr])/MSY.pr
          if(ct.MSY.prev > 0.8) { endbio.bt <- c(1.2*endbio.bt[1],0.8*endbio.bt[2]) }
          # if endbio estimated by neural network is low and cpue is well below max, use endbio
          if(mean(endbio.bt)>mean(endbio) & mean(endbio)<0.3 & bt.sm[yr.bt==end.bt]/max.bt.sm < 0.7) {endbio.bt <- endbio}
          # if cpue range is narrow, use lower endbio
          if(min.bt.sm/max.bt.sm>0.3) {endbio.bt <- c(0.8*endbio.bt[1],0.8*endbio.bt[2]) }

        # use endbio estimated from bt only if it is narrower or similar to endbio estimated by the neural network
          if((1.25*(endbio[2]-endbio[1])) >  (endbio.bt[2]-endbio.bt[1])) {
            endbio   <- endbio.bt }
        }
      } # end of endbio loop
     } # end of b/k prior loop
    } # end of bt priors loop

  # if user defined B/k priors in the ID file, use those
    if(is.na(stb.low)==F & is.na(stb.hi)==F) {startbio <- c(stb.low,stb.hi)}
    if(is.na(intb.low)==F & is.na(intb.hi)==F) {
      int.yr   <- cinfo$int.yr[cinfo$Stock==stock]
      intbio   <- c(intb.low,intb.hi)}
    if(is.na(endb.low)==F & is.na(endb.hi)==F) {endbio   <- c(endb.low,endb.hi)}

  cat("startbio=",startbio,ifelse(is.na(stb.low)==T,"default","expert"),
      ", intbio=",int.yr,intbio,ifelse(is.na(intb.low)==T,"default","expert"),
      ", endbio=",endbio,ifelse(is.na(endb.low)==T,"default","expert"),"\n")


  #-----------------------------------------------------------------
  #Plot data and progress -----
  #-----------------------------------------------------------------
  # check for operating system, open separate window for graphs if Windows
  if(grepl("win",tolower(Sys.info()['sysname']))) {windows(14,9)}
  par(mfrow=c(2,3),mar=c(5.1,4.5,4.1,2.1))
  # (a): plot catch ----
  plot(x=yr, y=ct.raw,
       ylim=c(0,max(ifelse(substr(id_file,1,3)=="Sim",
                           1.1*true.MSY,0),1.2*max(ct.raw))),
       type ="l", bty="l", main=paste("A:",gsub(":","",gsub("/","-",stock))), xlab="", ylab="Catch (1000 tonnes/year)", lwd=2, cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
  lines(x=yr,y=ct,col="blue", lwd=1)
  points(x=yr[max.yr.i], y=max.ct, col="red", lwd=2)
  points(x=yr[min.yr.i], y=min.ct, col="red", lwd=2)
  lines(x=yr,y=rep(MSY.pr,length(yr)),lty="dotted",col="purple")
  if(substr(id_file,1,3)=="Sim") lines(x=yr,y=rep(true.MSY,length(yr)),lty="dashed",col="green")

  # (b): plot r-k graph
  plot(x=ri1, y=ki1, xlim = c(0.95*quantile(ri1,0.001),1.2*quantile(ri1,0.999)),
       ylim = c(0.95*quantile(ki1,0.001),1.2*quantile(ki1,0.999)),
       log="xy", xlab="r", ylab="k (1000 tonnes)", main="B: Finding viable r-k", pch=".", cex=2, bty="l",
       col=grey(0.7,0.4), cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
  lines(x=c(prior.r[1],prior.r[2],prior.r[2],prior.r[1],prior.r[1]), # plot original prior range
        y=c(prior.k[1],prior.k[1],prior.k[2],prior.k[2],prior.k[1]),
        lty="dotted")

  #---------------------------------------------------------------------
  # Prepare MCMC analyses
  #---------------------------------------------------------------------
  # set inits for r-k in lower right corner of log r-k space to avoid intermediate maxima
  init.r      <- prior.r[1]+0.8*(prior.r[2]-prior.r[1])
  init.k      <- prior.k[1]+0.1*(prior.k[2]-prior.k[1])

  # vector with no penalty (=0) if predicted biomass is within viable range, else a penalty of 10 is set
  pen.bk = pen.F = rep(0,length(ct))

  # Add biomass priors
  b.yrs = c(1,length(start.yr:int.yr),length(start.yr:end.yr))
  b.prior = rbind(matrix(c(startbio[1],startbio[2],intbio[1],intbio[2],endbio[1],endbio[2]),2,3),rep(0,3)) # last row includes the 0 penalty

  #----------------------------------------------------------------
  # First run of BSM with only catch data = CMSY++
  #----------------------------------------------------------------
    # changes by RF to account for asymmetric distributions
    bt.start  <- mean(c(prior.k[1]*startbio[1],prior.k[2]*startbio[2])) # derive proxy for first bt value
    bt.cmsy   <- c(bt.start,rep(NA,length(ct)-1)) # create proxy abundance with one start value and rest = NA
    bt.int    <- mean(c(prior.k[1]*intbio[1],prior.k[2]*intbio[2]))
    bt.last  <- mean(c(prior.k[1]*endbio[1],prior.k[2]*endbio[2]))

    mean.cmsy.ct   <- mean(c(ct[1],ct[yr==int.yr],ct[nyr]),na.rm=T) # get mean catch of years with prior bt
    mean.cmsy.cpue <- mean(c(bt.start,bt.int,bt.last),na.rm=T) # get mean of prior bt

    q.prior.cmsy    <- c(0.99,1.01) # since no abundance data are available in this run,
    init.q.cmsy     <- 1            # q could be omitted and is set here to (practically) 1

  cat("Running MCMC analysis with only catch data....\n")

  # call Schaefer model function
  jags_cmsy <- bsm(ct=ct,btj=bt.cmsy,nyr=nyr,prior.r=prior.r,prior.k=prior.k,startbio=startbio,q.priorj=q.prior.cmsy,
                      init.q=init.q.cmsy,init.r=init.r,init.k=init.k,pen.bk=pen.bk,pen.F=pen.F,b.yrs=b.yrs,
                      b.prior=b.prior,CV.C=CV.C,CV.cpue=CV.cpue,nbk=nbk,rk.cor=rk.cor,cmsyjags=TRUE)

  #-----------------------------------------------
  # Get CMSY++ results
  #-----------------------------------------------
  rs                <- as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$r))   # unique.rk[,1]
  ks                <- as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$k))  # unique.rk[,2]
  ellipse.cmsy       <- traceEllipse(rs,ks,prior.r,prior.k) # GP
  r.cmsy             <- ellipse.cmsy[1] # GP
  k.cmsy             <- ellipse.cmsy[2] # GP
  # restrict CI quantiles to above 25th percentile of rs
  rs.025             <- as.numeric(quantile(rs,0.025))
  r.quant.cmsy       <- as.numeric(quantile(rs[rs>rs.025],c(0.5,0.025,0.975))) # median, 95% CIs in range around
  k.quant.cmsy       <- as.numeric(quantile(ks[rs>rs.025],c(0.5,0.025,0.975)))
  lcl.r.cmsy         <- r.quant.cmsy[2]
  ucl.r.cmsy         <- r.quant.cmsy[3]
  lcl.k.cmsy         <- k.quant.cmsy[2]
  ucl.k.cmsy         <- k.quant.cmsy[3]
  MSY.quant.cmsy     <- quantile(rs[rs>rs.025]*ks[rs>rs.025]/4,c(0.5,0.025,0.975))
  MSY.cmsy           <- r.cmsy*k.cmsy/4
  lcl.MSY.cmsy       <- MSY.quant.cmsy[2]
  ucl.MSY.cmsy       <- MSY.quant.cmsy[3]
  qs                 <- as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$q))
  q.quant.cmsy       <- quantile(qs,c(0.5,0.025,0.975))
  q.cmsy             <- q.quant.cmsy[1]
  lcl.q.cmsy         <- q.quant.cmsy[2]
  ucl.q.cmsy         <- q.quant.cmsy[3]

  Fmsy.quant.cmsy    <- as.numeric(quantile(rs[rs>rs.025]/2,c(0.5,0.025,0.975)))
  Fmsy.cmsy          <- r.cmsy/2 # HW checked
  lcl.Fmsy.cmsy      <- Fmsy.quant.cmsy[2] #><>HW to be added to report output
  ucl.Fmsy.cmsy      <- Fmsy.quant.cmsy[3] #><>HW to be added to report output
  Bmsy.quant.cmsy    <- as.numeric(quantile(ks[rs>rs.025]/2,c(0.5,0.025,0.975)))
  Bmsy.cmsy          <- k.cmsy/2 # HW checked
  lcl.Bmsy.cmsy      <- Bmsy.quant.cmsy[2] #><>HW to be added to report output
  ucl.Bmsy.cmsy      <- Bmsy.quant.cmsy[3] #><>HW to be added to report output
  # HW posterior predictives can stay unchanged
  ppd.r              <- exp(as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$ppd.logrk[,1])))
  ppd.k              <- exp(as.numeric(mcmc(jags_cmsy$BUGSoutput$sims.list$ppd.logrk[,2])))

  #><>HW get FFmsy directly from JAGS
  all.FFmsy.cmsy  = jags_cmsy$BUGSoutput$sims.list$FFmsy
  FFmsy.quant.cmsy = apply(all.FFmsy.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
  FFmsy.cmsy = FFmsy.quant.cmsy[1,]
  lcl.FFmsy.cmsy = FFmsy.quant.cmsy[2,]
  ucl.FFmsy.cmsy = FFmsy.quant.cmsy[3,]
  #><>HW get BBmsy directly from JAGS
  all.BBmsy.cmsy  = jags_cmsy$BUGSoutput$sims.list$BBmsy
  BBmsy.quant.cmsy = apply(all.BBmsy.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
  BBmsy.cmsy = BBmsy.quant.cmsy[1,]
  lcl.BBmsy.cmsy = BBmsy.quant.cmsy[2,]
  ucl.BBmsy.cmsy = BBmsy.quant.cmsy[3,]
  # get relative biomass P=B/k as predicted by BSM, including predictions for years with NA abundance
  all.bk.cmsy  = jags_cmsy$BUGSoutput$sims.list$P
  bk.quant.cmsy = apply(all.bk.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
  bk.cmsy = bk.quant.cmsy[1,]
  lcl.bk.cmsy = bk.quant.cmsy[2,]
  ucl.bk.cmsy = bk.quant.cmsy[3,]
  #><> NEW get biomass from JAGS posterior
  all.B.cmsy  = jags_cmsy$BUGSoutput$sims.list$B
  B.quant.cmsy = apply(all.B.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
  B.cmsy = B.quant.cmsy[1,]
  lcl.B.cmsy = B.quant.cmsy[2,]
  ucl.B.cmsy = B.quant.cmsy[3,]
  #><> NEW get F from JAGS posterior
  all.Ft.cmsy  = jags_cmsy$BUGSoutput$sims.list$F
  Ft.quant.cmsy = apply(all.Ft.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
  Ft.cmsy = Ft.quant.cmsy[1,]
  lcl.Ft.cmsy = Ft.quant.cmsy[2,]
  ucl.Ft.cmsy = Ft.quant.cmsy[3,]

  # get catch estimates given catch CV
  all.ct.cmsy  = jags_cmsy$BUGSoutput$sims.list$ct.jags
  ct.quants.cmsy = apply(all.ct.cmsy,2,quantile,c(0.5,0.025,0.975),na.rm=T)
  ct.cmsy          <- ct.quants.cmsy[1,]
  lcl.ct.cmsy      <- ct.quants.cmsy[2,]
  ucl.ct.cmsy      <- ct.quants.cmsy[3,]

  #-------------------------------------------------------------------
  # Plot results
  #-------------------------------------------------------------------
  # (b) continued
  # plot viable r-k pairs from catch-only BSM run
  points(x=rs,y=ks,pch=".",cex=1,col="gray55")

  # show CMSY++ estimate in prior space of graph B
  points(x=r.cmsy, y=k.cmsy, pch=19, col="blue")
  lines(x=c(lcl.r.cmsy, ucl.r.cmsy),y=c(k.cmsy,k.cmsy), col="blue")
  lines(x=c(r.cmsy,r.cmsy),y=c(lcl.k.cmsy, ucl.k.cmsy), col="blue")

  lines(x=c(prior.r[1],prior.r[2],prior.r[2],prior.r[1],prior.r[1]), # re-plot original prior range
        y=c(prior.k[1],prior.k[1],prior.k[2],prior.k[2],prior.k[1]),lty="dotted")

  # ------------------------------------------------------------------
  # Second run with Bayesian analysis of catch & biomass (or CPUE) with Schaefer model ----
  # ------------------------------------------------------------------
    FullSchaefer <- F
 #   bt           <- bt.raw
    if(btype != "None" & length(bt[is.na(bt)==F])>=nab) {
    FullSchaefer <- T
    cat("Running MCMC analysis with catch and CPUE.... \n")

    if(btype=="biomass") {
        q.prior <- q.biomass.pr
        init.q  <- mean(q.prior)
    } else { # if btype is CPUE
      # get mean of 3 highest bt values
      bt.sort <- sort(bt)
      mean.max.bt <- mean(bt.sort[(length(bt.sort)-2):length(bt.sort)],na.rm = T)
      # Estimate q.prior[2] from max cpue = q * k, q.prior[1] from max cpue = q * 0.25 * k
      q.1           <- mean.max.bt/prior.k[2]
      q.2           <- mean.max.bt/(0.25*prior.k[1])
      q.prior       <- c(q.1,q.2)
      q.init        <- mean(q.prior) }

    # call Schaefer model function
    jags_bsm <- bsm(ct=ct,btj=bt,nyr=nyr,prior.r=prior.r,prior.k=prior.k,startbio=startbio,q.priorj=q.prior,
                        init.q=init.q,init.r=init.r,init.k=init.k,pen.bk=pen.bk,pen.F=pen.F,b.yrs=b.yrs,
                        b.prior=b.prior,CV.C=CV.C,CV.cpue=CV.cpue,nbk=nbk,rk.cor=rk.cor,cmsyjags=FALSE)

    # --------------------------------------------------------------
    # Results from BSM Schaefer - ><>HW now consistent with CMSY++
    # --------------------------------------------------------------
    rs.bsm            <- as.numeric(mcmc(jags_bsm$BUGSoutput$sims.list$r))   # unique.rk[,1]
    ks.bsm            <- as.numeric(mcmc(jags_bsm$BUGSoutput$sims.list$k))  # unique.rk[,2]
    #><> HW: Go directly with posterior median and CIs (non-parametric)
    r.quant.bsm       <- as.numeric(quantile(rs.bsm,c(0.5,0.025,0.975))) #median, 95% CIs
    r.bsm             <- r.quant.bsm[1]
    lcl.r.bsm         <- r.quant.bsm[2]
    ucl.r.bsm         <- r.quant.bsm[3]
    k.quant.bsm          <- as.numeric(quantile(ks.bsm,c(0.5,0.025,0.975)))
    k.bsm             <- k.quant.bsm[1]
    lcl.k.bsm         <- k.quant.bsm[2]
    ucl.k.bsm         <- k.quant.bsm[3]
    MSY.quant.bsm     <- quantile(rs.bsm*ks.bsm/4,c(0.5,0.025,0.975))
    MSY.bsm           <- MSY.quant.bsm[1]
    lcl.MSY.bsm       <- MSY.quant.bsm[2]
    ucl.MSY.bsm       <- MSY.quant.bsm[3]
    qs.bsm            <- as.numeric(mcmc(jags_bsm$BUGSoutput$sims.list$q))
    q.quant.bsm       <- as.numeric(quantile(qs.bsm,c(0.5,0.025,0.975)))
    q.bsm             <- q.quant.bsm[1]
    lcl.q.bsm         <- q.quant.bsm[2]
    ucl.q.bsm         <- q.quant.bsm[3]

    Fmsy.quant.bsm      <- as.numeric(quantile(rs.bsm/2,c(0.5,0.025,0.975)))
    Fmsy.bsm           <- Fmsy.quant.bsm[1]
    lcl.Fmsy.bsm        <- Fmsy.quant.bsm[2] #><>HW to be added to report output
    ucl.Fmsy.bsm        <- Fmsy.quant.bsm[3] #><>HW to be added to report output
    Bmsy.quant.bsm      <- as.numeric(quantile(ks.bsm/2,c(0.5,0.025,0.975)))
    Bmsy.bsm            <- Bmsy.quant.bsm[1]
    lcl.Bmsy.bsm        <- Bmsy.quant.bsm[2] #><>HW to be added to report output
    ucl.Bmsy.bsm        <- Bmsy.quant.bsm[3] #><>HW to be added to report output

    #><>HW get FFmsy directly from JAGS
    all.FFmsy.bsm  = jags_bsm$BUGSoutput$sims.list$FFmsy
    FFmsy.quant.bsm = apply(all.FFmsy.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
    FFmsy.bsm = FFmsy.quant.bsm[1,]
    lcl.FFmsy.bsm = FFmsy.quant.bsm[2,]
    ucl.FFmsy.bsm = FFmsy.quant.bsm[3,]
    #><>HW get BBmsy directly from JAGS
    all.BBmsy.bsm  = jags_bsm$BUGSoutput$sims.list$BBmsy
    BBmsy.quant.bsm = apply(all.BBmsy.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
    BBmsy.bsm = BBmsy.quant.bsm[1,]
    lcl.BBmsy.bsm = BBmsy.quant.bsm[2,]
    ucl.BBmsy.bsm = BBmsy.quant.bsm[3,]
    # get relative biomass P=B/k as predicted by BSM, including predictions for years with NA abundance
    all.bk.bsm  = jags_bsm$BUGSoutput$sims.list$P
    bk.quant.bsm = apply(all.bk.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
    bk.bsm = bk.quant.bsm[1,]
    lcl.bk.bsm = bk.quant.bsm[2,]
    ucl.bk.bsm = bk.quant.bsm[3,]
    #><> NEW get biomass from JAGS posterior
    all.B.bsm  = jags_bsm$BUGSoutput$sims.list$B
    B.quant.bsm = apply(all.B.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
    B.bsm = B.quant.bsm[1,]
    lcl.B.bsm = B.quant.bsm[2,]
    ucl.B.bsm = B.quant.bsm[3,]
    #><> NEW get F from JAGS posterior
    all.Ft.bsm  = jags_bsm$BUGSoutput$sims.list$F
    Ft.quant.bsm = apply(all.Ft.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
    Ft.bsm = Ft.quant.bsm[1,]
    lcl.Ft.bsm = Ft.quant.bsm[2,]
    ucl.Ft.bsm = Ft.quant.bsm[3,]

    # get catch estimates given catch CV
    all.ct.bsm  = jags_bsm$BUGSoutput$sims.list$ct.jags
    ct.quants.bsm = apply(all.ct.bsm,2,quantile,c(0.5,0.025,0.975),na.rm=T)
    ct.bsm          <- ct.quants.bsm[1,]
    lcl.ct.bsm      <- ct.quants.bsm[2,]
    ucl.ct.bsm      <- ct.quants.bsm[3,]

    #-------------------------------------------
    # BSM fits
    #-------------------------------------------
    #><> HW PLOT E (observations)
    F.bt.jags       <- q.bsm*ct.raw/bt # F from raw data
    F.bt_Fmsy.jags  <- vector() # initialize vector
    for(z in 1: length(F.bt.jags)) {
      F.bt_Fmsy.jags[z] <- ifelse(is.na(bt[z])==T,NA,F.bt.jags[z]/
                                    ifelse(((bt[z]/q.bsm)/k.bsm)<0.25,Fmsy.bsm*4*(bt[z]/q.bsm)/k.bsm,Fmsy.bsm))}

    #><> get cpue fits from BSM
    cpue.bsm        <- exp(jags_bsm$BUGSoutput$sims.list$cpuem)
    pe.logbt.bsm   <- (jags_bsm$BUGSoutput$sims.list$proc.logB)
    # get cpue predicted
    pred.cpue            <- apply(cpue.bsm,2,quantile,c(0.5,0.025,0.975))
    cpue.bsm          <- pred.cpue[1,]
    lcl.cpue.bsm      <- pred.cpue[2,]
    ucl.cpue.bsm      <- pred.cpue[3,]
    # get process error on log(biomass)   pred.cpue            <- apply(cpue.jags,2,quantile,c(0.5,0.025,0.975))
    pred.pe         <- apply(pe.logbt.bsm,2,quantile,c(0.5,0.025,0.975))
    pe.bsm         <- pred.pe[1,]
    lcl.pe.bsm     <- pred.pe[2,]
    ucl.pe.bsm     <- pred.pe[3,]


    # get variance and correlation between log(r) and log(k)
    log.r.var    <- var(x=log(rs.bsm))
    log.k.var    <- var(x=log(ks.bsm))
    log.kr.cor   <- cor(x=log(rs.bsm),y=log(ks.bsm))
    log.kr.cov   <- cov(x=log(rs.bsm),y=log(ks.bsm))

  } # end of MCMC BSM Schaefer loop

  # --------------------------------------------
  # Get results for management ----
  # --------------------------------------------
  if(FullSchaefer==F | force.cmsy==T) { # if only CMSY is available or shall be used
    MSY   <-MSY.cmsy; lcl.MSY<-lcl.MSY.cmsy; ucl.MSY<-ucl.MSY.cmsy
    Bmsy  <-Bmsy.cmsy; lcl.Bmsy<-lcl.Bmsy.cmsy; ucl.Bmsy<-ucl.Bmsy.cmsy
    Fmsy  <-Fmsy.cmsy; lcl.Fmsy<-lcl.Fmsy.cmsy; ucl.Fmsy<-ucl.Fmsy.cmsy
    F.Fmsy<-FFmsy.cmsy;lcl.F.Fmsy<-lcl.FFmsy.cmsy; ucl.F.Fmsy<-ucl.FFmsy.cmsy
    B.Bmsy<-BBmsy.cmsy[1:nyr];lcl.B.Bmsy<-lcl.BBmsy.cmsy[1:nyr][1:nyr];ucl.B.Bmsy<-ucl.BBmsy.cmsy[1:nyr]
    B <- B.cmsy[1:nyr];lcl.B<-lcl.B.cmsy[1:nyr][1:nyr];ucl.B<-ucl.B.cmsy[1:nyr]
    Ft <- Ft.cmsy[1:nyr];lcl.Ft<-lcl.Ft.cmsy[1:nyr][1:nyr];ucl.Ft<-ucl.Ft.cmsy[1:nyr]
    bk <- bk.cmsy[1:nyr];lcl.bk<-lcl.bk.cmsy[1:nyr][1:nyr];ucl.bk<-ucl.bk.cmsy[1:nyr]

    ct.jags <- ct.cmsy; lcl.ct.jags = lcl.ct.cmsy; ucl.ct.jags=ucl.ct.cmsy #catch estimate given catch error

  } else { # if FullSchaefer is TRUE
    MSY   <-MSY.bsm; lcl.MSY<-lcl.MSY.bsm; ucl.MSY<-ucl.MSY.bsm
    Bmsy  <-Bmsy.bsm; lcl.Bmsy<-lcl.Bmsy.bsm; ucl.Bmsy<-ucl.Bmsy.bsm
    Fmsy  <-Fmsy.bsm; lcl.Fmsy<-lcl.Fmsy.bsm; ucl.Fmsy<-ucl.Fmsy.bsm
    F.Fmsy<-FFmsy.bsm;lcl.F.Fmsy<-lcl.FFmsy.bsm; ucl.F.Fmsy<-ucl.FFmsy.bsm
    B.Bmsy<-BBmsy.bsm[1:nyr];lcl.B.Bmsy<-lcl.BBmsy.bsm[1:nyr][1:nyr];ucl.B.Bmsy<-ucl.BBmsy.bsm[1:nyr]
    B <- B.bsm[1:nyr];lcl.B<-lcl.B.bsm[1:nyr][1:nyr];ucl.B<-ucl.B.bsm[1:nyr]
    Ft <- Ft.bsm[1:nyr];lcl.Ft<-lcl.Ft.bsm[1:nyr][1:nyr];ucl.Ft<-ucl.Ft.bsm[1:nyr]
    bk <- bk.bsm[1:nyr];lcl.bk<-lcl.bk.bsm[1:nyr][1:nyr];ucl.bk<-ucl.bk.bsm[1:nyr]
    ct.jags <- ct.bsm; lcl.ct.jags = lcl.ct.bsm; ucl.ct.jags=ucl.ct.bsm #catch estimate given catch error

  }

  #><> New section simplified for CMSY++ and BSM
    Fmsy.adj     <- ifelse(B.Bmsy>0.5,Fmsy,Fmsy*2*B.Bmsy)
    lcl.Fmsy.adj <- ifelse(B.Bmsy>0.5,lcl.Fmsy,lcl.Fmsy*2*B.Bmsy)
    ucl.Fmsy.adj <- ifelse(B.Bmsy>0.5,ucl.Fmsy,ucl.Fmsy*2*B.Bmsy)

    if(is.na(sel.yr)==F){
      B.Bmsy.sel<-B.Bmsy[yr==sel.yr]
      B.sel<-B.Bmsy.sel*Bmsy
      F.sel<-ct.raw[yr==sel.yr]/B.sel
      F.Fmsy.sel<-F.sel/Fmsy.adj[yr==sel.yr]
    }

  # ------------------------------------------
  # print input and results to screen ----
  #-------------------------------------------
  cat("---------------------------------------\n")
  cat("Species:", cinfo$ScientificName[cinfo$Stock==stock], ", stock:",stock,", ",ename,"\n")
  cat(cinfo$Name[cinfo$Stock==stock], "\n")
  cat("Region:",cinfo$Region[cinfo$Stock==stock],",",cinfo$Subregion[cinfo$Stock==stock],"\n")
  cat("Catch data used from years", min(yr),"-", max(yr),", abundance =", btype, "\n")
  cat("Prior initial relative biomass =", startbio[1], "-", startbio[2],ifelse(is.na(stb.low)==T,"default","expert"), "\n")
  cat("Prior intermediate rel. biomass=", intbio[1], "-", intbio[2], "in year", int.yr,ifelse(is.na(intb.low)==T,"default","expert"), "\n")
  cat("Prior final relative biomass   =", endbio[1], "-", endbio[2],ifelse(is.na(endb.low)==T,"default","expert"), "\n")
  cat("Prior range for r =", format(prior.r[1],digits=2), "-", format(prior.r[2],digits=2),ifelse(is.na(r.low)==T,"default","expert"),
      ", prior range for k =", prior.k[1], "-", prior.k[2],", MSY prior =",MSY.pr,"\n")
  # if Schaefer and CPUE, print prior range of q
  if(FullSchaefer==T) {
    cat("B/k prior used for first year in BSM",ifelse(nbk>1,"and intermediate year",""),ifelse(nbk==3,"and last year",""),"\n")
    cat("Prior range of q =",q.prior[1],"-",q.prior[2],", assumed effort creep",e.creep,"%\n") }
  if(substr(id_file,1,3)=="Sim") { # if data are simulated, print true values
    cat("True values: r =",true.r,", k = 1000, MSY =", true.MSY,", last B/k =", true.Bk,
        ", last F/Fmsy =",true.F_Fmsy,", q = 0.01\n") }

  # results of CMSY analysis
  cat("\nResults of CMSY analysis \n")
  cat("-------------------------\n")
  cat("r   =", r.cmsy,", 95% CL =", lcl.r.cmsy, "-", ucl.r.cmsy,", k =", k.cmsy,", 95% CL =", lcl.k.cmsy, "-", ucl.k.cmsy,"\n")
  cat("MSY =", MSY.cmsy,", 95% CL =", lcl.MSY.cmsy, "-", ucl.MSY.cmsy,"\n")
  cat("Relative biomass in last year =", bk.cmsy[nyr], "k, 2.5th perc =", lcl.bk.cmsy[nyr],
      ", 97.5th perc =", ucl.bk.cmsy[nyr],"\n")
  cat("Exploitation F/(r/2) in last year =", FFmsy.cmsy[nyr],", 2.5th perc =",lcl.FFmsy.cmsy[nyr],
      ", 97.5th perc =",ucl.FFmsy.cmsy[nyr],"\n\n")


  # print results from full Schaefer if available
  if(FullSchaefer==T) {
    cat("Results from Bayesian Schaefer model (BSM) using catch &",btype,"\n")
    cat("------------------------------------------------------------\n")
    cat("q   =", q.bsm,", lcl =", lcl.q.bsm, ", ucl =", ucl.q.bsm,"(derived from catch and CPUE) \n")
    cat("r   =", r.bsm,", 95% CL =", lcl.r.bsm, "-", ucl.r.bsm,", k =", k.bsm,", 95% CL =", lcl.k.bsm, "-", ucl.k.bsm,", r-k log correlation =", log.kr.cor,"\n")
    cat("MSY =", MSY.bsm,", 95% CL =", lcl.MSY.bsm, "-", ucl.MSY.bsm,"\n")
    cat("Relative biomass in last year =", bk.bsm[nyr], "k, 2.5th perc =",lcl.bk.bsm[nyr],
        ", 97.5th perc =", ucl.bk.bsm[nyr],"\n")
    cat("Exploitation F/(r/2) in last year =", FFmsy.bsm[nyr],", 2.5th perc =",lcl.FFmsy.bsm[nyr],
        ", 97.5th perc =",ucl.FFmsy.bsm[nyr],"\n\n")
  }

  # print results to be used in management
  cat("Results for Management (based on",ifelse(FullSchaefer==F | force.cmsy==T,"CMSY","BSM"),"analysis) \n")
  cat("-------------------------------------------------------------\n")
  if(force.cmsy==T) cat("Mangement results based on CMSY because abundance data seem unrealistic\n")
  cat("Fmsy =",Fmsy,", 95% CL =",lcl.Fmsy,"-",ucl.Fmsy,"(if B > 1/2 Bmsy then Fmsy = 0.5 r)\n")
  cat("Fmsy =",Fmsy.adj[nyr],", 95% CL =",lcl.Fmsy.adj[nyr],"-",ucl.Fmsy.adj[nyr],"(r and Fmsy are linearly reduced if B < 1/2 Bmsy)\n")
  cat("MSY  =",MSY,", 95% CL =",lcl.MSY,"-",ucl.MSY,"\n")
  cat("Bmsy =",Bmsy,", 95% CL =",lcl.Bmsy,"-",ucl.Bmsy,"\n")
  cat("Biomass in last year =",B[nyr],", 2.5th perc =", lcl.B[nyr], ", 97.5 perc =",ucl.B[nyr],"\n")
  cat("B/Bmsy in last year  =",B.Bmsy[nyr],", 2.5th perc =", lcl.B.Bmsy[nyr], ", 97.5 perc =",ucl.B.Bmsy[nyr],"\n")
  cat("Fishing mortality in last year =",Ft[nyr],", 2.5th perc =", lcl.Ft[nyr], ", 97.5 perc =",ucl.Ft[nyr],"\n")
  cat("Exploitation F/Fmsy  =",F.Fmsy[nyr],", 2.5th perc =", lcl.F.Fmsy[nyr], ", 97.5 perc =",ucl.F.Fmsy[nyr],"\n")

  # show stock status and exploitation for optional selected year
  if(is.na(sel.yr)==F) {
    cat("\nStock status and exploitation in",sel.yr,"\n")
    cat("Biomass =",B.sel, ", B/Bmsy =",B.Bmsy.sel,", F =",F.sel,", F/Fmsy =",F.Fmsy.sel,"\n") }

  cat("Comment:", comment,"\n")
  cat("----------------------------------------------------------\n")

  # -----------------------------------------
  # Plot results ----
  # -----------------------------------------
  # (b) continued
  # plot best r-k from full Schaefer analysis in prior space of graph B
  if(FullSchaefer==T) {
    points(x=r.bsm, y=k.bsm, pch=19, col="red")
    lines(x=c(lcl.r.bsm, ucl.r.bsm),y=c(k.bsm,k.bsm), col="red")
    lines(x=c(r.bsm,r.bsm),y=c(lcl.k.bsm, ucl.k.bsm), col="red")
  }
  if(substr(id_file,1,3)=="Sim") points(x=true.r,y=true.k,col="green", cex=3, lwd=2)


  # (c) Analysis of viable r-k plot -----
  # ----------------------------
  max.y    <- max(c(ifelse(FullSchaefer==T,max(ks.bsm,ucl.k.bsm),NA),
                  ifelse(substr(id_file,1,3)=="Sim",1.2*true.k,NA),ks),na.rm=T)
  min.y    <- min(c(ifelse(FullSchaefer==T,min(ks.bsm),NA),ks,
                  ifelse(substr(id_file,1,3)=="Sim",0.8*true.k,NA)),na.rm=T)
  max.x    <- max(c(ifelse(FullSchaefer==T,max(rs.bsm),NA),rs),na.rm=T)
  min.x    <- min(c(ifelse(FullSchaefer==T,min(rs.bsm),NA),0.9*lcl.r.cmsy,prior.r[1],rs),na.rm=T)

  plot(x=rs, y=ks, xlim=c(min.x,max.x),
       ylim=c(min.y,max.y),
       pch=16, col="gray",log="xy", bty="l",
       xlab="", ylab="k (1000 tonnes)", main="C: Analysis of viable r-k",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
  title(xlab = "r", line = 2.25, cex.lab = 1.55)

  # plot r-k pairs from MCMC
  if(FullSchaefer==T) {points(x=rs.bsm, y=ks.bsm, pch=16,cex=0.5)}

  # plot best r-k from full Schaefer analysis
  if(FullSchaefer==T) {
    points(x=r.bsm, y=k.bsm, pch=19, col="red")
    lines(x=c(lcl.r.bsm, ucl.r.bsm),y=c(k.bsm,k.bsm), col="red")
    lines(x=c(r.bsm,r.bsm),y=c(lcl.k.bsm, ucl.k.bsm), col="red")
  }

  # plot blue dot for CMSY r-k, with 95% CL lines
  points(x=r.cmsy, y=k.cmsy, pch=19, col="blue")
  lines(x=c(lcl.r.cmsy, ucl.r.cmsy),y=c(k.cmsy,k.cmsy), col="blue")
  lines(x=c(r.cmsy,r.cmsy),y=c(lcl.k.cmsy, ucl.k.cmsy), col="blue")

  if(substr(id_file,1,3)=="Sim") points(x=true.r,y=true.k,col="green", cex=3, lwd=2)

  # (d) Pred. biomass plot ----
  #--------------------
  # determine k to use for red line in b/k plot
  if(FullSchaefer==T)  {k2use <- k.bsm} else {k2use <- k.cmsy}
  # determine hight of y-axis in plot
  max.y  <- max(c(ucl.bk.cmsy,ifelse(FullSchaefer==T,max(ucl.bk.bsm[1:nyr]),NA),
                  ifelse(FullSchaefer==T,max(bt/(q.bsm*k.bsm),na.rm=T),NA),
                  0.6,startbio[2],endbio[2],intbio[2]),na.rm=T)
  max.y  <- ifelse(max.y>4,4,max.y)
  # Main plot of relative CMSY biomass
  plot(x=yr,y=bk.cmsy[1:nyr], lwd=1.5, xlab="", ylab="Relative biomass B/k", type="l",
       ylim=c(0,max.y), bty="l", main="D: Stock size",col="blue",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
  lines(x=yr, y=lcl.bk.cmsy[1:nyr],type="l",lty="dotted",col="blue")
  lines(x=yr, y=ucl.bk.cmsy[1:nyr],type="l",lty="dotted",col="blue")
  # plot lines for 0.5 and 0.25 biomass
  abline(h=0.5, lty="dashed")
  abline(h=0.25, lty="dotted")
  # Add BSM
  if(FullSchaefer==T){
   lines(x=yr, y=bk.bsm[1:nyr],type="l",col="red")
   lines(x=yr, y=lcl.bk.bsm[1:nyr],type="l",lty="dotted",col="red")
   lines(x=yr, y=ucl.bk.bsm[1:nyr],type="l",lty="dotted",col="red")
   # Add CPUE points
   points(x=yr,y=bt/(q.bsm*k.bsm),pch=21,bg="grey")
  }
  # plot biomass windows
  lines(x=c(yr[1],yr[1]), y=startbio, col="purple",lty=ifelse(is.na(stb.low)==T,"dotted","solid"))
  lines(x=c(int.yr,int.yr), y=intbio, col="purple",lty=ifelse(is.na(intb.low)==T,"dotted","solid"))
  lines(x=c(max(yr),max(yr)), y=endbio, col="purple",lty=ifelse(is.na(endb.low)==T,"dotted","solid"))

  # if CPUE has been corrected for effort creep, display uncorrected CPUE
  if(btype=="CPUE" & FullSchaefer==T & e.creep.line==T & is.na(e.creep)==FALSE) {
    lines(x=yr,y=bt.raw/(q.bsm*k.bsm),type="l", col="green", lwd=1)
  }
  if(substr(id_file,1,3)=="Sim") points(x=yr[nyr],y=true.Bk,col="green", cex=3, lwd=2)

  # (e) Exploitation rate plot ----
  # -------------------------
  # if CPUE data are available but fewer than nab years, plot on second axis
  if(btype == "CPUE" | btype=="biomass") {
    q=1/(max(bk.cmsy[1:nyr][is.na(bt)==F],na.rm=T)*k.cmsy/max(bt,na.rm=T))
    u.cpue      <- q.bsm*ct/bt
  }
  # determine upper bound of Y-axis
  max.y <- max(c(1.5,ucl.FFmsy.cmsy,ifelse(FullSchaefer==T,max(c(ucl.FFmsy.bsm),na.rm=T),NA),na.rm=T),na.rm=T)
  max.y <- ifelse(max.y>10,10,max.y)
  # plot F from CMSY
  plot(x=yr,y=FFmsy.cmsy, type="l", bty="l", lwd=1.5, ylim=c(0,max.y), xlab="",
       ylab=expression(F/F[MSY]), main="E: Exploitation rate", col="blue",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
  lines(x=yr,y=lcl.FFmsy.cmsy,lty="dotted",col="blue")
  lines(x=yr,y=ucl.FFmsy.cmsy,lty="dotted",col="blue")
  abline(h=1, lty="dashed")

  # plot F/Fmsy as points from observed catch and CPUE and as red curves from BSM predicted catch and biomass
  if(FullSchaefer==T){
    points(x=yr, y=F.bt_Fmsy.jags, pch=21,bg="grey")
    lines(x=yr,y=FFmsy.bsm, col="red")
    lines(x=yr,y=lcl.FFmsy.bsm, col="red",lty="dotted")
    lines(x=yr,y=ucl.FFmsy.bsm, col="red",lty="dotted")
  }
  if(substr(id_file,1,3)=="Sim") points(x=yr[nyr],y=true.F_Fmsy,col="green", cex=3, lwd=2)

  # (f) Parabola plot ----
  #-------------------------
  max.y <- max(c(ct/MSY.cmsy,ifelse(FullSchaefer==T,max(ct/MSY.bsm),NA),1.2),na.rm=T)
  # plot parabola
  x=seq(from=0,to=2,by=0.001)
  y.c  <- ifelse(x>0.25,1,ifelse(x>0.125,4*x,exp(-10*(0.125-x))*4*x)) # correction for low recruitment below half and below quarter of Bmsy
  y=(4*x-(2*x)^2)*y.c
  plot(x=x, y=y, xlim=c(0,1), ylim=c(0,max.y), type="l", bty="l",xlab="",
       ylab="Catch / MSY", main="F: Equilibrium curve",  cex.main = 1.5, cex.lab = 1.55, cex.axis = 1.5)
  title(xlab= "Relative biomass B/k", line = 2.25, cex.lab = 1.55)

  # plot catch against CMSY estimates of relative biomass
  #><> HW add catch with error from JAGS
  lines(x=bk.cmsy[1:nyr], y=ct.cmsy/MSY.cmsy, pch=16, col="blue", lwd=1)
  points(x=bk.cmsy[1], y=ct.cmsy[1]/MSY.cmsy[1], pch=0, cex=2, col="blue")
  points(x=bk.cmsy[nyr], y=ct.cmsy[length(ct)]/MSY.cmsy[length(MSY.cmsy)],cex=2,pch=2,col="blue")

  # for CPUE, plot catch scaled by BSM MSY against observed biomass derived as q * CPUE scaled by BSM k
  if(FullSchaefer==T) {
    points(x=bt/(q.bsm*k.bsm), y=ct/MSY.bsm, pch=21,bg="grey")
    lines(x=bk.bsm[1:nyr], y=ct.bsm/MSY.bsm, pch=16, col="red",lwd=1)
    points(x=bk.bsm[1], y=ct.bsm[1]/MSY.bsm, pch=0, cex=2, col="red")
    points(x=bk.bsm[nyr], y=ct.bsm[length(ct)]/MSY.bsm[length(MSY.bsm)], pch=2, cex=2,col="red")
  }
  if(substr(id_file,1,3)=="Sim") points(x=true.Bk,y=ct[nyr]/true.MSY,col="green", cex=3, lwd=2)
  #analysis.plot <- recordPlot()

  #save analytic chart to JPEG file
  if (save.plots==TRUE) {
    jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_AN.jpg",sep="")
	if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
	   dev.copy(jpeg,jpgfile,
             width = 1024,
             height = 768,
             units = "px",
             pointsize = 18,
             quality = 95,
             res=80,
             antialias="cleartype")
    dev.off()
  }

  #---------------------------------------------
  # Plot Management-Graphs if desired ----
  #---------------------------------------------
  if(mgraphs==T) {
    # open window for plot of four panels
    if(grepl("win",tolower(Sys.info()['sysname']))) {windows(14,12)}
    par(mfrow=c(2,2))
    # make margins narrower
    par(mar=c(3.1,4.2,2.1,2.1))

    #---------------------
    # plot catch with MSY ----
    #---------------------
    max.y <- max(c(1.1*max(ct.jags),ucl.MSY),na.rm=T)
    plot(x=yr,rep(0,nyr),type="n",ylim=c(0,max.y), bty="l", main=paste("Catch",gsub(":","",gsub("/","-",stock))),
         xlab="",ylab="Catch (1000 tonnes/year)",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
    rect(yr[1],lcl.MSY,yr[nyr],ucl.MSY,col="lightgray", border=NA)
    lines(x=c(yr[1],yr[nyr]),y=c(MSY,MSY),lty="dashed", col="black", lwd=2)
    lines(x=yr, y=ct.jags, lwd=2) #
    text("MSY",x=end.yr-1.5, y=MSY+MSY*0.1, cex = .75)

    #----------------------------------------
    # Plot of estimated biomass relative to Bmsy
    #----------------------------------------
    # plot empty frame
    plot(yr, rep(0,nyr),type="n", ylim=c(0,max(c(2, max(ucl.B.Bmsy)))), ylab=expression(B/B[MSY]),xlab="", main="Stock size", bty="l",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
    # plot gray area of uncertainty in predicted biomass
    polygon(c(yr,rev(yr)), c(lcl.B.Bmsy,rev(ucl.B.Bmsy)),col="lightgray", border=NA)
    # plot median biomass
    lines(yr,B.Bmsy,lwd=2)
    # plot lines for Bmsy and 0.5 Bmsy
    lines(x=c(yr[1],yr[nyr]),y=c(1,1), lty="dashed", lwd=1.5)
    lines(x=c(yr[1],yr[nyr]),y=c(0.5,0.5), lty="dotted", lwd=1.5)

    # -------------------------------------
    ## Plot of exploitation rate
    # -------------------------------------
    # plot empty frame
    plot(yr, rep(0,nyr),type="n", ylim=c(0,max(c(2,ucl.F.Fmsy))),
         ylab=expression(F/F[MSY]),xlab="", main="Exploitation", bty="l",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
    # plot gray area of uncertainty in predicted exploitation
    polygon(c(yr,rev(yr)), c(lcl.F.Fmsy,rev(ucl.F.Fmsy)),col="lightgray", border=NA)
    # plot median exploitation rate
    lines(x=yr,y=F.Fmsy,lwd=2)
    # plot line for u.msy
    lines(x=c(yr[1],yr[nyr]),y=c(1,1), lty="dashed", lwd=1.5)

    # -------------------------------------
    ## plot stock-status graph
    # -------------------------------------

    if(FullSchaefer==T & force.cmsy==F) {
    x.F_Fmsy = all.FFmsy.bsm[,nyr]
    y.b_bmsy = all.BBmsy.bsm[,nyr]} else { # use CMSY data
    x.F_Fmsy = all.FFmsy.cmsy[,nyr]
    y.b_bmsy = all.BBmsy.cmsy[,nyr]
    }

    kernelF <- ci2d(x.F_Fmsy,y.b_bmsy,nbins=201,factor=2.2,ci.levels=c(0.50,0.80,0.75,0.90,0.95),show="none")
    c1 <- c(-1,100)
    c2 <- c(1,1)

    max.x1   <- max(c(2, max(kernelF$contours$"0.95"$x,F.Fmsy),na.rm =T))
    max.x    <- ifelse(max.x1 > 5,min(max(5,F.Fmsy*2),8),max.x1)
    max.y    <- max(max(2,quantile(y.b_bmsy,0.96)))

    plot(1000,1000,type="b", xlim=c(0,max.x), ylim=c(0,max.y),lty=3,xlab="",ylab=expression(B/B[MSY]), bty="l",  cex.main = 1.6, cex.lab = 1.35, cex.axis = 1.35)
    mtext(expression(F/F[MSY]),side=1, line=2.3, cex=1,adj=0.55)

    # extract interval information from ci2d object
    # and fill areas using the polygon function
    polygon(kernelF$contours$"0.95",lty=2,border=NA,col="cornsilk4")
    polygon(kernelF$contours$"0.8",border=NA,lty=2,col="grey")
    polygon(kernelF$contours$"0.5",border=NA,lty=2,col="cornsilk2")

    ## Add points and trajectory lines
    lines(c1,c2,lty=3,lwd=0.7)
    lines(c2,c1,lty=3,lwd=0.7)
    lines(F.Fmsy,B.Bmsy, lty=1,lwd=1.)

    # points(F.Fmsy,B.Bmsy,cex=0.8,pch=4)
    points(F.Fmsy[1],B.Bmsy[1],col=1,pch=22,bg="white",cex=1.5)
    points(F.Fmsy[which(yr==int.yr)],B.Bmsy[which(yr==int.yr)],col=1,pch=21,bg="white",cex=1.5)
    points(F.Fmsy[nyr],B.Bmsy[nyr],col=1,pch=24,bg="white",cex=1.5)

    ## Add legend
    legend('topright', inset = .03, c(paste(start.yr),paste(int.yr),paste(end.yr),"50% C.I.","80% C.I.","95% C.I."),
           lty=c(1,1,1,-1,-1,-1),pch=c(22,21,24,22,22,22),pt.bg=c(rep("white",3),"cornsilk2","grey","cornsilk4"),
           col=1,lwd=.8,cex=0.85,pt.cex=c(rep(1.1,3),1.5,1.5,1.5),bty="n",y.intersp = 1.1)
    #End of Biplot

  } # end of management graphs

  #management.plot <- recordPlot()

  # save management chart to JPEG file
  if (save.plots==TRUE & mgraphs==TRUE)  {
    jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_MAN.jpg",sep="")
	  if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
    dev.copy(jpeg,jpgfile,
             width = 1024,
             height = 768,
             units = "px",
             pointsize = 18,
             quality = 95,
             res=80,
             antialias="cleartype")
    dev.off()
  }

  #----------------------------------------------------------
  #><> Optional prior - posterior plots
  #---------------------------------------------------------
  if(pp.plot==T) {
    # open window for plot of four panels
    if(grepl("win",tolower(Sys.info()['sysname']))) {windows(17,12)}
    # make margins narrower
    par(mfrow=c(2,3),mar=c(4.5,4.5,2,0.5))
    greycol = c(grey(0.7,0.5),grey(0.3,0.5)) # changed 0.6 to 0.7

    # plot PP-diagnostics for CMSY
    # r
    rk <- exp(rmvnorm(n,mean=mu_rk,cov_rk))
    
    pp.lab = "r"
    rpr = sort(rk[,1])
    post = rs
    prior <-dlnorm(sort(rpr),meanlog = mean.log.r, sdlog = sd.log.r) #><>HW now pdf

    # generic ><>HW streamlined GP to check
    nmc = length(post)
    pdf = stats::density(post,adjust=2)
    plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
         yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)

    # k
    pp.lab = "k (1000 tonnes)"
    rpr = sort(rk[,2])
    post = ks
    prior <-dlnorm(sort(rpr),meanlog = mean.log.k, sdlog = sd.log.k) #><>HW now pdf
    # generic ><>HW streamlined GP to check
    nmc = length(post)
    pdf = stats::density(post,adjust=2)
    plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
         yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

    # Header
    mtext(paste0("CMSY prior & posterior distributions for ",stock),  side=3,cex=1.5)

    # MSY
    pp.lab = "MSY (1000 tonnes/year)"
    rpr = sort(rk[,1]*rk[,2]/4)
    post = rs*ks/4
    prior <-dlnorm(sort(rpr),meanlog = mean(log(rpr)), sdlog = sd(log(rpr))) #><>HW now pdf
    prand <- rlnorm(2000,meanlog = mean(log(rpr)), sdlog = sd(log(rpr)))
    # generic ><>HW streamlined GP to check
    nmc = length(post)
    pdf = stats::density(post,adjust=2)
    plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
         yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prand),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

    #><> bk beta priors
    bk.beta = (beta.prior(b.prior))

    # bk1
    pp.lab=paste0("B/k ",yr[1])
    post = all.bk.cmsy[,1]
    nmc = length(post)
    rpr = seq(0.5*startbio[1],startbio[2]*1.5,0.005)
    pdf = stats::density(post,adjust=2)
    prand <- sort(rbeta(2000,bk.beta[1,1], bk.beta[2,1]))
    prior <-dbeta(sort(prand),bk.beta[1,1], bk.beta[2,1]) #><>HW now pdf
    #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
    plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
    polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

    # bk2
    pp.lab=paste0("B/k ", int.yr)
    post = all.bk.cmsy[,which(int.yr==yr)]
    rpr = seq(0.5*intbio[1],intbio[2]*1.5,0.005)
    pdf = stats::density(post,adjust=2)
    prand <- sort(rbeta(2000,bk.beta[1,2], bk.beta[2,2]))
    prior <-dbeta(sort(prand),bk.beta[1,2], bk.beta[2,2]) #><>HW now pdf
    #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
    plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
    polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

    # bk3
    pp.lab=paste0("B/k ",yr[length(yr)])
    post = all.bk.cmsy[,length(yr)]
    rpr = seq(0.5*endbio[1],endbio[2]*1.5,0.005)
    pdf = stats::density(post,adjust=2)
    prand <- sort(rbeta(2000,bk.beta[1,3], bk.beta[2,3]))
    prior <-dbeta(sort(prand),bk.beta[1,3], bk.beta[2,3]) #><>HW now pdf
    #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
    plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,prand,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
    polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

     #save analytic chart to JPEG file
    if (save.plots==TRUE) {
     jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_PP_CMSY.jpg",sep="")
	   if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
      dev.copy(jpeg,jpgfile,
               width = 1024,
               height = 768,
               units = "px",
               pointsize = 18,
               quality = 95,
               res=80,
               antialias="cleartype")
      dev.off()
    }

    # plot PP diagnostics for BSM if available
    if(FullSchaefer==T & force.cmsy==F){ # BSM PLOT
    # open window for plot of four panels
    if(grepl("win",tolower(Sys.info()['sysname']))) {windows(17,12)}
    # make margins narrower
    par(mfrow=c(2,3),mar=c(4.5,4.5,2,0.5))
    greycol = c(grey(0.7,0.5),grey(0.3,0.5))

    # r
    rk <- exp(rmvnorm(n=5000,mean=mu_rk,cov_rk))
    
    pp.lab = "r"
    rpr = sort(rk[,1])
    post = rs.bsm
    prior <-dlnorm(sort(rpr),meanlog = mean.log.r, sdlog = sd.log.r) #><>HW now pdf

    # generic ><>HW streamlined GP to check
    nmc = length(post)
    pdf = stats::density(post,adjust=2)
    plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
         yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

    # k
    pp.lab = "k (1000 tonnes)"
    rpr = sort(rk[,2])
    post = ks.bsm
    prior <-dlnorm(sort(rpr),meanlog = mean.log.k, sdlog = sd.log.k) #><>HW now pdf
    # generic ><>HW streamlined GP to check
    nmc = length(post)
    pdf = stats::density(post,adjust=2)
    plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
         yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    
    # Header
    mtext(paste0("BSM prior & posterior distributions for ",stock),  side=3,cex=1.5)

    # MSY
    pp.lab = "MSY (1000 tonnes/year)"
    rpr = sort(rk[,1]*rk[,2]/4)
    post = rs.bsm*ks.bsm/4
    prior <-dlnorm(sort(rpr),meanlog = mean(log(rpr)), sdlog = sd(log(rpr))) #><>HW now pdf
    # generic ><>HW streamlined GP to check
    nmc = length(post)
    pdf = stats::density(post,adjust=2)
    plot(pdf,type="l",ylim=range(prior,pdf$y*1.1),xlim=range(c(pdf$x,rpr,max(pdf$x,rpr)*1.1)),
         yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    polygon(c((rpr),rev(prior)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prior)/mean(prior))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    

    # bk1
    pp.lab=paste0("B/k ",yr[1])
    post = all.bk.bsm[,1]
    nmc = length(post)
    rpr = seq(0.5*startbio[1],startbio[2]*1.5,0.005)
    pdf = stats::density(post,adjust=2)
    prand <- sort(rbeta(2000,bk.beta[1,1], bk.beta[2,1]))
    prior <-dbeta(sort(prand),bk.beta[1,1], bk.beta[2,1]) #><>HW now pdf
    #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
    plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
    polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
    legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
    
    # bk2
    pp.lab=paste0("B/k ", int.yr)
    post = all.bk.bsm[,which(int.yr==yr)]
    rpr = seq(0.5*intbio[1],intbio[2]*1.5,0.005)
    pdf = stats::density(post,adjust=2)
    prand <- sort(rbeta(2000,bk.beta[1,2], bk.beta[2,2]))
    prior <-dbeta(sort(prand),bk.beta[1,2], bk.beta[2,2]) #><>HW now pdf
    #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
    plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,0.3*rpr,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
    if(nbk>1) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    if(nbk==1) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),lty=2)
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    if(nbk>1){
    legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
      legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
      
    } else {
      legend('right',c("Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol[2],bty="n",cex=1.5)
    }

    # bk3
    pp.lab=paste0("B/k ",yr[length(yr)])
    post = all.bk.bsm[,length(yr)]
    nmc = length(post)
    rpr = seq(0.5*endbio[1],endbio[2]*1.5,0.005)
    pdf = stats::density(post,adjust=2)
    prand <- sort(rbeta(2000,bk.beta[1,3], bk.beta[2,3]))
    prior <-dbeta(sort(prand),bk.beta[1,3], bk.beta[2,3]) #><>HW now pdf
    #prior.height<-1/(prior[2]-prior[1])	# modification by GP 03/12/2019
    plot(pdf,type="l",ylim=range(c(pdf$y,0,prior)),xlim=range(c(pdf$x,prand,min(1.7*rpr[2],1.05),max(pdf$x,rpr)*1.1)),yaxt="n",xlab=pp.lab,ylab="",xaxs="i",yaxs="i",main="",bty="l",cex.lab = 1.55, cex.axis = 1.5)
    #rect(prior[1],0,prior[2],prior.height,col=greycol[1])
    if(nbk>2) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),col=greycol[1])
    if(nbk<3) polygon(c(prand,rev(prand)),c(prior,rep(0,length(sort(prior)))),lty=2)
    polygon(c(pdf$x,rev(pdf$x)),c(pdf$y,rep(0,length(pdf$y))),col=greycol[2])
    PPVR = round((sd(post)/mean(post))^2/(sd(prand)/mean(prand))^2,2)
    PPVM = round(mean(post)/mean(prior),2)
    pp = c(paste("PPVR =",PPVR))
    if(nbk>2){
      legend('right',c("Prior","Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol,bty="n",cex=1.5)
      legend("topright",pp,cex=1.3,bty="n",x.intersp = -0.5)
      
    } else {
      legend('right',c("Posterior"),pch=22,pt.cex=1.5,pt.bg = greycol[2],bty="n",cex=1.5)
    }

     #save analytic chart to JPEG file
     if (save.plots==TRUE) {
       jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_PP_BSM.jpg",sep="")
	   if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
       dev.copy(jpeg,jpgfile,
                width = 1024,
                height = 768,
                units = "px",
                pointsize = 18,
                quality = 95,
                res=80,
                antialias="cleartype")
       dev.off()
     }
    } # end of BSM plot
  } # End of posterior/prior plot

  #----------------------------------------------------------
  #><> Optional BSM diagnostic plot
  #---------------------------------------------------------
  if(BSMfits.plot==T & FullSchaefer==T & force.cmsy==F){
    #---------------------------------------------
    # open window for plot of four panels
    if(grepl("win",tolower(Sys.info()['sysname']))) {windows(9,6)}
    # make margins narrower
    par(mfrow=c(2,2),mar=c(3.1,4.1,2.1,2.1),cex=1)
    cord.x <- c(yr,rev(yr))
    # Observed vs Predicted Catch
    cord.y<-c(lcl.ct.jags,rev(ucl.ct.jags))
    plot(yr,ct,type="n",ylim=c(0,max(ct.jags,na.rm=T)),lty=1,lwd=1.3,xlab="Year",
         ylab=paste0("Catch (1000 tonnes)"),main=paste("Catch fit",stock),bty="l")
    polygon(cord.x,cord.y,col="gray",border=0,lty=1)
    lines(yr,ct.jags,lwd=2,col=1)
    points(yr,(ct),pch=21,bg="white",cex=1.)
    legend("topright",c("Observed","Predicted","95%CIs"),pch=c(21,-1,22),pt.cex = c(1,1,1.5),
           pt.bg=c("white",-1,"grey"),lwd=c(-1,2,-1),col=c(1,1,"grey"),bty="n",y.intersp = 0.9)

    # Observed vs Predicted CPUE
    cord.y<-c(lcl.cpue.bsm,rev(ucl.cpue.bsm))
    plot(yr,bt,type="n",ylim=c(0,max(c(pred.cpue,bt),na.rm=T)),lty=1,lwd=1.3,xlab="Year",ylab=paste0("cpue"),
         main="cpue fit",bty="l")
    polygon(cord.x,cord.y,col="gray",border=0,lty=1)
    lines(yr,cpue.bsm,lwd=2,col=1)
    points(yr,(bt),pch=21,bg="white",cex=1.)
    legend("topright",c("Observed","Predicted","95%CIs"),pch=c(21,-1,22),pt.cex = c(1,1,1.5),pt.bg=c("white",-1,"grey"),lwd=c(-1,2,-1),col=c(1,1,"grey"),bty="n",y.intersp = 0.9)

    # Process error log-biomass
    cord.y<-c(lcl.pe.bsm,rev(ucl.pe.bsm))
    plot(yr,rep(0,length(yr)),type="n",ylim=c(-max(c(abs(pred.pe),0.2),na.rm=T),max(c(abs(pred.pe),0.2),na.rm=T)),lty=1,lwd=1.3,xlab="Year",ylab=paste0("Deviation log(B)"),main="Process variation",bty="l")
    polygon(cord.x,cord.y,col="gray",border=0,lty=1)
    abline(h=0,lty=2)
    lines(yr,pe.bsm,lwd=2)


    #-------------------------------------------------
    # Function to do runs.test and 3 x sigma limits
    #------------------------------------------------
    runs.sig3 <- function(x,type="resid") {
      if(type=="resid"){mu = 0}else{mu = mean(x, na.rm = TRUE)}
      # Average moving range
      mr  <- abs(diff(x - mu))
      amr <- mean(mr, na.rm = TRUE)
      # Upper limit for moving ranges
      ulmr <- 3.267 * amr
      # Remove moving ranges greater than ulmr and recalculate amr, Nelson 1982
      mr  <- mr[mr < ulmr]
      amr <- mean(mr, na.rm = TRUE)
      # Calculate standard deviation, Montgomery, 6.33
      stdev <- amr / 1.128
      # Calculate control limits
      lcl <- mu - 3 * stdev
      ucl <- mu + 3 * stdev
      if(nlevels(factor(sign(x)))>1){
        runstest = snpar::runs.test(resid)
        pvalue = round(runstest$p.value,3)} else {
        pvalue = 0.001
      }

      return(list(sig3lim=c(lcl,ucl),p.runs= pvalue))
    }

    # get residuals
    resid = (log(bt)-log(cpue.bsm))[is.na(bt)==F]
    res.yr = yr[is.na(bt)==F]
    runstest = runs.sig3(resid)

    # CPUE Residuals with runs test
    plot(yr,rep(0,length(yr)),type="n",ylim=c(min(-0.25,runstest$sig3lim[1]*1.1),max(0.25,runstest$sig3lim[2]*1.1)),lty=1,lwd=1.3,xlab="Year",ylab=expression(log(cpue[obs])-log(cpue[pred])),main="Residual diagnostics",bty="l")
    abline(h=0,lty=2)
    RMSE = sqrt(mean(resid^2)) # Residual mean sqrt error
    if(RMSE>0.1){lims = runstest$sig3lim} else {lims=c(-1,1)}
    cols = c(rgb(1,0,0,0.5),rgb(0,1,0,0.5))[ifelse(runstest$p.runs<0.05,1,2)]
    if(RMSE>=0.1) rect(min(yr),lims[1],max(yr),lims[2],col=cols,border=cols) # only show runs if RMSE >= 0.1
    for(i in 1:length(resid)){
      lines(c(res.yr[i],res.yr[i]),c(0,resid[i]))
    }
    points(res.yr,resid,pch=21,bg=ifelse(resid < lims[1] | resid > lims[2],2,"white"),cex=1)

    # save management chart to JPEG file
    if (save.plots==TRUE & FullSchaefer == T & BSMfits.plot==TRUE) {
      jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_bsmfits.jpg",sep="")
	  if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
      dev.copy(jpeg,jpgfile,
               width = 1024,
               height = 768,
               units = "px",
               pointsize = 18,
               quality = 95,
               res=80,
               antialias="cleartype")
      dev.off()
    }
  }


  #-------------------------------------
  # HW Produce optional kobe plot
  #-------------------------------------

  if(kobe.plot==T){
    # open window for plot of four panels
    if(grepl("win",tolower(Sys.info()['sysname']))) {windows(7,7)}
    par(mfrow=c(1,1))
    # make margins narrower
    par(mar=c(5.1,5.1,2.1,2.1))

    if(FullSchaefer==T & force.cmsy==F) {
      x.F_Fmsy = all.FFmsy.bsm[,nyr]
      y.b_bmsy = all.BBmsy.bsm[,nyr]} else { # use CMSY data
        x.F_Fmsy = all.FFmsy.cmsy[,nyr]
        y.b_bmsy = all.BBmsy.cmsy[,nyr]
      }
    #><>HW better performance if FFmsy = x for larger values
    kernel.temp <- ci2d(x.F_Fmsy,y.b_bmsy,nbins=201,factor=2.2,ci.levels=c(0.50,0.80,0.75,0.90,0.95),show="none")
    kernelF = kernel.temp

    max.x1=max.y1   <- max(c(2, max(kernelF$contours$"0.95"$x,F.Fmsy),na.rm =T))
    max.y    <- ifelse(max.x1 > 5,min(max(5,F.Fmsy*2),8),max.x1)
    max.x    <- max(max(2,quantile(y.b_bmsy,0.96)))

    # -------------------------------------
    ## KOBE plot building
    # -------------------------------------
    #Create plot
    plot(1000,1000,type="b", xlim=c(0,max.x), ylim=c(0,max.y),lty=3,xlab="",ylab=expression(F/F[MSY]), bty="l",  cex.main = 2, cex.lab = 1.35, cex.axis = 1.35,xaxs = "i",yaxs="i")
    mtext(expression(B/B[MSY]),side=1, line=3, cex=1.3)
    c1 <- c(-1,100)
    c2 <- c(1,1)

    # extract interval information from ci2d object
    # and fill areas using the polygon function
    zb2 = c(0,1)
    zf2  = c(1,100)
    zb1 = c(1,100)
    zf1  = c(0,1)
    polygon(c(zb1,rev(zb1)),c(0,0,1,1),col="green",border=0)
    polygon(c(zb2,rev(zb2)),c(0,0,1,1),col="yellow",border=0)
    polygon(c(1,100,100,1),c(1,1,100,100),col="orange",border=0)
    polygon(c(0,1,1,0),c(1,1,100,100),col="red",border=0)

    polygon(kernelF$contours$"0.95"[,2:1],lty=2,border=NA,col="cornsilk4")
    polygon(kernelF$contours$"0.8"[,2:1],border=NA,lty=2,col="grey")
    polygon(kernelF$contours$"0.5"[,2:1],border=NA,lty=2,col="cornsilk2")
    points(B.Bmsy,F.Fmsy,pch=16,cex=1)
    lines(c1,c2,lty=3,lwd=0.7)
    lines(c2,c1,lty=3,lwd=0.7)
    lines(B.Bmsy,F.Fmsy, lty=1,lwd=1.)
    points(B.Bmsy[1],F.Fmsy[1],col=1,pch=22,bg="white",cex=1.5)
    points(B.Bmsy[which(yr==int.yr)],F.Fmsy[which(yr==int.yr)],col=1,pch=21,bg="white",cex=1.5)
    points(B.Bmsy[nyr],F.Fmsy[nyr],col=1,pch=24,bg="white",cex=1.5)
    # Get Propability
    Pr.green = sum(ifelse(y.b_bmsy>1 & x.F_Fmsy<1,1,0))/length(y.b_bmsy)*100
    Pr.red = sum(ifelse(y.b_bmsy<1 & x.F_Fmsy>1,1,0))/length(y.b_bmsy)*100
    Pr.yellow = sum(ifelse(y.b_bmsy<1 & x.F_Fmsy<1,1,0))/length(y.b_bmsy)*100
    Pr.orange = sum(ifelse(y.b_bmsy>1 & x.F_Fmsy>1,1,0))/length(y.b_bmsy)*100

    sel.years = c(yr[sel.yr])

    legend('topright',
           c(paste(start.yr),paste(int.yr),paste(end.yr),"50% C.I.","80% C.I.","95% C.I.",paste0(round(c(Pr.red,Pr.yellow,Pr.orange,Pr.green),1),"%")),
           lty=c(1,1,1,rep(-1,8)),pch=c(22,21,24,rep(22,8)),pt.bg=c(rep("white",3),"cornsilk2","grey","cornsilk4","red","yellow","orange","green"),
           col=1,lwd=1.1,cex=1.1,pt.cex=c(rep(1.3,3),rep(1.7,3),rep(2.2,4)),bty="n",y.intersp = 1.)

    if (save.plots==TRUE & kobe.plot==TRUE) {
      jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_KOBE.jpg",sep="")
	  if (retrosp.step>0) jpgfile<-gsub(".jpg", paste0("_retrostep_",retrosp.step,".jpg"), jpgfile) #modification added to save all steps in retrospective analysis
      dev.copy(jpeg,jpgfile,
               width = 1024*0.7,
               height = 1024*0.7,
               units = "px",
               pointsize = 18,
               quality = 95,
               res=80,
               antialias="cleartype")
      dev.off()
    }
  }

  #HW Kobe plot end
  #------------------------------------------------------
  # Write cmsy rdata oject (new ><>HW July 2021)
  #------------------------------------------------------
  if(write.rdata==TRUE){
    cmsy = list()
    cmsy$stock = stock
    cmsy$yr = yr
    cmsy$catch = ct
    cmsy$cmsy = list()
    cmsy$cmsy$timeseries = array(data=NA,dim=c(length(yr),3,2),dimnames = list(yr,c("mu","lci","uci"),c("BBmsy","FFmsy")))
    cmsy$cmsy$timeseries[,1,"BBmsy"] =  BBmsy.cmsy
    cmsy$cmsy$timeseries[,2,"BBmsy"] =  lcl.BBmsy.cmsy
    cmsy$cmsy$timeseries[,3,"BBmsy"] =  ucl.BBmsy.cmsy
    cmsy$cmsy$timeseries[,1,"FFmsy"] =  FFmsy.cmsy
    cmsy$cmsy$timeseries[,2,"FFmsy"] =  lcl.FFmsy.cmsy
    cmsy$cmsy$timeseries[,3,"FFmsy"] =  ucl.FFmsy.cmsy
    cmsy$cmsy$brp = t(data.frame(mu=c(r.cmsy,k.cmsy,MSY.cmsy,Bmsy.cmsy,Fmsy.cmsy),lci=c(lcl.r.cmsy,lcl.k.cmsy,lcl.MSY.cmsy,lcl.Bmsy.cmsy,lcl.Fmsy.cmsy),uci=c(ucl.r.cmsy,ucl.k.cmsy,ucl.MSY.cmsy,ucl.Bmsy.cmsy,ucl.Fmsy.cmsy)))
    colnames(cmsy$cmsy$brp) = c("r","k","MSY","Bmsy","Fmsy")
    cmsy$cmsy$rk = data.frame(r=rs,k=ks)
    cmsy$cmsy$kobe = data.frame(BBmsy=all.BBmsy.cmsy[,nyr],FFmsy=all.FFmsy.cmsy[,nyr])


    if(FullSchaefer==F){
      cmsy$bsm = NULL
    } else {
      cmsy$bsm = list()
      cmsy$bsm$timeseries = array(data=NA,dim=c(length(yr),3,2),dimnames = list(yr,c("mu","lci","uci"),c("BBmsy","FFmsy")))
      cmsy$bsm$timeseries[,1,"BBmsy"] =  BBmsy.bsm
      cmsy$bsm$timeseries[,2,"BBmsy"] =  lcl.BBmsy.bsm
      cmsy$bsm$timeseries[,3,"BBmsy"] =  ucl.BBmsy.bsm
      cmsy$bsm$timeseries[,1,"FFmsy"] =  FFmsy.bsm
      cmsy$bsm$timeseries[,2,"FFmsy"] =  lcl.FFmsy.bsm
      cmsy$bsm$timeseries[,3,"FFmsy"] =  ucl.FFmsy.bsm
      cmsy$bsm$brp = t(data.frame(mu=c(r.bsm,k.bsm,MSY.bsm,Bmsy.bsm,Fmsy.bsm),lci=c(lcl.r.bsm,lcl.k.bsm,lcl.MSY.bsm,lcl.Bmsy.bsm,lcl.Fmsy.bsm),uci=c(ucl.r.bsm,ucl.k.bsm,ucl.MSY.bsm,ucl.Bmsy.bsm,ucl.Fmsy.bsm)))
      colnames(cmsy$bsm$brp) = c("r","k","MSY","Bmsy","Fmsy")
      cmsy$bsm$rk = data.frame(r=rs.bsm,k=ks.bsm)
      cmsy$bsm$kobe = data.frame(BBmsy=all.BBmsy.bsm[,nyr],FFmsy=all.FFmsy.bsm[,nyr])
    } # end of Full Schaefer condition

    # save
    save(cmsy,file=paste0("cmsy_",stock,".rdata"))

  } #Write Rdata


  # -------------------------------------
  ## Write results into csv outfile
  # -------------------------------------
  if(write.output == TRUE && retrosp.step==0) { #account for retrospective analysis - write only the last result

    # fill catches from 1970 to 2020
    # if leading catches are missing, set them to zero; if trailing catches are missing, set them to NA
    ct.out     <- vector()
    F.Fmsy.out <- vector()
    bt.out     <- vector()

    j <- 1
    for(i in 1950 : 2030) {
      if(yr[1]>i) {
        ct.out[j]     <-0
        F.Fmsy.out[j] <-0
        bt.out[j]     <-2*Bmsy
      } else {
        if(i>yr[length(yr)]) {
          ct.out[j]     <-NA
          F.Fmsy.out[j] <-NA
          bt.out[j]     <-NA } else {
            ct.out[j]     <- ct.raw[yr==i]
            F.Fmsy.out[j] <- F.Fmsy[yr==i]
            bt.out[j]     <- B[yr==i]}
      }
      j=j+1
    }

    # write data into csv file
    output = data.frame(as.character(cinfo$Group[cinfo$Stock==stock]),
                        as.character(cinfo$Region[cinfo$Stock==stock]),
                        as.character(cinfo$Subregion[cinfo$Stock==stock]),
                        as.character(cinfo$Name[cinfo$Stock==stock]),
                        cinfo$ScientificName[cinfo$Stock==stock],
                        stock, start.yr, end.yr, start.yr.new, btype,length(bt[is.na(bt)==F]),
                        ifelse(FullSchaefer==T,yr[which(bt>0)[1]],NA),
                        ifelse(FullSchaefer==T,yr[max(which(bt>0))],NA),
                        ifelse(FullSchaefer==T,min(bt[is.na(bt)==F],na.rm=T),NA),
                        ifelse(FullSchaefer==T,max(bt[is.na(bt)==F],na.rm=T),NA),
                        ifelse(FullSchaefer==T,yr[which.min(bt)],NA),
                        ifelse(FullSchaefer==T,yr[which.max(bt)],NA),
                        endbio[1],endbio[2],
                        ifelse(FullSchaefer==T,q.prior[1],NA),
                        ifelse(FullSchaefer==T,q.prior[2],NA),
                        max(ct.raw),MSY.pr,mean(ct.raw[(nyr-4):nyr]),sd(ct.raw[(nyr-4):nyr]),ct.raw[nyr],
                        min(ct),max(ct),mean(ct),gm.prior.r,
                        ifelse(FullSchaefer==T,MSY.bsm,NA), # full Schaefer
                        ifelse(FullSchaefer==T,lcl.MSY.bsm,NA),
                        ifelse(FullSchaefer==T,ucl.MSY.bsm,NA),
                        ifelse(FullSchaefer==T,r.bsm,NA),
                        ifelse(FullSchaefer==T,lcl.r.bsm,NA),
                        ifelse(FullSchaefer==T,ucl.r.bsm,NA),
                        ifelse(FullSchaefer==T,log.r.var,NA),
                        ifelse(FullSchaefer==T,k.bsm,NA),
                        ifelse(FullSchaefer==T,lcl.k.bsm,NA),
                        ifelse(FullSchaefer==T,ucl.k.bsm,NA),
                        ifelse(FullSchaefer==T,log.k.var,NA),
                        ifelse(FullSchaefer==T,log.kr.cor,NA),
                        ifelse(FullSchaefer==T,log.kr.cov,NA),
                        ifelse(FullSchaefer==T, q.bsm,NA),
                        ifelse(FullSchaefer==T,lcl.q.bsm,NA),
                        ifelse(FullSchaefer==T,ucl.q.bsm,NA),
                        ifelse(FullSchaefer==T,bk.bsm[nyr],B.Bmsy[nyr]/2), # last B/k JAGS
                        ifelse(FullSchaefer==T,lcl.bk.bsm[nyr],NA),
                        ifelse(FullSchaefer==T,ucl.bk.bsm[nyr],NA),
                        ifelse(FullSchaefer==T,bk.bsm[1],B.Bmsy[1]/2), # first B/k JAGS
                        ifelse(FullSchaefer==T,lcl.bk.bsm[1],NA),
                        ifelse(FullSchaefer==T,ucl.bk.bsm[1],NA),
                        ifelse(FullSchaefer==T,bk.bsm[yr==int.yr],B.Bmsy[yr==int.yr]/2), # int year B/k JAGS
                        ifelse(FullSchaefer==T,lcl.bk.bsm[yr==int.yr],NA),
                        ifelse(FullSchaefer==T,ucl.bk.bsm[yr==int.yr],NA),
                        int.yr, # int year
                        ifelse(FullSchaefer==T,FFmsy.bsm[nyr],NA), # last F/Fmsy JAGS
                        r.cmsy, lcl.r.cmsy, ucl.r.cmsy, # CMSY r
                        k.cmsy, lcl.k.cmsy, ucl.k.cmsy, # CMSY k
                        MSY.cmsy, lcl.MSY.cmsy, ucl.MSY.cmsy, # CMSY MSY
                        bk.cmsy[nyr],lcl.bk.cmsy[nyr],ucl.bk.cmsy[nyr], # CMSY B/k in last year with catch data
                        bk.cmsy[1],lcl.bk.cmsy[1],ucl.bk.cmsy[1], # CMSY B/k in first year
                        bk.cmsy[yr==int.yr],lcl.bk.cmsy[yr==int.yr],ucl.bk.cmsy[yr==int.yr], # CMSY B/k in intermediate year
                        FFmsy.cmsy[nyr],lcl.FFmsy.cmsy[nyr],ucl.FFmsy.cmsy[nyr],
                        Fmsy,lcl.Fmsy,ucl.Fmsy,Fmsy.adj[nyr],lcl.Fmsy.adj[nyr],ucl.Fmsy.adj[nyr],
                        MSY,lcl.MSY,ucl.MSY,Bmsy,lcl.Bmsy,ucl.Bmsy,
                        B[nyr], lcl.B[nyr], ucl.B[nyr], B.Bmsy[nyr], lcl.B.Bmsy[nyr], ucl.B.Bmsy[nyr],
                        Ft[nyr], lcl.Ft[nyr], ucl.Ft[nyr], F.Fmsy[nyr], lcl.F.Fmsy[nyr], ucl.F.Fmsy[nyr],
                        ifelse(is.na(sel.yr)==F,B.sel,NA),
                        ifelse(is.na(sel.yr)==F,B.Bmsy.sel,NA),
                        ifelse(is.na(sel.yr)==F,F.sel,NA),
                        ifelse(is.na(sel.yr)==F,F.Fmsy.sel,NA),
                        ct.out[1],ct.out[2],ct.out[3],ct.out[4],ct.out[5],ct.out[6],ct.out[7],ct.out[8],ct.out[9],ct.out[10],          # 1950-1959
                        ct.out[11],ct.out[12],ct.out[13],ct.out[14],ct.out[15],ct.out[16],ct.out[17],ct.out[18],ct.out[19],ct.out[20], # 1960-1969
                        ct.out[21],ct.out[22],ct.out[23],ct.out[24],ct.out[25],ct.out[26],ct.out[27],ct.out[28],ct.out[29],ct.out[30], # 1970-1979
                        ct.out[31],ct.out[32],ct.out[33],ct.out[34],ct.out[35],ct.out[36],ct.out[37],ct.out[38],ct.out[39],ct.out[40], # 1980-1989
                        ct.out[41],ct.out[42],ct.out[43],ct.out[44],ct.out[45],ct.out[46],ct.out[47],ct.out[48],ct.out[49],ct.out[50], # 1990-1999
                        ct.out[51],ct.out[52],ct.out[53],ct.out[54],ct.out[55],ct.out[56],ct.out[57],ct.out[58],ct.out[59],ct.out[60], # 2000-2009
                        ct.out[61],ct.out[62],ct.out[63],ct.out[64],ct.out[65],ct.out[66],ct.out[67],ct.out[68],ct.out[69],ct.out[70], # 2010-2019
                        ct.out[71],ct.out[72],ct.out[73],ct.out[74],ct.out[75],ct.out[76],ct.out[77],ct.out[78],ct.out[79],ct.out[80],ct.out[81], # 2020-2030
                        F.Fmsy.out[1],F.Fmsy.out[2],F.Fmsy.out[3],F.Fmsy.out[4],F.Fmsy.out[5],F.Fmsy.out[6],F.Fmsy.out[7],F.Fmsy.out[8],F.Fmsy.out[9],F.Fmsy.out[10], # 1950-1959
                        F.Fmsy.out[11],F.Fmsy.out[12],F.Fmsy.out[13],F.Fmsy.out[14],F.Fmsy.out[15],F.Fmsy.out[16],F.Fmsy.out[17],F.Fmsy.out[18],F.Fmsy.out[19],F.Fmsy.out[20], # 1960-1969
                        F.Fmsy.out[21],F.Fmsy.out[22],F.Fmsy.out[23],F.Fmsy.out[24],F.Fmsy.out[25],F.Fmsy.out[26],F.Fmsy.out[27],F.Fmsy.out[28],F.Fmsy.out[29],F.Fmsy.out[30], # 1970-1979
                        F.Fmsy.out[31],F.Fmsy.out[32],F.Fmsy.out[33],F.Fmsy.out[34],F.Fmsy.out[35],F.Fmsy.out[36],F.Fmsy.out[37],F.Fmsy.out[38],F.Fmsy.out[39],F.Fmsy.out[40], # 1980-1989
                        F.Fmsy.out[41],F.Fmsy.out[42],F.Fmsy.out[43],F.Fmsy.out[44],F.Fmsy.out[45],F.Fmsy.out[46],F.Fmsy.out[47],F.Fmsy.out[48],F.Fmsy.out[49],F.Fmsy.out[50], # 1990-1999
                        F.Fmsy.out[51],F.Fmsy.out[52],F.Fmsy.out[53],F.Fmsy.out[54],F.Fmsy.out[55],F.Fmsy.out[56],F.Fmsy.out[57],F.Fmsy.out[58],F.Fmsy.out[59],F.Fmsy.out[60], # 2000-2009
                        F.Fmsy.out[61],F.Fmsy.out[62],F.Fmsy.out[63],F.Fmsy.out[64],F.Fmsy.out[65],F.Fmsy.out[66],F.Fmsy.out[67],F.Fmsy.out[68],F.Fmsy.out[69],F.Fmsy.out[70], # 2010-2019
                        F.Fmsy.out[71],F.Fmsy.out[72],F.Fmsy.out[73],F.Fmsy.out[74],F.Fmsy.out[75],F.Fmsy.out[76],F.Fmsy.out[77],F.Fmsy.out[78],F.Fmsy.out[79],F.Fmsy.out[80],F.Fmsy.out[81], # 2020-2030
                        bt.out[1],bt.out[2],bt.out[3],bt.out[4],bt.out[5],bt.out[6],bt.out[7],bt.out[8],bt.out[9],bt.out[10],           # 1950-1959
                        bt.out[11],bt.out[12],bt.out[13],bt.out[14],bt.out[15],bt.out[16],bt.out[17],bt.out[18],bt.out[19],bt.out[20],  # 1960-1969
                        bt.out[21],bt.out[22],bt.out[23],bt.out[24],bt.out[25],bt.out[26],bt.out[27],bt.out[28],bt.out[29],bt.out[30],  # 1970-1979
                        bt.out[31],bt.out[32],bt.out[33],bt.out[34],bt.out[35],bt.out[36],bt.out[37],bt.out[38],bt.out[39],bt.out[40],  # 1980-1989
                        bt.out[41],bt.out[42],bt.out[43],bt.out[44],bt.out[45],bt.out[46],bt.out[47],bt.out[48],bt.out[49],bt.out[50],  # 1990-1999
                        bt.out[51],bt.out[52],bt.out[53],bt.out[54],bt.out[55],bt.out[56],bt.out[57],bt.out[58],bt.out[59],bt.out[60],  # 2000-2009
                        bt.out[61],bt.out[62],bt.out[63],bt.out[64],bt.out[65],bt.out[66],bt.out[67],bt.out[68],bt.out[69],bt.out[70],  # 2010-2019
                        bt.out[71],bt.out[72],bt.out[73],bt.out[74],bt.out[75],bt.out[76],bt.out[77],bt.out[78],bt.out[79],bt.out[80],bt.out[81]) # 2020-2030

    write.table(output, file=outfile, append = T, sep = ",",
                dec = ".", row.names = FALSE, col.names = FALSE)
  }

  #----------------------------------------------------------------------------------
  # The code below creates a report in PDF format if write.pdf is TRUE ----
  #----------------------------------------------------------------------------------
  ## To generate reports in PDF format, install a LaTeX program. For Windows, you can use https://miktex.org/howto/install-miktex (restart after installation)
  ## Set write.pdf to 'TRUE' if you want pdf output.

  options(tinytex.verbose = TRUE)

  # Using MarkdownReports, this creates a markdown file for each stock then using rmarkdown to render each markdown file into a pdf file.
  if(write.pdf == TRUE) {
    library(knitr)
    library(tinytex)

    docTemplate <- "\\documentclass[12pt,a4paper]{article}
    \\setlength\\parindent{0pt}
    \\usepackage{geometry}
    \\usepackage{graphicx}
    \\usepackage{grffile}
    \\geometry{margin=0.5in}
    \\begin{document}

    \\section*{#TITLE#}


    #INTRO#

    \\begin{figure}[ht]
    \\centering
    \\includegraphics[width=1.00\\textwidth ext=.jpg type=jpg]{#IMAGE1#}
    \\end{figure}

    #MANAGEMENT#

    \\pagebreak

    \\begin{figure}[ht]
    \\centering
    \\includegraphics[width=1.00\\textwidth ext=.jpg type=jpg]{#IMAGE2#}
    \\end{figure}

    #ANALYSIS#

    \\end{document}"

    title = gsub(":","",gsub("/","-",cinfo$Name[cinfo$Stock==stock]))

    intro = (paste("Species: \\\\emph{",cinfo$ScientificName[cinfo$Stock==stock],"}, Stock code: ",
                   gsub(":","",gsub("/","-",stock)), sep=""))
    intro = (paste(intro,"\n\n","Region: ",gsub(":","",gsub("/","-",cinfo$Region[cinfo$Stock==stock])), sep=""))
    intro = (paste(intro,"\n\n","Marine Ecoregion: ",gsub(":","",gsub("/","-",cinfo$Subregion[cinfo$Stock==stock])), sep="" ))
    intro = (paste(intro,"\n\n","Reconstructed catch data used from years ", min(yr)," - ", max(yr),sep=""))
    intro = (paste(intro,"\n\n","For figure captions and method see http://www.seaaroundus.org/cmsy-method"))


    docTemplate<-gsub("#TITLE#", title, docTemplate)
    docTemplate<-gsub("#INTRO#", intro, docTemplate)


    management_text<-paste("\\\\textbf{Results for management (based on",ifelse(FullSchaefer==F | force.cmsy==T,"CMSY","BSM"),"analysis)}\\\\\\\\")
    management_text<-(paste(management_text,"\n\n","Fmsy = ",format(Fmsy, digits =3),", 95% CL = ",format(lcl.Fmsy, digits =3)," - ",format(ucl.Fmsy, digits =3)," (if B $>$ 1/2 Bmsy then Fmsy = 0.5 r)", sep=""))
    management_text<-(paste(management_text,"\n\n","Fmsy = ",format(Fmsy.adj[nyr], digits =3),", 95% CL = ",format(lcl.Fmsy.adj[nyr], digits =3)," - ",format(ucl.Fmsy.adj[nyr], digits =3)," (r and Fmsy are linearly reduced if B $<$ 1/2 Bmsy)",sep=""))
    management_text<-(paste(management_text,"\n\n","MSY = ",format(MSY, digits =3),",  95% CL = ",format(lcl.MSY, digits =3)," - ",format(ucl.MSY, digits =3),'; Bmsy = ',format(Bmsy, digits =3),",  95% CL = ",format(lcl.Bmsy, digits =3)," - ",format(ucl.Bmsy, digits =3)," (1000 tonnes)",sep=""))
    management_text<-(paste(management_text,"\n\n","Biomass in last year = ",format(B[nyr], digits =3),", 95% CL = ", format(lcl.B[nyr], digits =3), " - ",format(ucl.B[nyr], digits =3)," (1000 tonnes)",sep=""))
    management_text<-(paste(management_text,"\n\n","B/Bmsy in last year = " ,format(B.Bmsy[nyr], digits =3),", 95% CL = ", format(lcl.B.Bmsy[nyr], digits =3), " - ",format(ucl.B.Bmsy[nyr], digits =3),sep=""))
    management_text<-(paste(management_text,"\n\n","Fishing mortality in last year = ",format(Ft[nyr], digits =3),", 95% CL =", format(lcl.Ft[nyr], digits =3), " - ",format(ucl.Ft[nyr], digits =3),sep=""))
    management_text<-(paste(management_text,"\n\n","F/Fmsy  = ",format(F.Fmsy[nyr], digits =3),", 95% CL = ", format(lcl.F.Fmsy[nyr], digits =3), " - ",format(ucl.F.Fmsy[nyr], digits =3),sep=""))
    management_text<-(paste(management_text,"\n\n","Comment:", gsub(":","",gsub("/","",comment)), ""))
    docTemplate<-gsub("#MANAGEMENT#", management_text, docTemplate)

    analysis_text<-(paste("\\\\textbf{Results of CMSY analysis conducted in JAGS}\\\\\\\\",sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","r = ", format(r.cmsy, digits =3),", 95% CL = ", format(lcl.r.cmsy, digits =3), " - ", format(ucl.r.cmsy, digits =3),"; k = ", format(k.cmsy, digits =3),", 95% CL = ", format(lcl.k.cmsy, digits =3), " - ", format(ucl.k.cmsy, digits =3)," (1000 tonnes)",sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","MSY = ", format(MSY.cmsy, digits =3),", 95% CL = ", format(lcl.MSY.cmsy, digits =3), " - ", format(ucl.MSY.cmsy, digits =3)," (1000 tonnes/year)",sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Relative biomass last year = ", format(bk.cmsy[nyr], digits =3), " k, 95% CL = ", format(lcl.bk.cmsy[nyr], digits =3), " - ", format(ucl.bk.cmsy[nyr], digits =3),sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Exploitation F/(r/2) in last year = ", format((FFmsy.cmsy)[length(bk.cmsy)-1], digits =3),sep=""))

    if(FullSchaefer==T) {
      analysis_text <- paste(analysis_text,"\\\\\\\\")
      analysis_text<-(paste(analysis_text,"\n\n", "\\\\textbf{Results from Bayesian Schaefer model using catch and ",btype,"}\\\\\\\\",sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","r = ", format(r.bsm, digits =3),", 95% CL = ", format(lcl.r.bsm, digits =3), " - ", format(ucl.r.bsm, digits =3),"; k = ", format(k.bsm, digits =3),", 95% CL = ", format(lcl.k.bsm, digits =3), " - ", format(ucl.k.bsm, digits =3),sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","r-k log correlation = ", format(log.kr.cor, digits =3),sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","MSY = ", format(MSY.bsm, digits =3),", 95% CL = ", format(lcl.MSY.bsm, digits =3), " - ", format(ucl.MSY.bsm, digits =3)," (1000 tonnes/year)",sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","Relative biomass in last year = ", format(bk.cmsy[nyr], digits =3), " k, 95% CL = ",format(lcl.bk.cmsy[nyr], digits =3)," - ", format(ucl.bk.cmsy[nyr], digits =3),sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","Exploitation F/(r/2) in last year = ", format((ct.raw[nyr]/(bk.cmsy[nyr]*k.bsm))/(r.bsm/2), digits =3),sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","q = ", format(q.bsm, digits =3),", 95% CL = ", format(lcl.q.bsm, digits =3), " - ", format(ucl.q.bsm, digits =3),sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","Prior range of q = ",format(q.prior[1], digits =3)," - ",format(q.prior[2], digits =3),sep=""))
    }
    # show stock status and exploitation for optional selected year
    if(is.na(sel.yr)==F) {
      analysis_text<-(paste(analysis_text,"\n\n","Stock status and exploitation in ",sel.yr,sep=""))
      analysis_text<-(paste(analysis_text,"\n\n","Biomass = ",format(B.sel, digits =3), ", B/Bmsy = ",format(B.Bmsy.sel, digits =3),", fishing mortality F = ",format(F.sel, digits =3),", F/Fmsy = ",format(F.Fmsy.sel, digits =3),sep=""))
    }

    if(btype !="None" & length(bt[is.na(bt)==F])<nab) {
      analysis_text<-(paste(analysis_text,"\n\n","Less than ",nab," years with abundance data available, shown on second axis",sep="")) }


    analysis_text<-(paste(analysis_text,"\n\n","Relative abundance data type = ", format(btype, digits =3),sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Prior initial relative biomass = ", format(startbio[1], digits =3) , " - ", format(startbio[2], digits =3),ifelse(is.na(stb.low)==T," default"," expert"),sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Prior intermediate relative biomass = ", format(intbio[1], digits =3), " - ", format(intbio[2], digits =3), " in year ", int.yr,ifelse(is.na(intb.low)==T," default"," expert"),sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Prior final relative biomass = ", format(endbio[1], digits =3), " - ", format(endbio[2], digits =3),ifelse(is.na(endb.low)==T,", default"," expert"),sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Prior range for r = ", format(prior.r[1],digits=2), " - ", format(prior.r[2],digits=2),ifelse(is.na(r.low)==T," default"," expert"),", prior range for k = " , format(prior.k[1], digits =3), " - ", format(prior.k[2], digits =3)," (1000 tonnes) default",sep=""))
    analysis_text<-(paste(analysis_text,"\n\n","Source for relative biomass: \n\n",source,"",sep=""))

    docTemplate<-gsub("#ANALYSIS#", analysis_text, docTemplate)

    docTemplate<-gsub("_", "\\\\_", docTemplate)
    docTemplate<-gsub("%", "\\\\%", docTemplate)


    analysischartfile<-paste(gsub(":","",gsub("/","-",stock)),"_AN.jpg",sep="")
    managementchartfile<-paste(gsub(":","",gsub("/","-",stock)),"_MAN.jpg",sep="")
    docTemplate<-gsub("#IMAGE1#", managementchartfile, docTemplate)
    docTemplate<-gsub("#IMAGE2#", analysischartfile, docTemplate)

    # unique filenames to prevent error if files exists from previous run
    documentfile<-paste(gsub(":","",gsub("/","-",stock)),substr(as.character(Sys.time()),1,10),"-",sub(":","",substr(as.character(Sys.time()),12,16)),".RnW",sep="") # concatenated hours and minutes added to file name
    cat(docTemplate,file=documentfile,append=F)

    knit(documentfile)
    knitr::knit2pdf(documentfile)

    cat("PDF document is ",gsub(".RnW",".pdf",documentfile))

  }
  # end of loop to write text to file


  if(close.plots==T) graphics.off() # close on-screen graphics windows after files are saved

  FFmsy.retrospective[[retrosp.step+1]]<-F.Fmsy #retrospective analysis
  BBmsy.retrospective[[retrosp.step+1]]<-B.Bmsy #retrospective analysis
  years.retrospective[[retrosp.step+1]]<-yr #retrospective analysis

  } #retrospective analysis - end loop

	#retrospective analysis plots
	if (retros == T){

	   if(grepl("win",tolower(Sys.info()['sysname']))) {windows(14,7)}
		par(mfrow=c(1,2), mar=c(4,5,4,5),  oma=c(2,2,2,2))

	  allyears<-years.retrospective[[1]]
	  nyrtotal<-length(allyears)
	  legendyears<-c("All years")
	  #CHECK IF ALL YEARS HAVE BEEN COMPUTED
	  for (ll in 1:4){
	    if (ll>length(FFmsy.retrospective)){
	      FFmsy.retrospective[[ll]]<-c(0)
	      BBmsy.retrospective[[ll]]<-c(0)
	    }
	    else {
	      if(ll>1)
	        legendyears<-c(legendyears,allyears[nyrtotal-ll+1])
	    }
	  }

	  #PLOT FFMSY RETROSPECTIVE ANALYSIS
	  plot(x=allyears[1:nyrtotal],y=FFmsy.retrospective[[1]], main="",
	       ylim=c(0,max(max(FFmsy.retrospective[[1]],na.rm=T),
	               max(FFmsy.retrospective[[2]],na.rm=T),
	               max(FFmsy.retrospective[[3]],na.rm=T),
	               max(FFmsy.retrospective[[4]],na.rm=T))),
	       lwd=2, xlab="Year", ylab="F/Fmsy", type="l", bty="l",
	       cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5) #, xaxs="i",yaxs="i",xaxt="n",yaxt="n")
	  #PLOT ONLY THE TIME SERIES THAT ARE COMPLETE
	  if (length(FFmsy.retrospective[[2]])>1 || FFmsy.retrospective[[2]]!=0)
	    lines(x=allyears[1:(nyrtotal-1)],y=FFmsy.retrospective[[2]], type = "o", pch=15, col="red")
	  if (length(FFmsy.retrospective[[3]])>1 || FFmsy.retrospective[[3]]!=0)
	    lines(x=allyears[1:(nyrtotal-2)],y=FFmsy.retrospective[[3]], type = "o", pch=16, col="green")
	  if (length(FFmsy.retrospective[[4]])>1 || FFmsy.retrospective[[4]]!=0)
	    lines(x=allyears[1:(nyrtotal-3)],y=FFmsy.retrospective[[4]], type = "o", pch=17, col="blue")
	  legend("bottomleft", legend = legendyears,
	         col=c("black","red", "green", "blue"), lty=1, pch=c(-1,15,16,17))
	  #PLOT BBMSY RETROSPECTIVE ANALYSIS
	  plot(x=allyears[1:(nyrtotal)],y=BBmsy.retrospective[[1]],main="", ylim=c(0,max(max(BBmsy.retrospective[[1]],na.rm=T),
	                                                                                 max(BBmsy.retrospective[[2]],na.rm=T),
	                                                                                 max(BBmsy.retrospective[[3]],na.rm=T),
	                                                                                 max(BBmsy.retrospective[[4]],na.rm=T))),
	       lwd=2, xlab="Year", ylab="B/Bmsy", type="l", bty="l",cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5) #, xaxs="i",yaxs="i",xaxt="n",yaxt="n")
	  if (length(BBmsy.retrospective[[2]])>1 || BBmsy.retrospective[[2]]!=0)
	    lines(x=allyears[1:(nyrtotal-1)],y=BBmsy.retrospective[[2]], type = "o", pch=15, col="red")
	  if (length(BBmsy.retrospective[[3]])>1 || BBmsy.retrospective[[3]]!=0)
	    lines(x=allyears[1:(nyrtotal-2)],y=BBmsy.retrospective[[3]], type = "o", pch=16, col="green")
	  if (length(BBmsy.retrospective[[4]])>1 || BBmsy.retrospective[[4]]!=0)
	    lines(x=allyears[1:(nyrtotal-3)],y=BBmsy.retrospective[[4]], type = "o", pch=17, col="blue")
	  legend("bottomleft", legend = legendyears,
	         col=c("black","red", "green", "blue"), lty=1, pch=c(-1,15,16,17))

	  mtext(paste0("Retrospective analysis for ",stock),  outer = T , cex=1.5)

	  #save analytic chart to JPEG file
    if (save.plots==TRUE) {
      jpgfile<-paste(gsub(":","",gsub("/","-",stock)),"_RetrospectiveAnalysis.jpg",sep="")
      dev.copy(jpeg,jpgfile,
               width = 1024,
               height = 576,
               units = "px",
               pointsize = 10,
               quality = 95,
               res=80,
               antialias="default")
      dev.off()
    }

	  if(close.plots==T) graphics.off() # close on-screen graphics windows after files are saved
	} #retrospective analysis plots - end

} # end of stocks loop



