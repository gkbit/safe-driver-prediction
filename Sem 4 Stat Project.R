###Statistics Project

##----Importing the dataset----##

x=read.csv("porto-seguro-safe-driver-prediction\\train\\train.csv",header=T)		#--- training dataset
z=read.csv("porto-seguro-safe-driver-prediction\\test\\test.csv",header=T)		#--- test dataset


##---Removing observations with missing values from the datasets---##

missing=function(vec)		#--- To determine whether vec has a value -1
{
	return(-1 %in% vec)
}

miss_train=apply(x,1,missing)
miss_test=apply(z,1,missing)
omit=which(miss_train==T)
omit2=which(miss_test==T)

length(omit)		#--- Number of observations to be discarded from the training set
length(omit2)		#--- Number of observations to be discarded from the test set

x_mod=x[-omit,]		#--- training set after discarding observations with missing values
z_mod=z[-omit2,]		#--- test set after discarding observations with missing values

##---Modifying the training and test sets---##

set.seed(1)
N=nrow(x_mod)
train=sample(N,floor(0.6*N))
train_set=x_mod[train,-1]		#---excluding the covariate "id" as it corresponds to the identity of an individual, and hence, is not necesssary for model fitting---#
test_set=x_mod[-train,-1]		#---excluding the covariate "id" as it corresponds to the identity of an individual, and hence, is not necesssary for model fitting---#


y=train_set[,1]			#---training response
X=train_set[,-1]			#---training covariates
Z=test_set				#---test covariates

iden=function(mat,sep)		#--- to identify which columns of 'mat' are categorical covariates
{
	vec=c()
	for(j in 1:ncol(mat))
	{
		char=unlist(strsplit(colnames(mat)[j],split=sep))
		if("cat" %in% char||"bin" %in% char)
		{
			vec=c(vec,j)
		}
	}
	return(vec)
}
v=iden(X, sep="_")	#--- identifying the training categorical covariates
u=iden(Z, sep="_")	#--- identifying the test categorical covariates
D=X[,v]			#--- matrix consisting of only categorical covariates as columns
E=Z[,u]

for(j in 1:ncol(D))
{
	D[,j]=as.factor(D[,j])		#--- converting categorical column to factor
}

for(j in 1:ncol(E))
{
	E[,j]=as.factor(E[,j])
}

X1=cbind2(X[,-v],D)
X11=cbind2(y,X1)
colnames(X11)[1]="target"
Z1=cbind2(Z[,-u],E)


##--- Choosing Best Variables ---##




#---Method 1: Using LASSO and Ridge --#

require(glmnet)
nlam = 100				
lam = seq(0,100,length=nlam)
lam2=lam/(2*length(y))
fitg=glmnet(data.matrix(X1),as.factor(y),family="binomial",lambda=lam2)
plot(fitg,xvar="lambda",label=T)
aa2=as.matrix(coef(fitg))
aa2=t(aa2)
aa2=aa2[,-1]
aa2=aa2[nrow(aa2):1,]
matplot(fitg$lambda,aa2,type="l",lty=1,
xlab=expression(paste(lambda)),
ylab="Coefficients",main="Lasso paths")
abline(h=0,lwd=4,lty=2)
names=colnames(Z[,-1])
text(rep(0,ncol(Z[,-1])),aa2[1,],names)

mis=function(l)
{
	fit.lasso=glmnet(data.matrix(X1),as.factor(y),family="binomial",lambda=l)
	pred.lasso=predict(fit.lasso,newx=data.matrix(Z[,-1]),type="class")
	mis=mean(as.numeric(pred.lasso)!=Z[,1])
	return(mis)
}
m=sapply(lam,mis)
plot(lam,m,main="Misclassification Rate",xlab=expression(lambda),ylab='Misclassification Rate',type="l")
curve(mis,from=min(lam2),to=max(lam2),n=100)
plot(lam2,mis,main="Misclassification Error",type="l")
fit.lasso=glmnet(data.matrix(X1),as.factor(y),family="binomial")
plot(fit.lasso,xvar="lambda")

cv.ridge=cv.glmnet(data.matrix(X1),as.vector(y),nfolds=10,family="binomial",alpha=0,type.measure="deviance")
plot(cv.ridge)
abline(h=cv.ridge$cvup[which.min(cv.ridge$cvm)])
coef(fit)
cv.ridge$lambda.min
cv.ridge$lambda.1se
coef(cv.ridge,s="lambda.min")
coef(cv.ridge,s="lambda.1se")

cv.lasso=cv.glmnet(data.matrix(X1),y,family="binomial",type.measure="deviance")
plot(cv.lasso)
abline(h=cv.lasso$cvup[which.min(cv.lasso$cvm)])
coef(fit)
cv.lasso$lambda.min
cv.lasso$lambda.1se
coef(cv.lasso,s="lambda.min")
coef(cv.lasso,s="lambda.1se")

cv.lasso_mis=cv.glmnet(data.matrix(X1),y,family="binomial",alpha=1,type.measure="class")
plot(cv.lasso_mis)


cv.elastic=cv.glmnet(data.matrix(X1),as.factor(y),family="binomial",alpha=0.88,type.measure="deviance")
plot(cv.elastic)

cv.elastic_mis=cv.glmnet(data.matrix(X1),y,nfolds=10,family="binomial",alpha=0.5,type.measure="class")





#---Fitting GLM based on the best covariates---#

fitlasso=glmnet(data.matrix(X1),y,lambda=cv.lasso$lambda.1se,family="binomial")
fitlasso_mis=glmnet(data.matrix(X1),y,lambda=cv.lasso_mis$lambda.1se,family="binomial")
fitelastic=glmnet(data.matrix(X1),as.factor(y),lambda=cv.elastic$lambda.1se,family="binomial",alpha=0.88)
fitelastic_mis=glmnet(data.matrix(X1),y,lambda=cv.elastic_mis$lambda.1se,family="binomial",alpha=0.5)

cverror.lasso=cv.lasso$cvm[which(cv.lasso$lambda==cv.lasso$lambda.1se)]
cverror.elastic=cv.elastic$cvm[which(cv.elastic$lambda==cv.elastic$lambda.1se)]
cverror.lasso_mis=cv.lasso_mis$cvm[which(cv.lasso$lambda==cv.lasso$lambda.1se)]
cverror.elastic_mis=cverror.elastic_mis=cv.elastic_mis$cvm[which(cv.elastic$lambda==cv.elastic$lambda.1se)]



p1=predict(cv.lasso_mis,newx=data.matrix(Z1),s="lambda.1se",type="response")
p2=predict(cv.elastic_mis,newx=data.matrix(X1),s="lambda.1se")
p3=predict(fitelastic,newx=data.matrix(Z[,-1]),type="class")
p4=predict(fitelastic,newx=data.matrix(Z[,-1]),type="response")
mean(as.numeric(as.vector(p3))!=Z[,1])
table(p3,Z[,1])


#---Predicting based on the fitted model with glmnet---#


pred=predict(cv.lasso,newx=data.matrix(Z1[,-1]), type="class",s="lambda.1se")
mean(as.numeric(pred)!=Z[,1])
table(pred,Z[,1])

pred_prob=predict(cv.lasso,newx=data.matrix(Z1[,-1]), type="response",s="lambda.1se")
str(pred_prob)

fit=glm(target~.,data=train_set)
yhat=predict(fit)
length(yhat)==length(y)
for(i in 1:length(yhat))
{
	yhat[i]=ifelse(yhat[i]>=0.5,1,0)
}
yhat
sum(yhat==0)==length(yhat)
which(yhat==1)


#Implementing imputation by mean
impute=function(x)
{
	n=length(x)
	if(any(x==0) & any(x==1))
	{
		z=ifelse(mean(x[x!=-1])>=0.5,1,0)
		for(i in 1:n)
		{
			if(x[i]==-1)
			{
				x[i]=z
			}
		}
	}
	return(x)
}

for(i in 1:nrow(X))
{
	impute(X[i,])
}

#- Using LDA -#
library(MASS)
g=lda(target~ps_car_13+ps_ind_05_cat+ps_ind_17_bin,data=X11)
plot(g)
fitl=predict(g,newdata=Z1[,-1])
mean(as.numeric(as.vector(fitl$class))!=Z[,1])
table(fitl$class,Z[,1])
fitl$posterior[,2]

gfull=lda(target~.,data=X11)
plot(gfull)
fitful=predict(gfull,newdata=Z1[,-1])
mean(as.numeric(as.vector(fitful$class))!=Z[,1])
table(fitful$class,Z[,1])

#-- Using Classification Trees --#

library(rpart)
library(rpart.plot)
fit=rpart(target~.,data=X11,method="class",cp=0)
rpart.plot(fit)
cpseq=fit$cptable[,1]
error=function(cp)
{
	fitt=rpart(target~.,data=X11,method="class",cp=cp)
	fitt_pred=predict(fitt,newdata=Z1[,-1],type="class")
	mis=mean(as.numeric(as.vector(fitt_pred))!=Z[,1])
	return(mis)
}
cl=sapply(cpseq,error)
fit_optimum=rpart(target~.,data=X11,method="class",cp=cpseq[2])
rpart.plot(fit_optimum)
rpart.plot(fit)
printcp(fit)
plot(fit)
plotcp(fit)
fit_opt=rpart(target~.,data=X11,method="class",cp=3.5201*10^(-4))
rpart.plot(fit_opt)
tree_pred=predict(fit_optimum,newdata=Z1[,-1], type="prob")
tree_pred2=predict(fit_optimum,newdata=Z1[,-1], type="class")
table(tree_pred2,Z[,1])
mean(as.numeric(as.vector(tree_pred2))!=Z[,1])
plot(fit_opt)


#-- Using random forest --#  (Not successful due to large no. of levels)
library(randomForest)
rf=randomForest(X1,as.factor(y),xtest=Z1[,-1],ytest=Z[,1])
