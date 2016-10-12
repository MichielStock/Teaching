#################################
#     PERCEPTRON LEARNING:      #
#       AN ILLUSTRATION         #
#################################

# AUTHOR: Michiel Stock
# mail: michielfmstock@gmail.com
# date: 12 October 2016

# A small illustration of perceptron learning


# make simple classification dataset

toy.data <- data.frame(X1=rnorm(40), X2=rnorm(40))
toy.data$y <- ifelse(toy.data$X1>2*toy.data$X2, -1, 1)

toy.data

plot(X2~X1, data=toy.data,
     col=ifelse(toy.data$y < 0, 'green', 'red'),
     pch=20)
# -1 > green
# 1 > red


beta <- rnorm(2)/5

abline(a=0, b=-beta[1]/beta[2], col='blue', lwd=3)
arrows(0, 0, beta[1], beta[2], col='blue', lwd=3)

update_beta <- function(beta){
  # plot
  plot(X2~X1, data=toy.data,
       col=ifelse(toy.data$y < 0, 'green', 'red'),
       pch=20)
  abline(a=0, b=-beta[1]/beta[2], col='blue', lwd=1, lty=2)
  arrows(0, 0, beta[1], beta[2], col='blue', lwd=1, lty=2)
  # get random instance
  i <- sample(1:40, 1)
  x <- toy.data[i, 1:2]
  y <- toy.data[i, 3]
  # highlight point
  points(x, col='pink', cex=4)
  arrows(0, 0, x[1,1], x[2,1], col='pink', lty=2)
  # calculate and highlight direction
  if(y * sum(x * beta) < 0){  # if misclassified
  update.dir <- as.numeric(x*y)
  arrows(0, 0, update.dir[1], update.dir[2], col='pink', lwd=2)
  # new parameter + show
  beta <- beta + update.dir
  # regularization
  # beta <- 0.7 * beta
  abline(a=0, b=-beta[1]/beta[2], col='blue', lwd=3)
  arrows(0, 0, beta[1], beta[2], col='blue', lwd=3)
  }
  return(beta)
}

beta <- update_beta(beta)

