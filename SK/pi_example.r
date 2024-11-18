# Example of Monte Carlo approximation of Pi showed in the classes
runs <- 100000
# runif samples from a uniform distribution
xs <- runif(runs, min = -0.5, max = 0.5)
ys <- runif(runs, min = -0.5, max = 0.5)

in.circle <- xs^2 + ys^2 <= 0.5^2
mc.pi <- (sum(in.circle) / runs) * 4

plot(xs, ys, pch = '.', col = ifelse(in.circle, "blue", "grey"),
     xlab = '', ylab = '', asp = 1,
     main = paste("MC Approximation of Pi =", mc.pi))
