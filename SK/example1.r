# Calculate the area of a complex shape using Monte Carlo method
calculate_area <- function(num_points) {
    # Generate random points within the bounding box
    x <- runif(num_points, min = 0, max = 8)  # x-coordinates from 0 to 8
    y <- runif(num_points, min = 0, max = 4)  # y-coordinates from 0 to 4

    # Each condition represents a different region of the shape
    points_inside <- (
        (x <= 1 & y <= x) |
        (x > 1 & x <= 2 & y <= 2 * x - 1) |
        (x > 2 & x <= 3 & y <= x + 1) |
        (x > 3 & x <= 5 & y <= 4) |
        (x > 5 & x <= 6 & y <= -2 * (x - 5) + 4) |
        (x > 6 & x <= 7 & y <= 2) |
        (x > 7 & x <= 8 & y <= -2 * (x - 7) + 2)
    )

    # Calculate area based on proportion of points inside the shape
    estimated_area <- (sum(points_inside) / num_points) * 32

    # Create visualization plot
    plot(x, y,
         pch = ".",
         col = ifelse(points_inside, 'blue', 'grey'),
         main = "Monte Carlo Area Estimation",
         xlab = "X coordinate",
         ylab = "Y coordinate")

    return(estimated_area)
}

# Calculate ideal area of the shape
get_ideal_area <- function() {
    triangle_1 <- 1 * 1 * 0.5
    region_2 <- 2 * 1 * 0.5 + 1
    region_3 <- 1 * 1 * 0.5 + 3
    rectangle_4 <- 4 * 2
    region_5 <- 2 * 1 * 0.5 + 2
    rectangle_6 <- 2 * 1
    region_7 <- 2 * 1 * 0.5

    return(triangle_1 + region_2 + region_3 + rectangle_4 +
           region_5 + rectangle_6 + region_7)
}

NUM_SIMULATIONS <- 10
POINTS_PER_SIM <- 100000

# Open a graphical device
quartz()

cat("Running", NUM_SIMULATIONS, "simulations with",
    POINTS_PER_SIM, "points each...\n")
areas <- replicate(NUM_SIMULATIONS, calculate_area(POINTS_PER_SIM))
ideal_area <- get_ideal_area()
errors <- abs(theoretical_area - areas)

cat("\nResults:\n")
cat("Areas in each run:\n")
print(round(areas, 4))

cat("\nErrors in each run:\n")
print(round(errors, 4))

mean_error <- mean(errors)
cat("\nMean error:", round(mean_error, 4))

mean_areas <- mean(areas)
cat("\nMean estimated area:", round(mean_areas, 4))
cat("\nTheoretical area:", round(theoretical_area, 4))
cat("\nFinal accuracy: ",
    round((1 - abs(mean_areas - theoretical_area)/theoretical_area) * 100, 2),
    "%\n", sep="")