% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/errors.R
\name{.check_first_batch}
\alias{.check_first_batch}
\title{Check First Batch Validity}
\usage{
.check_first_batch(batch_sizes, ys)
}
\arguments{
\item{batch_sizes}{A numeric vector specifying batch sizes.}

\item{ys}{A matrix of counterfactual conditions.}
}
\value{
Returns NULL if the first batch size is valid; otherwise, throws an error.
}
\description{
This function checks if the first batch size is greater than or equal to the number of treatment arms.
}
