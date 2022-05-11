## Data-Vis Functions

#' Calculate sector x sector total impacts (single indicator) for purchases
#' Multiply each row of sector x sector A matrix by scalar elements of an
#' indicator (single) x sector array from N
#' @param model A complete EEIO model: a list with USEEIO model components and attributes
#' @param indicator str, index of a model indicator, e.g. "Greenhouse Gases".
#' @return A sector x sector, impact-per-tier-1-purchase matrix.
calculateTotalImpactbyTier1Purchases <- function(model, indicator) {
  A_impactPerPurchase <- model$N[indicator,] * model$A
  return(A_impactPerPurchase)
}


#' Cleaning {reshape, drop 0-values & nan's, append sector Name labels}


#' ggplot2 visualization function(s) w/ flexible params to allow for labeling

