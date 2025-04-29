# install.packages('oaxaca')
library('dplyr')
library('oaxaca')

# Define func to drop vars with no variance
drop_no_variance <- function(df) {
  keep_cols <- sapply(df, function(x) {
    if (is.numeric(x)) {
      return(var(x, na.rm = TRUE) != 0)  # keep numeric vars only if non-zero variance
    } else {
      return(TRUE)  # always keep factor/categorical variables!
    }
  })
  return(df[, keep_cols, drop = FALSE])
}


# ####################### ANALYSIS FOR 176-CATEGORY FOD1P #######################
# Load data
pums_data <- read.csv(unz("pums_data_for_OBKD.zip", "pums_data_for_OBKD.csv"),
                      colClasses = c(AGEP = "integer", AGE_SQUARED = "integer", ESR = "factor", FOD1P = "factor",
                                     FOD1P5 = "factor", INDP = "factor", log_WAGP = "numeric", NATIVITY = "factor",
                                     OCCP = "factor", race_ethnicity_sex = "factor", STATE = "factor", WKHP="numeric",
                                     WKWN = "numeric"),
                      # nrows = 100000 # for testing
                      )

factor_cols <- c("FOD1P", "STATE", "OCCP", "INDP", "NATIVITY", "ESR", "race_ethnicity_sex")

# Filter for reference group and others
ref_group <- "White non-Hispanic Male"
comparison_groups <- unique(pums_data$race_ethnicity_sex)
comparison_groups <- comparison_groups[comparison_groups != ref_group]

# Save original levels
original_levels <- lapply(pums_data[factor_cols], levels)

# Loop over each comparison group
oaxaca_results <- list()
for (target_group in comparison_groups) {

  cat("Running Oaxaca-Blinder for:", target_group, "\n")

  regression_df <- pums_data %>%
    filter(race_ethnicity_sex %in% c(ref_group, target_group)) %>%
    mutate(group = ifelse(race_ethnicity_sex == ref_group, 0, 1))

  cat("Sample size for", target_group, ":", nrow(regression_df), "\n")

  # Reset factor levels to original full dataset levels
  for (colname in factor_cols) {
    if (colname %in% names(regression_df)) {
      regression_df[[colname]] <- factor(regression_df[[colname]], levels = original_levels[[colname]])
    }
  }

  # Convert single-level factors to numeric constants
  for (colname in names(regression_df)) {
    if (is.factor(regression_df[[colname]])) {
      if (nlevels(droplevels(regression_df[[colname]])) <= 1) {
        regression_df[[colname]] <- as.numeric(regression_df[[colname]])
      }
    }
  }

  # Run Oaxaca decomposition
  oaxaca_model <- tryCatch({
    oaxaca(
      formula = log_WAGP ~ AGEP + AGE_SQUARED + ESR + FOD1P + INDP + NATIVITY + NOC + OCCP + STATE + WKHP + WKWN | group,
      data = regression_df,
      R = 2  # low for testing
    )
  }, error = function(e) {
    cat("Failed for group:", target_group, "\n")
    traceback()
    stop(e)
    return(NULL)
  })

  oaxaca_results[[target_group]] <- oaxaca_model
}

rm(regression_df, oaxaca_model)

#######################  ANALYSIS FOR 5-CATEGORY FOD1P5 #######################
