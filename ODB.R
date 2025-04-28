# install.packages('oaxaca')
library('oaxaca')
library('dplyr')

pums_data <- read.csv('pums_data_for_OBD1.csv')
pums_data_5cat <- read.csv('pums_data_for_OBD_5cat.csv')

# Filter columns
pums_data_for_obd <- pums_data %>%
  select(
    log_WAGP, race_ethnicity_sex,
    AGEP, AGE_SQUARED, ESR, FOD1P, INDP, NATIVITY, NOC, OCCP, SCHL, STATE, WKHP, WKWN
  )

# Filter for reference group and others
ref_group <- "White non-Hispanic Male"
comparison_groups <- unique(pums_data_for_obd$race_ethnicity_sex)
comparison_groups <- comparison_groups[comparison_groups != ref_group]

oaxaca_results <- list()
# Loop over each comparison group
for (target_group in comparison_groups) {

  cat("Running Oaxaca-Blinder for:", target_group, "\n")

  regression_df <- pums_data_for_obd %>%
    filter(race_ethnicity_sex %in% c(ref_group, target_group)) %>%
    mutate(group = ifelse(race_ethnicity_sex == ref_group, 0, 1))

  regression_df$FOD1P <- factor(regression_df$FOD1P, levels = levels(pums_data$FOD1P))
  regression_df$STATE <- factor(regression_df$STATE, levels = levels(pums_data$STATE))
  regression_df$OCCP  <- factor(regression_df$OCCP, levels = levels(pums_data$OCCP))
  regression_df$INDP  <- factor(regression_df$INDP, levels = levels(pums_data$INDP))
  regression_df$NATIVITY <- factor(regression_df$NATIVITY, levels = levels(pums_data$NATIVITY))
  regression_df$ESR <- factor(regression_df$ESR, levels = levels(pums_data$ESR))

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
      formula = log_WAGP ~ AGEP + AGE_SQUARED + ESR + FOD1P + INDP + NATIVITY + NOC + OCCP + SCHL + STATE + WKHP + WKWN | group,
      data = regression_df,
      R = 30
    )
  }, error = function(e) {
    cat("Failed for group:", target_group, "\n")
    return(NULL)
  })

  oaxaca_results[[target_group]] <- oaxaca_model
}
