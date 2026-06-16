# plot_reliability_diagram.R
#
# Produces a reliability (calibration) diagram comparing raw and
# temperature-scaled confidence scores.
#
# Input:  runs/.../results_calibrated/per_prediction_<split>.csv
# Output: scripts/results/reliability_diagram.pdf/.png
#
# Run from the carabID root:
#   Rscript scripts/plot_reliability_diagram.R

library(ggplot2)
library(dplyr)
library(scales)

# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #

pred_path <- file.path(
  "runs", "classify",
  "final_carabid_v2_11ncls_ep30_do02_lr001",
  "results_calibrated", "per_prediction_test.csv"
)
out_dir <- file.path("scripts", "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------- #
#  Load                                                                        #
# --------------------------------------------------------------------------- #

df <- read.csv(pred_path)
df$correct <- as.logical(df$correct)
cat(sprintf("Loaded %d predictions (%d correct, %d incorrect)\n",
            nrow(df), sum(df$correct), sum(!df$correct)))

# --------------------------------------------------------------------------- #
#  Reliability curve via quantile binning (matches sklearn calibration_curve)  #
# --------------------------------------------------------------------------- #
# For each confidence column, divide predictions into n_bins equal-frequency
# bins, then compute mean confidence and fraction correct per bin.

reliability_bins <- function(df, conf_col, series_label, n_bins = 10) {
  df %>%
    arrange(.data[[conf_col]]) %>%
    mutate(bin = ntile(.data[[conf_col]], n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_conf    = mean(.data[[conf_col]]),
      frac_correct = mean(correct),
      n            = n(),
      .groups      = "drop"
    ) %>%
    mutate(series = series_label)
}

plot_df <- bind_rows(
  reliability_bins(df, "raw_confidence", "Before calibration"),
  reliability_bins(df, "cal_confidence", "After calibration")
) %>%
  mutate(series = factor(series,
                         levels = c("Before calibration", "After calibration")))

# --------------------------------------------------------------------------- #
#  Plot                                                                        #
# --------------------------------------------------------------------------- #

col_before <- "#E69F00"
col_after  <- "#0072B2"

p <- ggplot(plot_df, aes(x = mean_conf, y = frac_correct,
                         colour = series, group = series)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey60", linewidth = 0.6) +
  geom_line(linewidth = 0.8) +
  geom_point(shape = 19, size = 2.5) +
  scale_colour_manual(
    values = c("Before calibration" = col_before,
               "After calibration"  = col_after)
  ) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0.5, 1.0, by = 0.1)
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0.5, 1.0, by = 0.1)
  ) +
  coord_cartesian(xlim = c(0.5, 1.0), ylim = c(0.5, 1.0)) +
  labs(
    x      = "Mean predicted confidence",
    y      = "Fraction correct",
    colour = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    legend.position = "inside",
    legend.position.inside  = c(0.85, 0.12),
    legend.background = element_rect(fill = "white", colour = NA),
    panel.grid.minor  = element_blank()
  )

ggsave(file.path(out_dir, "reliability_diagram.pdf"),
       p, width = 5, height = 5, device = "pdf")
ggsave(file.path(out_dir, "reliability_diagram.png"),
       p, width = 5, height = 5, dpi = 300)
cat("Saved: reliability_diagram.pdf / .png\n")
