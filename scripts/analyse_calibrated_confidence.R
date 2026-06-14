# analyse_calibrated_confidence.R
#
# Compares calibrated confidence scores between correct and incorrect
# classifications using:
#   1. Welch two-sample t-test (replaces raw-confidence version)
#   2. Linear mixed-effects model with genus as random intercept
#      (addresses reviewer request for species-level decomposition)
#
# Input:  runs/.../results_calibrated/per_prediction_val.csv
# Output: scripts/results/confidence_ttest.csv
#         scripts/results/confidence_lmm_summary.txt
#         scripts/results/confidence_distribution.pdf/.png
#
# Run from the carabID/ root:
#   Rscript scripts/analyse_calibrated_confidence.R
#
# Required packages: ggplot2, lme4, lmerTest, dplyr, scales

library(ggplot2)
library(lme4)
library(lmerTest)   # adds p-values to lmer summary via Satterthwaite df
library(dplyr)
library(scales)
library(patchwork)

# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #

pred_path <- file.path(
  "runs", "classify",
  "final_carabid_model_11ncls_ep30_autobatch_do02_lr001",
  "results_calibrated", "per_prediction_val.csv"
)
out_dir <- file.path("scripts", "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------- #
#  Load                                                                        #
# --------------------------------------------------------------------------- #

df <- read.csv(pred_path)
df$correct <- as.logical(df$correct)

cat(sprintf("Predictions loaded: %d total  (%d correct, %d incorrect)\n",
            nrow(df), sum(df$correct), sum(!df$correct)))

# --------------------------------------------------------------------------- #
#  1. Welch two-sample t-test (calibrated confidence)                          #
# --------------------------------------------------------------------------- #

tt <- t.test(cal_confidence ~ correct, data = df)

cat(sprintf(
  "\nWelch t-test (calibrated confidence, correct vs incorrect):\n  t(%s) = %.3f,  p %s\n  mean correct = %.4f,  mean incorrect = %.4f\n",
  format(round(tt$parameter, 1)),
  tt$statistic,
  ifelse(tt$p.value < 0.001, "< 0.001", sprintf("= %.3f", tt$p.value)),
  tt$estimate[2],   # correct = TRUE is second level
  tt$estimate[1]
))

ttest_result <- data.frame(
  group_incorrect = round(tt$estimate[1], 6),
  group_correct   = round(tt$estimate[2], 6),
  mean_difference = round(diff(tt$estimate), 6),
  t_statistic     = round(tt$statistic, 4),
  df              = round(tt$parameter, 2),
  p_value         = signif(tt$p.value, 3),
  ci_lower        = round(tt$conf.int[1], 6),
  ci_upper        = round(tt$conf.int[2], 6)
)
write.csv(ttest_result, file.path(out_dir, "confidence_ttest.csv"),
          row.names = FALSE)
cat("  Saved: confidence_ttest.csv\n")

# --------------------------------------------------------------------------- #
#  2. Linear mixed-effects model                                               #
# --------------------------------------------------------------------------- #
# Fixed effect: correct (TRUE/FALSE) — does classification outcome predict
#               calibrated confidence?
# Random intercept: true_class (genus) — accounts for non-independence of
#                   predictions within a genus and genus-level baseline
#                   differences in confidence.

m <- lmer(cal_confidence ~ correct + (1 | true_class), data = df,
          REML = TRUE)

sink(file.path(out_dir, "confidence_lmm_summary.txt"))
cat("Linear mixed-effects model: cal_confidence ~ correct + (1 | true_class)\n")
cat("Fitted with lme4/lmerTest; p-values via Satterthwaite approximation\n\n")
print(summary(m))
cat("\nRandom effects variance partition:\n")
print(as.data.frame(VarCorr(m)))
sink()

cat("\nLMM fixed effects:\n")
fe <- coef(summary(m))
print(fe)
cat("  (Full summary saved to confidence_lmm_summary.txt)\n")

# --------------------------------------------------------------------------- #
#  3. Distribution plot (two-panel)                                            #
# --------------------------------------------------------------------------- #
# Panel A (full scale): shows the stark contrast between groups at 0-100%.
#   Correct group appears as a dense band at the top; incorrect group shows
#   all 9 individual points spread across 30-75%.
# Panel B (zoomed): 90-100% range reveals the correct group's internal
#   distribution, which is invisible at the full scale.

df$Classification <- factor(
  ifelse(df$correct, "Correct", "Incorrect"),
  levels = c("Correct", "Incorrect")
)

df_correct   <- df[df$correct,  ]
df_incorrect <- df[!df$correct, ]

col_correct   <- "#0072B2"
col_incorrect <- "#D55E00"

shared_theme <- theme_classic(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        legend.position  = "none")

# Panel A — full scale, both groups
p_all <- ggplot(df, aes(x = Classification, y = cal_confidence,
                        fill = Classification, colour = Classification)) +
  geom_boxplot(width = 0.45, outlier.shape = NA, alpha = 0.35,
               linewidth = 0.6) +
  geom_jitter(data   = df_correct,
              width  = 0.16, size = 0.3, alpha = 0.10, shape = 16) +
  geom_jitter(data   = df_incorrect,
              width  = 0.08, size = 2.8, alpha = 0.85, shape = 21,
              colour = col_incorrect, fill = col_incorrect) +
  annotate("text", x = 1, y = 0.03,
           label = sprintf("n = %d", nrow(df_correct)),
           colour = "grey40", size = 3.2) +
  annotate("text", x = 2, y = 0.03,
           label = sprintf("n = %d", nrow(df_incorrect)),
           colour = "grey40", size = 3.2) +
  scale_fill_manual(values   = c("Correct" = col_correct,
                                 "Incorrect" = col_incorrect)) +
  scale_colour_manual(values = c("Correct" = col_correct,
                                 "Incorrect" = col_incorrect)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(0, 1.005)) +
  labs(x = NULL, y = "Calibrated confidence score", tag = "A") +
  shared_theme

# Panel B — correct group zoomed to 90-100%
p_zoom <- ggplot(df_correct, aes(x = "Correct", y = cal_confidence)) +
  geom_boxplot(width = 0.4, fill = col_correct, colour = col_correct,
               alpha = 0.35, outlier.shape = NA, linewidth = 0.6) +
  geom_jitter(width = 0.14, size = 0.9, alpha = 0.25,
              colour = col_correct, shape = 16) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1),
                     position = "right") +
  coord_cartesian(ylim = c(0.90, 1.005)) +
  labs(x = NULL, y = "Calibrated confidence (zoomed)", tag = "B") +
  shared_theme

p_combined <- p_all + p_zoom + plot_layout(widths = c(2, 1.5))

ggsave(file.path(out_dir, "confidence_distribution.pdf"),
       p_combined, width = 7.5, height = 5, device = "pdf")
ggsave(file.path(out_dir, "confidence_distribution.png"),
       p_combined, width = 7.5, height = 5, dpi = 300)
cat("  Saved: confidence_distribution.pdf / .png\n")

cat("\nDone. All outputs written to", out_dir, "\n")
