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
#  3. Distribution plot                                                        #
# --------------------------------------------------------------------------- #
# Density is misleading when n_incorrect = 9 vs n_correct = 2082.
# Box plot + jitter shows both groups clearly at their natural scales:
#   - correct: boxplot with light jitter (n too large for individual points)
#   - incorrect: individual points overlaid (all 9 visible)

df$Classification <- factor(
  ifelse(df$correct, "Correct", "Incorrect"),
  levels = c("Correct", "Incorrect")
)

df_correct   <- df[df$correct,   ]
df_incorrect <- df[!df$correct,  ]

n_label <- data.frame(
  Classification = factor(c("Correct", "Incorrect"),
                           levels = c("Correct", "Incorrect")),
  label          = c(sprintf("n = %d", nrow(df_correct)),
                     sprintf("n = %d",  nrow(df_incorrect))),
  y              = 0.02
)

p <- ggplot(df, aes(x = Classification, y = cal_confidence,
                    fill = Classification, colour = Classification)) +
  geom_boxplot(width = 0.45, outlier.shape = NA, alpha = 0.35,
               linewidth = 0.6) +
  geom_jitter(data    = df_correct,
              width   = 0.18, size = 0.35, alpha = 0.12, shape = 16) +
  geom_jitter(data    = df_incorrect,
              width   = 0.08, size = 2.8,  alpha = 0.85, shape = 21,
              colour  = "#D55E00", fill = "#D55E00") +
  geom_text(data  = n_label,
            aes(x = Classification, y = y, label = label),
            colour = "grey40", size = 3.2, inherit.aes = FALSE) +
  scale_fill_manual(values   = c("Correct" = "#0072B2",
                                 "Incorrect" = "#D55E00")) +
  scale_colour_manual(values = c("Correct" = "#0072B2",
                                 "Incorrect" = "#D55E00")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(0, 1.005)) +
  labs(x = NULL, y = "Calibrated confidence score") +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position  = "none"
  )

ggsave(file.path(out_dir, "confidence_distribution.pdf"),
       p, width = 4.5, height = 5, device = "pdf")
ggsave(file.path(out_dir, "confidence_distribution.png"),
       p, width = 4.5, height = 5, dpi = 300)
cat("  Saved: confidence_distribution.pdf / .png\n")

cat("\nDone. All outputs written to", out_dir, "\n")
