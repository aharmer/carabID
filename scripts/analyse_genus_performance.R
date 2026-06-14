# analyse_genus_performance.R
#
# Analyses the relationship between per-genus training image count
# and cross-validation F1 score across 76 genera.
#
# Outputs (written to scripts/results/):
#   spearman_result.csv          -- rho, 95 % CI, p-value
#   bin_descriptive_stats.csv    -- mean/SD F1 per sample-size bin
#   f1_vs_imagecount_scatter.pdf/.png -- scatter + LOESS figure
#
# Run from the carabID/ root:
#   Rscript scripts/analyse_genus_performance.R
#
# Required packages: ggplot2, ggrepel, dplyr, scales

library(ggplot2)
library(ggrepel)
library(dplyr)
library(scales)

# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #

f1_path <- file.path(
  "runs", "classify",
  "carabid_cv_11ncls_ep30_autobatch_do02_lr001_cv_assessment",
  "cv_per_class_summary_test.csv"
)
counts_path <- file.path("app", "static", "class_counts.csv")
out_dir     <- file.path("scripts", "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------- #
#  Load and merge                                                              #
# --------------------------------------------------------------------------- #

f1_df <- read.csv(f1_path)

counts_df <- read.csv(counts_path) |>
  filter(Class.Name != "TOTAL") |>
  rename(class_name = Class.Name, n_images = Image.Count) |>
  select(class_name, n_images)

df <- left_join(f1_df, counts_df, by = "class_name")

cat(sprintf("Genera matched: %d\n", nrow(df)))
if (any(is.na(df$n_images))) {
  warning("Some genera did not match between files: ",
          paste(df$class_name[is.na(df$n_images)], collapse = ", "))
}

# --------------------------------------------------------------------------- #
#  1. Spearman correlation                                                     #
# --------------------------------------------------------------------------- #

sp <- cor.test(df$n_images, df$f1_mean, method = "spearman", exact = FALSE)

# Approximate 95 % CI via Fisher Z transformation
# (standard approximation; valid for n >= 10, ties acceptable with exact=FALSE)
z     <- atanh(sp$estimate)          # Fisher Z of rho
se_z  <- 1 / sqrt(nrow(df) - 3)
ci_lo <- tanh(z - 1.96 * se_z)
ci_hi <- tanh(z + 1.96 * se_z)

cat(sprintf(
  "\nSpearman correlation (training images vs F1):\n  rho = %.3f  (95%% CI: %.3f to %.3f)  p = %s  n = %d\n",
  sp$estimate, ci_lo, ci_hi,
  ifelse(sp$p.value < 0.001, "< 0.001", sprintf("%.3f", sp$p.value)),
  nrow(df)
))

cor_result <- data.frame(
  rho        = round(as.numeric(sp$estimate), 4),
  ci_lower   = round(ci_lo, 4),
  ci_upper   = round(ci_hi, 4),
  p_value    = signif(sp$p.value, 3),
  n_genera   = nrow(df)
)
write.csv(cor_result,
          file.path(out_dir, "spearman_result.csv"),
          row.names = FALSE)
cat("  Saved: spearman_result.csv\n")

# --------------------------------------------------------------------------- #
#  2. Scatter plot with LOESS smoother                                         #
# --------------------------------------------------------------------------- #

# Label genera with notably lower F1 (below 0.96) for context
label_df <- filter(df, f1_mean < 0.96)

caption_text <- sprintf(
  "Spearman rho = %.3f (95%% CI: %.3f-%.3f), %s, n = %d genera",
  as.numeric(sp$estimate), ci_lo, ci_hi,
  ifelse(sp$p.value < 0.001, "p < 0.001", sprintf("p = %.3f", sp$p.value)),
  nrow(df)
)

p_scatter <- ggplot(df, aes(x = log10(n_images), y = f1_mean)) +
  geom_smooth(method       = "loess",
              formula      = y ~ x,
              method.args  = list(degree = 1),
              se           = TRUE,
              colour       = "#E69F00",
              fill         = "#E69F00",
              alpha        = 0.15,
              linewidth    = 0.9) +
  geom_point(colour = "#0072B2", alpha = 0.75, size = 2.5) +
  geom_text_repel(
    data          = label_df,
    aes(label     = class_name),
    size          = 3,
    box.padding   = 0.4,
    point.padding = 0.2,
    max.overlaps  = 20,
    colour        = "grey30"
  ) +
  scale_x_continuous(
    breaks = log10(c(10, 20, 50, 100, 200)),
    labels = c("10", "20", "50", "100", "200")
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(NA, 1.005)) +
  labs(
    x       = "Training images per genus (log scale)",
    y       = "Mean F1 score (cross-validation test set)",
    caption = caption_text
  ) +
  theme_classic(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.caption     = element_text(size = 8, colour = "grey40")
  )

ggsave(file.path(out_dir, "f1_vs_imagecount_scatter.pdf"),
       p_scatter, width = 6, height = 5, device = "pdf")
ggsave(file.path(out_dir, "f1_vs_imagecount_scatter.png"),
       p_scatter, width = 6, height = 5, dpi = 300)
cat("  Saved: f1_vs_imagecount_scatter.pdf / .png\n")

# --------------------------------------------------------------------------- #
#  3. Descriptive statistics per sample-size bin                               #
# --------------------------------------------------------------------------- #

df <- df |>
  mutate(bin = case_when(
    n_images < 20  ~ "< 20",
    n_images <= 50 ~ "20–50",
    TRUE           ~ "> 50"
  )) |>
  mutate(bin = factor(bin, levels = c("< 20", "20–50", "> 50")))

bin_summary <- df |>
  group_by(bin) |>
  summarise(
    n_genera   = n(),
    images_min = min(n_images),
    images_max = max(n_images),
    mean_f1    = round(mean(f1_mean),   4),
    sd_f1      = round(sd(f1_mean),     4),
    median_f1  = round(median(f1_mean), 4),
    min_f1     = round(min(f1_mean),    4),
    max_f1     = round(max(f1_mean),    4),
    .groups = "drop"
  )

cat("\nDescriptive statistics per sample-size bin:\n")
print(as.data.frame(bin_summary), digits = 4)

write.csv(bin_summary,
          file.path(out_dir, "bin_descriptive_stats.csv"),
          row.names = FALSE)
cat("  Saved: bin_descriptive_stats.csv\n")

cat("\nDone. All outputs written to", out_dir, "\n")
