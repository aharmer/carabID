# plot_reliability_diagram.R
#
# Reads the reliability diagram data produced by compute_calibration_metrics.py
# and plots a publication-quality reliability diagram with ggplot2.
#
# Output: calibrated/results/reliability_diagram.pdf  (.png)
#
# Usage (from carabID/ root):
#   Rscript calibrated/plot_reliability_diagram.R

library(ggplot2)
library(patchwork)

# ---- paths ----------------------------------------------------------------
data_file <- "calibrated/results/reliability_diagram_data.csv"
out_dir   <- "calibrated/results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(data_file)

# ---- axis labels ----------------------------------------------------------
# x-axis uses midpoints; label with bin edges for clarity
x_breaks <- unique(df$bin_midpoint)
x_labels <- sprintf("%.2f", x_breaks)

# ---- colour palette (accessible) -----------------------------------------
pal <- c("Uncalibrated" = "#E69F00", "Calibrated" = "#0072B2")

# ---- build plot -----------------------------------------------------------
# Panel A: reliability diagram (accuracy vs mean confidence per bin)
p_rel <- ggplot(df, aes(x = mean_confidence, y = accuracy,
                         colour = condition, shape = condition)) +
  # perfect calibration line
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  geom_point(size = 3, alpha = 0.9) +
  geom_line(aes(group = condition), linewidth = 0.8, alpha = 0.8) +
  scale_colour_manual(values = pal, name = NULL) +
  scale_shape_manual(values = c("Uncalibrated" = 16, "Calibrated" = 17),
                     name = NULL) +
  scale_x_continuous(limits = c(0, 1),
                     labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(limits = c(0, 1),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Mean confidence (per bin)",
       y = "Accuracy (per bin)",
       title = "Reliability diagram") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank())

# Panel B: sample fraction per bin (histogram-style)
p_hist <- ggplot(df, aes(x = bin_midpoint, y = fraction,
                          fill = condition)) +
  geom_col(position = position_dodge(width = 0.05),
           width = 0.04, alpha = 0.75) +
  scale_fill_manual(values = pal, name = NULL) +
  scale_x_continuous(limits = c(0, 1),
                     labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Confidence bin",
       y = "Fraction of samples",
       title = "Sample distribution") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank())

# ---- combine with patchwork ----------------------------------------------
combined <- p_rel / p_hist +
  plot_layout(heights = c(2, 1)) +
  plot_annotation(
    caption = paste0(
      "Reliability diagram for CarabID (YOLOv11n-cls, 76 genera). ",
      "Dashed line indicates perfect calibration. ",
      "Points represent bins with at least one sample."
    )
  )

# ---- save ----------------------------------------------------------------
pdf_path <- file.path(out_dir, "reliability_diagram.pdf")
png_path <- file.path(out_dir, "reliability_diagram.png")

ggsave(pdf_path, combined, width = 6, height = 7, device = "pdf")
ggsave(png_path, combined, width = 6, height = 7, dpi = 300, device = "png")

cat("Saved:\n  ", pdf_path, "\n  ", png_path, "\n")
