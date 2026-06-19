library(ggplot2)

results_path <- file.path(
  "runs", "detect",
  "model_11n_ep30_autobatch",
  "results.csv"
)
out_dir <- file.path("scripts", "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(results_path)
names(df) <- trimws(names(df))

col_train <- "#1B4F8A"
col_val   <- "#A8D0E6"
col_map   <- "#1B4F8A"
col_prec  <- "#2980B9"
col_rec   <- "#5DADE2"

panel_loss    <- "Training and Validation Loss"
panel_metrics <- "Detection Metrics"

df_long <- rbind(
  data.frame(epoch  = df$epoch,
             value  = df$train.box_loss,
             Metric = "Training",
             panel  = panel_loss),
  data.frame(epoch  = df$epoch,
             value  = df$val.box_loss,
             Metric = "Validation",
             panel  = panel_loss),
  data.frame(epoch  = df$epoch,
             value  = df$metrics.mAP50.95.B.,
             Metric = "mAP50-95",
             panel  = panel_metrics)
)

df_long$Metric <- factor(df_long$Metric,
  levels = c("mAP50-95", "Training", "Validation"))
df_long$panel <- factor(df_long$panel,
  levels = c(panel_loss, panel_metrics))

p <- ggplot(df_long, aes(x = epoch, y = value,
                          colour = Metric, group = Metric)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  facet_wrap(~panel, scales = "free_y") +
  scale_colour_manual(
    values = c("Training"  = col_train,
               "Validation" = col_val,
               "mAP50-95"  = col_map)
  ) +
  scale_x_continuous(breaks = seq(0, 30, by = 10)) +
  labs(x = "Epoch", y = "Value", colour = "Metric") +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor  = element_line(colour = "grey92"),
    panel.grid.major  = element_line(colour = "grey85"),
    legend.position   = "bottom",
    legend.title      = element_text(size = 10),
    strip.text        = element_text(size = 11),
    strip.background  = element_rect(fill = "white", colour = NA)
  )

ggsave(file.path(out_dir, "detection_curves.pdf"),
       p, width = 8, height = 4.5, device = "pdf")
ggsave(file.path(out_dir, "detection_curves.png"),
       p, width = 8, height = 4.5, dpi = 300)
cat("Saved: detection_curves.pdf / .png\n")
