library(ggplot2)

results_path <- file.path(
  "runs", "classify",
  "final_carabid_v2_11ncls_ep30_do02_lr001",
  "results.csv"
)
out_dir <- file.path("scripts", "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(results_path)
names(df) <- trimws(names(df))

col_train <- "#1B4F8A"
col_val   <- "#A8D0E6"
col_top1  <- "#2980B9"

panel_loss <- "Training and Validation Loss"
panel_acc  <- "Top1 Accuracy"

df_long <- rbind(
  data.frame(epoch = df$epoch, value = df$train.loss,
             Metric = "Training", panel = panel_loss),
  data.frame(epoch = df$epoch, value = df$val.loss,
             Metric = "Validation", panel = panel_loss),
  data.frame(epoch = df$epoch, value = df$metrics.accuracy_top1,
             Metric = "Top1 accuracy", panel = panel_acc)
)

df_long$Metric <- factor(df_long$Metric,
                         levels = c("Top1 accuracy", "Training", "Validation"))
df_long$panel  <- factor(df_long$panel, levels = c(panel_loss, panel_acc))

p <- ggplot(df_long, aes(x = epoch, y = value,
                          colour = Metric, group = Metric)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  facet_wrap(~panel, scales = "free_y") +
  scale_colour_manual(
    values = c("Training"     = col_train,
               "Validation"   = col_val,
               "Top1 accuracy" = col_top1),
    breaks = c("Top1 accuracy", "Training", "Validation")
  ) +
  scale_x_continuous(breaks = seq(0, 30, by = 10)) +
  labs(x = "Epoch", y = "Value", colour = "Metric") +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor    = element_line(colour = "grey92"),
    panel.grid.major    = element_line(colour = "grey85"),
    legend.position     = "bottom",
    legend.title        = element_text(size = 10),
    strip.text          = element_text(size = 11),
    strip.background    = element_rect(fill = "white", colour = NA)
  )

ggsave(file.path(out_dir, "training_curves.pdf"),
       p, width = 8, height = 4.5, device = "pdf")
ggsave(file.path(out_dir, "training_curves.png"),
       p, width = 8, height = 4.5, dpi = 300)
cat("Saved: training_curves.pdf / .png\n")
