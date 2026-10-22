library(tidyverse)
library(scales)
library(here)

# =========================================================
# PATHS
# =========================================================
base_dir <- here("data", "raw")
dir_error <- file.path(base_dir, "adl_error")
dir_noerror <- file.path(base_dir, "adl_noerror")

outdir <- here("results", "analysis", "graphs")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# HELPERS
# =========================================================

# Extract activity (t1–t5) from filename
extract_activity <- function(filename) {
  str_match(filename, "\\.(t[1-5])\\.")[, 2]
}

# Convert AD1 sensors into ACTIVE / INACTIVE labels
transform_ad1 <- function(df) {
  df %>%
    mutate(
      sensor = case_when(
        str_to_lower(sensor) == "asterisk" ~ "PHONE",  # 🔁 rename
        TRUE ~ sensor
      ),
      
      sensor_lower = str_to_lower(sensor),
      val = suppressWarnings(as.numeric(message)),
      
      message = case_when(
        # AD1 sensors: keep ONLY ACTIVE
        sensor_lower %in% c("ad1-a", "ad1-b", "ad1-c") & val > 0 ~ "ACTIVE",
        
        # all other sensors unchanged
        TRUE ~ as.character(message)
      )
    ) %>%
    filter(
      !is.na(message),
      !str_starts(str_to_upper(sensor), "E01")
    ) %>% 
    select(-sensor_lower, -val)
}

# Read all CSVs and tag with dataset + activity
read_all_files <- function(directory, dataset_label) {
  files <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  
  map_dfr(files, function(fp) {
    filename <- basename(fp)
    activity <- extract_activity(filename)
    
    df <- read_csv(fp, show_col_types = FALSE)
    
    if (!all(c("sensor", "message") %in% colnames(df))) {
      return(NULL)
    }
    
    df %>%
      transform_ad1() %>%
      mutate(
        file = filename,
        dataset = dataset_label,
        activity = activity
      )
  })
}

# =========================================================
# LOAD DATA
# =========================================================
df_error <- read_all_files(dir_error, "error")
df_noerror <- read_all_files(dir_noerror, "noerror")

df_all <- bind_rows(df_error, df_noerror)

# =========================================================
# UNIQUE SENSOR-MESSAGE PER FILE
# =========================================================
df_unique <- df_all %>%
  distinct(dataset, activity, file, sensor, message)

# Build sensor_message label for plotting
df_unique <- df_unique %>%
  mutate(sensor_message = paste(sensor, message, sep = "_"))

# =========================================================
# COUNT TOTAL EPISODES PER DATASET + ACTIVITY
# =========================================================
episode_totals <- df_unique %>%
  distinct(dataset, activity, file) %>%
  count(dataset, activity, name = "total_episodes")

# =========================================================
# COUNT FILE PRESENCE PER SENSOR-MESSAGE
# =========================================================
sensor_message_counts <- df_unique %>%
  group_by(dataset, activity, sensor_message) %>%
  summarise(
    n_files = n(),
    .groups = "drop"
  ) %>%
  left_join(episode_totals, by = c("dataset", "activity")) %>%
  mutate(
    episode_pct = n_files / total_episodes
  )

# =========================================================
# ORDER (GLOBAL CONSISTENT ORDER)
# =========================================================

# 1. Get ALL sensor_message values across both datasets
all_sensor_messages <- sensor_message_counts %>%
  distinct(sensor_message) %>%
  arrange(sensor_message) %>%
  pull(sensor_message)

# 2. Ensure all combinations exist (prevents missing rows)
sensor_message_counts_plot <- sensor_message_counts %>%
  complete(
    dataset,
    activity,
    sensor_message = all_sensor_messages,
    fill = list(episode_pct = 0)
  )

# 3. Apply consistent ordering
sensor_message_counts_plot <- sensor_message_counts_plot %>%
  mutate(
    activity = factor(activity, levels = c("t1", "t2", "t3", "t4", "t5")),
    dataset = factor(dataset, levels = c("noerror", "error")) %>%
      fct_recode(
        "Error" = "error",
        "Normal" = "noerror"
      ),
    
    sensor_message = factor(
      sensor_message,
      levels = all_sensor_messages
    ),
    
    activity_label = paste0(activity, "_count")
  )


# =========================================================
# PLOT
# =========================================================
p <- ggplot(
  sensor_message_counts_plot,
  aes(x = activity_label, y = sensor_message, fill = episode_pct)
) +
  geom_tile(color = "white", linewidth = 0.2) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_gradient(
    low = "grey95",
    high = "steelblue",
    labels = percent_format(accuracy = 1),
    limits = c(0, 1),
    name = "Episode %"
  ) +
  labs(
    title = "Sensor Presence Rate per Activity",
    x = "activity",
    y = "sensor_message"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    strip.text = element_text(face = "bold"),
    legend.position = "right"
  )

print(p)

# =========================================================
# SAVE OUTPUTS
# =========================================================
write_csv(
  sensor_message_counts,
  here("results", "analysis", "sensor_message_counts_by_activity_long.csv")
)

ggsave(
  filename = file.path(outdir, "sensor_presence_rate_per_activity_heatmap.png"),
  plot = p,
  width = 16,
  height = 10,
  dpi = 300
)