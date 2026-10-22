library(tidyverse)
library(lubridate)
library(here)

# =========================================================
# CONFIG
# =========================================================
base_dir <- here("data", "raw")
dir_error <- file.path(base_dir, "adl_error")
dir_noerror <- file.path(base_dir, "adl_noerror")

target_sensors <- c("AD1-A")

# =========================================================
# HELPERS
# =========================================================
extract_activity <- function(filename) {
  str_extract(filename, "t[1-5]")
}

read_dataset <- function(directory, label) {
  files <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  
  map_dfr(files, function(fp) {
    df <- read_csv(fp, show_col_types = FALSE)
    
    # make names lowercase in case files vary
    names(df) <- tolower(names(df))
    
    # require expected columns
    required_cols <- c("date", "time", "sensor", "message")
    missing_cols <- setdiff(required_cols, names(df))
    if (length(missing_cols) > 0) {
      warning(sprintf(
        "Skipping %s because it is missing columns: %s",
        basename(fp),
        paste(missing_cols, collapse = ", ")
      ))
      return(tibble())
    }
    
    df %>%
      mutate(
        timestamp = ymd_hms(paste(.data$date, .data$time), quiet = TRUE),
        sensor = toupper(as.character(.data$sensor)),
        filename = basename(fp),
        dataset = label,
        activity = extract_activity(filename)
      )
  })
}

plot_dataset <- function(df, title_text) {
  if (nrow(df) == 0) {
    message("No rows available for: ", title_text)
    return(NULL)
  }
  
  ggplot(df, aes(x = timestamp, y = value, color = sensor)) +
    geom_line(alpha = 0.8) +
    geom_vline(
      aes(xintercept = as.numeric(end_time)),
      linetype = "dotted",
      linewidth = 0.7
    ) +
    facet_wrap(~ activity + filename, scales = "free_x") +
    labs(
      title = title_text,
      subtitle = "Dotted line = episode end",
      x = "Time",
      y = "Burner Value",
      color = "Sensor"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 6),
      legend.position = "bottom"
    )
}

# =========================================================
# LOAD DATA
# =========================================================
df_error <- read_dataset(dir_error, "Error")
df_noerror <- read_dataset(dir_noerror, "No Error")

df_all <- bind_rows(df_error, df_noerror)

# =========================================================
# FILTER + PREP
# =========================================================
df_filtered <- df_all %>%
  filter(sensor %in% target_sensors) %>%
  mutate(value = suppressWarnings(as.numeric(message))) %>%
  filter(!is.na(value), !is.na(timestamp), !is.na(activity))

episode_end <- df_filtered %>%
  group_by(dataset, activity, filename) %>%
  summarise(end_time = max(timestamp, na.rm = TRUE), .groups = "drop")

df_filtered <- df_filtered %>%
  left_join(episode_end, by = c("dataset", "activity", "filename"))

# =========================================================
# GENERATE SEPARATE PLOTS
# =========================================================
p_error <- plot_dataset(
  df_filtered %>% filter(dataset == "Error"),
  "AD1-A Burner — ERROR Episodes (All Activities)"
)

p_noerror <- plot_dataset(
  df_filtered %>% filter(dataset == "No Error"),
  "AD1-A Burner — NO ERROR Episodes (All Activities)"
)

if (!is.null(p_error)) print(p_error)
if (!is.null(p_noerror)) print(p_noerror)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================
out_dir <- here("results", "analysis", "graphs")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
print(out_dir)

if (!is.null(p_error)) {
  ggsave(
    filename = file.path(out_dir, "all_activities_burner_error.png"),
    plot = p_error,
    width = 18,
    height = 12,
    dpi = 300
  )
}

if (!is.null(p_noerror)) {
  ggsave(
    filename = file.path(out_dir, "all_activities_burner_noerror.png"),
    plot = p_noerror,
    width = 18,
    height = 12,
    dpi = 300
  )
}