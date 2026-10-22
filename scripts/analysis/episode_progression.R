library(tidyverse)
library(lubridate)
library(scales)
library(here)

# =========================================================
# PATHS
# =========================================================
base_dir <- here("data", "raw")
dir_error <- file.path(base_dir, "adl_error")
dir_noerror <- file.path(base_dir, "adl_noerror")

# Set to NULL for all activities, or "t1", "t2", ..., "t5"
target_activity <- "t5"

# Optional: collapse consecutive identical sensor-message rows
collapse_consecutive <- FALSE

# Optional: exclude motion OFF events (e.g. M17 OFF)
exclude_motion_off <- FALSE

# =========================================================
# HELPERS
# =========================================================
extract_activity <- function(filename) {
  str_match(filename, "\\.(t[1-5])\\.")[, 2]
}

read_episode_events <- function(directory, dataset_label, target_activity = NULL,
                                collapse_consecutive = FALSE,
                                exclude_motion_off = FALSE) {
  files <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  
  map_dfr(files, function(fp) {
    filename <- basename(fp)
    activity <- extract_activity(filename)
    
    if (is.na(activity)) return(NULL)
    if (!is.null(target_activity) && activity != target_activity) return(NULL)
    
    df <- read_csv(fp, show_col_types = FALSE)
    
    required_cols <- c("date", "time", "sensor", "message")
    if (!all(required_cols %in% names(df))) return(NULL)
    
    df <- df %>%
      mutate(
        sensor = as.character(sensor),
        message = as.character(message),
        timestamp = suppressWarnings(ymd_hms(paste(date, time), quiet = TRUE))
      ) %>%
      filter(!is.na(timestamp), !is.na(sensor))
    
    # -------------------------------------------------
    # Remove motion OFF events (Mxx OFF)
    # -------------------------------------------------
    if (exclude_motion_off) {
      df <- df %>%
        filter(!(str_detect(sensor, "^M\\d+") & toupper(message) == "OFF"))
    }
    
    df <- df %>%
      arrange(timestamp)
    
    if (nrow(df) == 0) return(NULL)
    
    if (collapse_consecutive) {
      df <- df %>%
        mutate(
          prev_sensor = lag(sensor),
          prev_message = lag(message),
          keep = row_number() == 1 | sensor != prev_sensor | message != prev_message
        ) %>%
        filter(keep) %>%
        select(-prev_sensor, -prev_message, -keep)
    }
    
    episode_start <- min(df$timestamp, na.rm = TRUE)
    episode_end   <- max(df$timestamp, na.rm = TRUE)
    duration_sec  <- as.numeric(difftime(episode_end, episode_start, units = "secs"))
    
    df <- df %>%
      mutate(
        dataset = dataset_label,
        filename = filename,
        activity = activity,
        episode_start = episode_start,
        episode_end = episode_end,
        episode_duration_sec = duration_sec,
        seconds_from_start = as.numeric(difftime(timestamp, episode_start, units = "secs"))
      )
    
    if (duration_sec > 0) {
      df <- df %>%
        mutate(episode_progress = seconds_from_start / duration_sec)
    } else {
      df <- df %>%
        mutate(episode_progress = 0)
    }
    
    df %>%
      select(
        dataset, activity, filename, sensor, message, timestamp,
        seconds_from_start, episode_progress, episode_duration_sec
      )
  })
}

# =========================================================
# LOAD DATA
# =========================================================
error_events <- read_episode_events(
  dir_error,
  "Error",
  target_activity = target_activity,
  collapse_consecutive = collapse_consecutive,
  exclude_motion_off = exclude_motion_off
)

noerror_events <- read_episode_events(
  dir_noerror,
  "No Error",
  target_activity = target_activity,
  collapse_consecutive = collapse_consecutive,
  exclude_motion_off = exclude_motion_off
)

events_all <- bind_rows(error_events, noerror_events)

# Optional: order sensors alphabetically
events_all <- events_all %>%
  mutate(sensor = factor(sensor, levels = rev(sort(unique(sensor)))))

# =========================================================
# PLOT: one panel per episode
# =========================================================
p <- ggplot(events_all, aes(x = episode_progress, y = sensor, colour = sensor)) +
  geom_point(alpha = 0.75, size = 1.8) +
  facet_wrap(~ dataset + filename, scales = "free_y") +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.1),
    labels = percent_format(accuracy = 1)
  ) +
  labs(
    title = paste(
      "Sensor events across episode progression",
      if (!is.null(target_activity)) paste0("(", target_activity, ")") else "(all activities)"
    ),
    subtitle = paste(
      "One panel per episode",
      if (exclude_motion_off) "| motion OFF events removed" else ""
    ),
    x = "Position in episode",
    y = "Sensor",
    colour = "Sensor"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "right"
  )

print(p)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================
output_base_dir <- here("results", "analysis", "graphs")
out_dir <- file.path(output_base_dir, "episode_progress")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# SAVE PLOT
# =========================================================

cat("Rows in error_events: ", nrow(error_events), "\n")
cat("Rows in noerror_events: ", nrow(noerror_events), "\n")
cat("Rows in events_all: ", nrow(events_all), "\n")
print(unique(events_all$sensor))
print(unique(events_all$activity))

suffix <- if (exclude_motion_off) "_no_motion_off" else ""

outfile <- file.path(
  out_dir,
  paste0(target_activity, "_episode_progress_plot", suffix, ".png")
)

ggsave(
  filename = outfile,
  plot = p,
  width = 24,
  height = 10,
  dpi = 300
)

cat("Plot saved to: ", outfile, "\n", sep = "")
file.exists(outfile)

