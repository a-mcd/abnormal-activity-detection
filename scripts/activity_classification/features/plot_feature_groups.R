library(tidyverse)
library(scales)
library(here)

# =========================================================
# PATHS
# =========================================================
base_dir <- here("results", "activity_classification", "features")
error_file <- file.path(base_dir, "engineered_features/error_all_features.csv")
noerror_file <- file.path(base_dir, "engineered_features/noerror_all_features.csv")
outdir <- file.path(base_dir, "graphs")

evt_bin_outdir <- file.path(outdir, "evt_bin")
evt_outdir    <- file.path(outdir, "evt")
sensor_outdir <- file.path(outdir, "sensor")
msg_outdir    <- file.path(outdir, "msg")
trans_outdir  <- file.path(outdir, "trans")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(evt_bin_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(evt_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(sensor_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(msg_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(trans_outdir, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# LOAD DATA
# =========================================================
err <- read_csv(error_file, show_col_types = FALSE) %>%
  mutate(dataset = "error")

noerr <- read_csv(noerror_file, show_col_types = FALSE) %>%
  mutate(dataset = "noerror")

df <- bind_rows(err, noerr) %>%
  rename(activity = activity_code)

df$activity <- factor(df$activity, levels = c("t1", "t2", "t3", "t4", "t5"))
df$dataset <- factor(df$dataset, levels = c("error", "noerror"))

# =========================================================
# HELPERS
# =========================================================
safe_name <- function(x) {
  x %>%
    str_replace_all("::", "__") %>%
    str_replace_all("[^A-Za-z0-9_]+", "_") %>%
    str_replace_all("_+", "_") %>%
    str_replace_all("^_|_$", "")
}

extract_evt_sensor <- function(feature_name) {
  parts <- str_split(feature_name, "::", simplify = TRUE)
  if (ncol(parts) >= 3) return(parts[, 2])
  NA_character_
}

extract_evt_bin_sensor <- function(feature_name) {
  parts <- str_split(feature_name, "::", simplify = TRUE)
  if (ncol(parts) >= 3) return(parts[, 2])
  NA_character_
}

extract_sensor_name <- function(feature_name) {
  str_remove(feature_name, "^sensor::")
}

extract_msg_name <- function(feature_name) {
  str_remove(feature_name, "^msg::")
}

extract_trans_source_sensor <- function(feature_name) {
  x <- str_remove(feature_name, "^trans::")
  parts <- str_split(x, "::", simplify = TRUE)
  if (ncol(parts) >= 1) return(parts[, 1])
  NA_character_
}

sensor_type_from_sensor <- function(sensor) {
  sensor_upper <- str_to_upper(sensor)
  
  case_when(
    str_detect(sensor_upper, "^M")  ~ "Motion",
    str_detect(sensor_upper, "^D")  ~ "Door",
    str_detect(sensor_upper, "^I")  ~ "Item",
    str_detect(sensor_upper, "^AD") ~ "Analogue",
    str_detect(sensor_upper, "^E")  ~ "Event",
    str_detect(sensor_upper, "^T")  ~ "Temperature",
    str_detect(sensor_upper, "^L")  ~ "Light",
    str_detect(sensor_upper, "^P")  ~ "Presence",
    TRUE ~ "Other"
  )
}

message_type_from_message <- function(msg) {
  msg_upper <- str_to_upper(msg)
  
  case_when(
    msg_upper %in% c("ON", "OFF", "OPEN", "CLOSE", "CLOSED") ~ "Binary state",
    msg_upper %in% c("PRESENT", "ABSENT") ~ "Presence",
    msg_upper %in% c("START", "STOP", "END") ~ "Start/Stop",
    msg_upper %in% c("UP", "DOWN") ~ "Direction",
    msg_upper %in% c("ACTIVE", "INACTIVE") ~ "Active state",
    str_detect(msg_upper, "^[0-9.\\-]+$") ~ "Numeric",
    TRUE ~ "Other"
  )
}

# =========================================================
# COLOUR SETTINGS
# =========================================================
activity_fill <- c(
  "t1" = "#e9968d",
  "t2" = "#bdbf6f",
  "t3" = "#7ccba2",
  "t4" = "#6baed6",
  "t5" = "#c994c7"
)

dataset_col <- c(
  "error" = "#d94202",
  "noerror" = "#117c88"
)

# =========================================================
# IDENTIFY ONLY REQUIRED FEATURE GROUPS
# =========================================================
all_numeric_features <- names(df)[sapply(df, is.numeric)]

evt_bin_features_to_group <- grep("^evt_bin::", all_numeric_features, value = TRUE)
evt_features_to_group     <- grep("^evt::", all_numeric_features, value = TRUE)
sensor_features_to_group  <- grep("^sensor::", all_numeric_features, value = TRUE)
msg_features_to_group     <- grep("^msg::", all_numeric_features, value = TRUE)
trans_features_to_group   <- grep("^trans::", all_numeric_features, value = TRUE)

# =========================================================
# GENERIC GROUPED PLOTTER
# =========================================================
plot_grouped_features <- function(
    data,
    features,
    group_info,
    group_var,
    outdir,
    file_prefix,
    title_prefix,
    max_facets_per_plot = 9
) {
  if (length(features) == 0 || nrow(group_info) == 0) return(character(0))
  
  saved <- c()
  
  groups <- group_info %>%
    distinct(.data[[group_var]]) %>%
    arrange(.data[[group_var]]) %>%
    pull(.data[[group_var]])
  
  for (grp in groups) {
    grp_features <- group_info %>%
      filter(.data[[group_var]] == grp) %>%
      arrange(feature) %>%
      pull(feature)
    
    if (length(grp_features) == 0) next
    
    feature_chunks <- split(
      grp_features,
      ceiling(seq_along(grp_features) / max_facets_per_plot)
    )
    
    for (i in seq_along(feature_chunks)) {
      chunk_features <- feature_chunks[[i]]
      
      plot_df <- data %>%
        select(activity, dataset, all_of(chunk_features)) %>%
        pivot_longer(
          cols = all_of(chunk_features),
          names_to = "feature",
          values_to = "value"
        ) %>%
        filter(!is.na(value)) %>%
        group_by(feature) %>%
        filter(n_distinct(value) > 1) %>%
        ungroup() %>%
        mutate(feature = factor(feature, levels = chunk_features))
      
      if (nrow(plot_df) == 0) next
      
      n_feat <- length(unique(plot_df$feature))
      ncol_facets <- min(3, n_feat)
      nrow_facets <- ceiling(n_feat / ncol_facets)
      
      plot_title <- if (length(feature_chunks) == 1) {
        paste(title_prefix, "-", grp)
      } else {
        paste(title_prefix, "-", grp, "(Part", i, "of", length(feature_chunks), ")")
      }
      
      p <- ggplot(plot_df, aes(x = activity, y = value, fill = activity)) +
        geom_boxplot(
          alpha = 0.7,
          outlier.shape = NA
        ) +
        geom_jitter(
          aes(color = dataset),
          width = 0.12,
          height = 0,
          size = 1.1,
          alpha = 0.75
        ) +
        facet_wrap(~ feature, scales = "free_y", ncol = ncol_facets) +
        scale_fill_manual(values = activity_fill) +
        scale_color_manual(values = dataset_col) +
        scale_y_continuous(labels = comma) +
        labs(
          title = plot_title,
          x = "Activity",
          y = "Value",
          fill = "Activity",
          color = "Dataset"
        ) +
        theme_minimal(base_size = 11) +
        theme(
          legend.position = "right",
          strip.text = element_text(size = 8),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      outfile <- if (length(feature_chunks) == 1) {
        file.path(outdir, paste0(file_prefix, "_", safe_name(grp), ".png"))
      } else {
        file.path(
          outdir,
          paste0(
            file_prefix, "_", safe_name(grp),
            "_part_", str_pad(i, 2, pad = "0"), ".png"
          )
        )
      }
      
      ggsave(
        filename = outfile,
        plot = p,
        width = 3.2 * ncol_facets,
        height = max(4.5, 2.4 * nrow_facets),
        dpi = 300
      )
      
      saved <- c(saved, outfile)
    }
  }
  
  saved
}

# =========================================================
# BUILD GROUPING TABLES
# =========================================================
evt_bin_group_info <- tibble(
  feature = evt_bin_features_to_group,
  sensor = map_chr(evt_bin_features_to_group, extract_evt_bin_sensor),
  group = sensor_type_from_sensor(sensor)
)

evt_group_info <- tibble(
  feature = evt_features_to_group,
  sensor = map_chr(evt_features_to_group, extract_evt_sensor),
  group = sensor_type_from_sensor(sensor)
)

sensor_group_info <- tibble(
  feature = sensor_features_to_group,
  sensor = map_chr(sensor_features_to_group, extract_sensor_name),
  group = sensor_type_from_sensor(sensor)
)

msg_group_info <- tibble(
  feature = msg_features_to_group,
  message = map_chr(msg_features_to_group, extract_msg_name),
  group = message_type_from_message(message)
)

trans_group_info <- tibble(
  feature = trans_features_to_group,
  sensor = map_chr(trans_features_to_group, extract_trans_source_sensor),
  group = sensor_type_from_sensor(sensor)
)

# =========================================================
# CREATE GROUPED EVT_BIN PLOTS
# =========================================================
evt_bin_group_files <- tryCatch(
  plot_grouped_features(
    data = df,
    features = evt_bin_features_to_group,
    group_info = evt_bin_group_info,
    group_var = "group",
    outdir = evt_bin_outdir,
    file_prefix = "evt_bin_grouped",
    title_prefix = "evt_bin Features",
    max_facets_per_plot = 9
  ),
  error = function(e) {
    message("Skipped evt_bin grouped plots: ", e$message)
    character(0)
  }
)

# =========================================================
# CREATE GROUPED EVT PLOTS
# =========================================================
evt_group_files <- tryCatch(
  plot_grouped_features(
    data = df,
    features = evt_features_to_group,
    group_info = evt_group_info,
    group_var = "group",
    outdir = evt_outdir,
    file_prefix = "evt_grouped",
    title_prefix = "evt Features",
    max_facets_per_plot = 9
  ),
  error = function(e) {
    message("Skipped evt grouped plots: ", e$message)
    character(0)
  }
)

# =========================================================
# CREATE GROUPED SENSOR PLOTS
# =========================================================
sensor_group_files <- tryCatch(
  plot_grouped_features(
    data = df,
    features = sensor_features_to_group,
    group_info = sensor_group_info,
    group_var = "group",
    outdir = sensor_outdir,
    file_prefix = "sensor_grouped",
    title_prefix = "sensor Features",
    max_facets_per_plot = 9
  ),
  error = function(e) {
    message("Skipped sensor grouped plots: ", e$message)
    character(0)
  }
)

# =========================================================
# CREATE GROUPED MSG PLOTS
# =========================================================
msg_group_files <- tryCatch(
  plot_grouped_features(
    data = df,
    features = msg_features_to_group,
    group_info = msg_group_info,
    group_var = "group",
    outdir = msg_outdir,
    file_prefix = "msg_grouped",
    title_prefix = "msg Features",
    max_facets_per_plot = 9
  ),
  error = function(e) {
    message("Skipped msg grouped plots: ", e$message)
    character(0)
  }
)

# =========================================================
# CREATE GROUPED TRANS PLOTS
# =========================================================
trans_group_files <- tryCatch(
  plot_grouped_features(
    data = df,
    features = trans_features_to_group,
    group_info = trans_group_info,
    group_var = "group",
    outdir = trans_outdir,
    file_prefix = "trans_grouped",
    title_prefix = "trans Features",
    max_facets_per_plot = 9
  ),
  error = function(e) {
    message("Skipped trans grouped plots: ", e$message)
    character(0)
  }
)

# =========================================================
# SUMMARY
# =========================================================
all_saved <- c(
  evt_bin_group_files,
  evt_group_files,
  sensor_group_files,
  msg_group_files,
  trans_group_files
)

cat("Saved", length(all_saved), "plots under:\n", outdir, "\n")