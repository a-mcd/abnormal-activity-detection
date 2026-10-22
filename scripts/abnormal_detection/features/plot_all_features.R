library(tidyverse)
library(readr)
library(stringr)
library(here)

# =========================================================
# CONFIG
# =========================================================
base_dir <- here("results", "abnormal_detection", "features")

target_activity <- "t5"   # change as needed

error_file <- file.path(
  base_dir,
  target_activity,
  "engineered_features",
  paste0(target_activity, "_error_all_features.csv")
)

noerror_file <- file.path(
  base_dir,
  target_activity,
  "engineered_features",
  paste0(target_activity, "_noerror_all_features.csv")
)

filter_activity <- TRUE
max_features_per_plot <- 40
restore_minus1_to_na <- TRUE


# =========================================================
# HELPERS
# =========================================================
safe_filename <- function(x) {
  x %>%
    str_replace_all("[^A-Za-z0-9_\\-]+", "_") %>%
    str_replace_all("_+", "_") %>%
    str_replace_all("^_|_$", "")
}

feature_group <- function(feature_name) {
  case_when(
    feature_name %in% c(
      "UNIQUE_SENSOR_COUNT", "EPISODE_DURATION", "NUM_EVENTS",
      "N_BIGRAMS", "N_TRIGRAMS", "UNIQUE_MESSAGES",
      "UNIQUE_SENSOR_MESSAGE_COUNT"
    ) ~ "01_Count_And_Diversity",
    
    feature_name %in% c(
      "MEAN_DT", "STD_DT", "DT_MEDIAN", "P90", "MAX_DT",
      "PCT_DT_ZERO", "LOGDT_MEAN", "LOGDT_STD", "EVENT_RATE",
      "EVENTS_PER_MIN", "CV_DT", "BURSTINESS",
      "BURSTINESS_X_P90", "SWITCH_RATE", "REPEAT_RATIO",
      "SAME_EVENT_REPEAT_RATIO", "P90_MINUS_P50",
      "P90_X_SWITCH", "SWITCH_OVER_P90", "MAX_DT_OVER_P90",
      "SPIKE_COUNT"
    ) ~ "02_Temporal_And_Dynamics",
    
    feature_name %in% c(
      "BIGRAM_MATCH_RATIO", "TRIGRAM_MATCH_RATIO",
      "SENSOR_BIGRAM_MATCH_RATIO", "EXTRA_BIGRAM_RATIO",
      "EXTRA_EVENT_RATIO", "EXTRA_TRIGRAM_RATIO",
      "MISSING_BIGRAM_RATIO", "MISSING_EVENT_RATIO",
      "MEAN_TRANSITION_LOG_LIK", "TRANSITION_LOG_LIK"
    ) ~ "03_Sequence_Match",
    
    feature_name %in% c(
      "HAS_ANY_PHONE_CALL",
      "FIRST_PHONEBOOK_TO_LAST_PHONE_DELAY",
      "PHONE_COMPLETED_CALL_COUNT",
      "PHONE_INCOMPLETE_CALL_COUNT",
      "FIRST_PHONEBOOK_TO_FIRST_PHONE_START_DELAY",
      "PHONE_TOTAL_CALL_TIME",
      "PHONE_LONGEST_CALL_TIME",
      "PHONE_SHORTEST_CALL_TIME",
      "LAST_CALL_END_TO_EPISODE_END"
    ) ~ "04_Phone_Features",
    
    feature_name %in% c(
      "UNEXPECTED_SENSOR_DURING_TAP_WINDOW",
      "LAST_TAP_VALUE",
      "LAST_TAP_VALUE_LT_2_5",
      "AD1_BC_ACTIVE_LAST_8S",
      "AD1_BC_ACTIVE_LAST_TO_END",
      "AD1_BC_TOTAL_INTERACTIONS",
      "AD1_B_LAST_IS_DOWN",
      "AD1_C_LAST_IS_DOWN"
    ) ~ "14_Tap_Value_Features",
    
    str_starts(feature_name, "analogue_ad1_a") ~ "05_AD1_A_Analogue",
    str_starts(feature_name, "analogue_ad1_b") ~ "06_AD1_B_Analogue",
    str_starts(feature_name, "analogue_ad1_c") ~ "07_AD1_C_Analogue",
    
    str_starts(feature_name, "sensor::") ~ "08_Sensor_Features",
    str_starts(feature_name, "msg::")    ~ "09_Message_Features",
    str_starts(feature_name, "evt::")    ~ "10_Event_Features",
    str_starts(feature_name, "state::")  ~ "11_State_Features",
    str_starts(feature_name, "type::")   ~ "12_Type_Features",
    str_starts(feature_name, "trans::")  ~ "13_Transition_Features",
    
    TRUE ~ "99_Other"
  )
}

restore_plot_na <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  
  if (!restore_minus1_to_na) {
    return(x_num)
  }
  
  ifelse(!is.na(x_num) & x_num == -1, NA_real_, x_num)
}

# =========================================================
# OUTPUT DIRECTORY
# =========================================================
if (filter_activity) {
  base_activity_dir <- file.path(base_dir, target_activity)
} else {
  base_activity_dir <- file.path(base_dir, "all_activities")
}

graphs_dir <- file.path(base_activity_dir, "graphs")

out_dir_groups <- file.path(graphs_dir, "feature_scatter_by_group")
out_dir_features <- file.path(graphs_dir, "feature_scatter_individual")

out_file_groups <- file.path(base_activity_dir, "feature_groups.csv")
out_file_summary <- file.path(base_activity_dir, "feature_summary_by_dataset.csv")

dir.create(out_dir_groups, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir_features, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# READ DATA
# =========================================================
error_df <- read_csv(error_file, show_col_types = FALSE)
noerror_df <- read_csv(noerror_file, show_col_types = FALSE)

all_df <- bind_rows(error_df, noerror_df) %>%
  mutate(episode_id = row_number())

# Safety checks
required_cols <- c("filename", "participant_id", "activity_code", "activity_index", "dataset")
missing_required <- setdiff(required_cols, names(all_df))

if (length(missing_required) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_required, collapse = ", ")
    )
  )
}

if (filter_activity) {
  all_df <- all_df %>%
    filter(activity_code == target_activity)
}

# =========================================================
# RESHAPE TO LONG FORMAT
# =========================================================
id_cols <- c(
  "filename", "participant_id", "activity_code",
  "activity_index", "dataset", "episode_id"
)

feature_cols <- setdiff(names(all_df), id_cols)

long_df <- all_df %>%
  pivot_longer(
    cols = all_of(feature_cols),
    names_to = "feature",
    values_to = "value_raw"
  ) %>%
  mutate(
    #value = suppressWarnings(as.numeric(value)),
    value = restore_plot_na(value_raw),
    feature_group = feature_group(feature)
  ) %>%
  filter(!is.na(value))

# =========================================================
# SAVE FEATURE LIST BY GROUP
# =========================================================
group_summary <- long_df %>%
  distinct(feature_group, feature) %>%
  arrange(feature_group, feature)

write_csv(group_summary, out_file_groups)

# =========================================================
# GROUPED PLOTS: BOXPLOT + JITTER
# =========================================================
groups <- unique(long_df$feature_group)

for (g in groups) {
  plot_df <- long_df %>%
    filter(feature_group == g)
  
  if (nrow(plot_df) == 0) next
  
  feature_list <- plot_df %>%
    distinct(feature) %>%
    arrange(feature) %>%
    pull(feature)
  
  feature_chunks <- split(
    feature_list,
    ceiling(seq_along(feature_list) / max_features_per_plot)
  )
  
  for (i in seq_along(feature_chunks)) {
    chunk_features <- feature_chunks[[i]]
    
    chunk_df <- plot_df %>%
      filter(feature %in% chunk_features)
    
    n_chunk_features <- n_distinct(chunk_df$feature)
    
    plot_height <- min(
      max(8, ceiling(n_chunk_features / 4) * 2.6),
      40
    )
    
    p <- ggplot(chunk_df, aes(x = dataset, y = value, colour = dataset)) +
      geom_boxplot(
        width = 0.5,
        alpha = 0.2,
        outlier.shape = NA
      ) +
      geom_jitter(
        width = 0.10,
        height = 0,
        alpha = 0.75,
        size = 1.8
      ) +
      facet_wrap(~ feature, scales = "free_y", ncol = 4) +
      labs(
        title = paste("Error vs No Error:", g, "- Part", i),
        x = NULL,
        y = "Feature value"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        legend.position = "none",
        strip.text = element_text(size = 7),
        axis.text.x = element_text(face = "bold")
      )
    
    ggsave(
      filename = file.path(
        out_dir_groups,
        paste0(safe_filename(g), "_part_", i, ".png")
      ),
      plot = p,
      width = 16,
      height = plot_height,
      dpi = 300
    )
  }
}

# =========================================================
# INDIVIDUAL FEATURE PLOTS
# =========================================================
all_features <- unique(long_df$feature)

for (f in all_features) {
  plot_df <- long_df %>%
    filter(feature == f)
  
  if (nrow(plot_df) == 0) next
  
  p <- ggplot(plot_df, aes(x = dataset, y = value, colour = dataset)) +
    geom_boxplot(
      width = 0.5,
      alpha = 0.2,
      outlier.shape = NA
    ) +
    geom_jitter(
      width = 0.10,
      height = 0,
      alpha = 0.75,
      size = 2
    ) +
    labs(
      title = f,
      subtitle = unique(plot_df$feature_group),
      x = NULL,
      y = "Feature value"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(face = "bold")
    )
  
  ggsave(
    filename = file.path(out_dir_features, paste0(safe_filename(f), ".png")),
    plot = p,
    width = 6,
    height = 5,
    dpi = 300
  )
}

# =========================================================
# SUMMARY TABLE
# =========================================================
summary_df <- long_df %>%
  group_by(feature_group, feature, dataset) %>%
  summarise(
    n = n(),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(summary_df, out_file_summary)

cat("Done.\n")
cat("Grouped plots saved to: ", out_dir_groups, "\n", sep = "")
cat("Individual feature plots saved to: ", out_dir_features, "\n", sep = "")
cat("Feature group table saved to: ", out_file_groups, "\n", sep = "")
cat("Summary table saved to: ", out_file_summary, "\n", sep = "")