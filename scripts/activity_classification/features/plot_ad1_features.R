library(tidyverse)
library(scales)
library(here)

# =========================================================
# PATHS
# =========================================================
base_dir <- here("results", "activity_classification", "features")
error_file <- file.path(base_dir, "engineered_features/error_all_features.csv")
noerror_file <- file.path(base_dir, "engineered_features/noerror_all_features.csv")
outdir <- file.path(base_dir, "graphs", "ad1_features")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

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

use_log_scale <- function(x) {
  x <- x[is.finite(x)]
  x <- x[!is.na(x)]
  
  if (length(x) < 5) return(FALSE)
  if (any(x <= 0)) return(FALSE)
  
  rng <- max(x) / min(x)
  is.finite(rng) && rng >= 20
}

extract_ad1_family <- function(feature_name) {
  case_when(
    str_detect(feature_name, "^ad1_a_") ~ "AD1-A",
    str_detect(feature_name, "^ad1_b_") ~ "AD1-B",
    str_detect(feature_name, "^ad1_c_") ~ "AD1-C",
    str_detect(feature_name, "^analogue_ad1_a") ~ "AD1-A",
    str_detect(feature_name, "^analogue_ad1_b") ~ "AD1-B",
    str_detect(feature_name, "^analogue_ad1_c") ~ "AD1-C",
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
# FIND AD1 FEATURES
# =========================================================
all_numeric_features <- names(df)[sapply(df, is.numeric)]

ad1_features <- grep(
  "^(ad1_[abc]_.*|analogue_ad1_[abc].*)",
  all_numeric_features,
  value = TRUE
)

ad1_group_info <- tibble(
  feature = ad1_features,
  sensor = extract_ad1_family(ad1_features)
) %>%
  filter(sensor %in% c("AD1-A", "AD1-B", "AD1-C"))

# =========================================================
# SPLIT BUCKET / NON-BUCKET FEATURES
# =========================================================
bucket_features <- grep("_bucket::", ad1_features, value = TRUE)

ad1_nonbucket_group_info <- ad1_group_info %>%
  filter(!feature %in% bucket_features)

ad1_bucket_group_info <- ad1_group_info %>%
  filter(feature %in% bucket_features)

# =========================================================
# PLOT NON-BUCKET AD1 FEATURES
# =========================================================
plot_ad1_sensor_features <- function(
    data,
    sensor_name,
    sensor_features,
    outdir,
    max_facets_per_plot = 12
) {
  if (length(sensor_features) == 0) return(character(0))
  
  saved <- c()
  
  feature_chunks <- split(
    sensor_features,
    ceiling(seq_along(sensor_features) / max_facets_per_plot)
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
      mutate(
        value = ifelse(
          str_detect(feature, "^analogue_") & value == -1,
          NA,
          value
        )
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
      paste(sensor_name, "Features by Activity")
    } else {
      paste(sensor_name, "Features by Activity (Part", i, "of", length(feature_chunks), ")")
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
    
    p <- p + scale_y_continuous(labels = comma)
    
    outfile <- if (length(feature_chunks) == 1) {
      file.path(outdir, paste0("ad1_", safe_name(sensor_name), ".png"))
    } else {
      file.path(
        outdir,
        paste0(
          "ad1_", safe_name(sensor_name),
          "_part_", str_pad(i, 2, pad = "0"), ".png"
        )
      )
    }
    
    ggsave(
      filename = outfile,
      plot = p,
      width = 3.2 * ncol_facets,
      height = max(4.5, 2.5 * nrow_facets),
      dpi = 300
    )
    
    saved <- c(saved, outfile)
  }
  
  saved
}

# =========================================================
# PLOT BUCKET FEATURES SEPARATELY
# =========================================================
plot_ad1_bucket_features <- function(data, sensor_name, sensor_features, outdir) {
  if (length(sensor_features) == 0) return(character(0))
  
  plot_df <- data %>%
    select(activity, dataset, all_of(sensor_features)) %>%
    pivot_longer(
      cols = all_of(sensor_features),
      names_to = "bucket_feature",
      values_to = "value"
    ) %>%
    filter(!is.na(value)) %>%
    mutate(
      bucket = str_extract(bucket_feature, "DUR_[A-Z0-9_]+$"),
      bucket = factor(
        bucket,
        levels = c(
          "DUR_0_60",
          "DUR_61_120",
          "DUR_121_200",
          "DUR_201_320",
          "DUR_321_PLUS"
        )
      )
    )
  
  if (nrow(plot_df) == 0) return(character(0))
  
  p <- ggplot(plot_df, aes(x = bucket, y = value, fill = activity)) +
    geom_boxplot(
      alpha = 0.7,
      outlier.shape = NA,
      position = position_dodge(width = 0.8)
    ) +
    geom_jitter(
      aes(color = dataset),
      width = 0.12,
      height = 0,
      size = 1.4,
      alpha = 0.75
    ) +
    facet_wrap(~ activity, nrow = 1) +
    scale_fill_manual(values = activity_fill) +
    scale_color_manual(values = dataset_col) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = c(0, 1)
    ) +
    labs(
      title = paste(sensor_name, "Duration Bucket Features by Activity"),
      x = "Duration Bucket",
      y = "Value",
      fill = "Activity",
      color = "Dataset"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "right",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  outfile <- file.path(outdir, paste0("ad1_", safe_name(sensor_name), "_buckets.png"))
  
  ggsave(
    filename = outfile,
    plot = p,
    width = 12,
    height = 4.8,
    dpi = 300
  )
  
  outfile
}

# =========================================================
# CREATE FEATURE LISTS
# =========================================================
ad1_a_features <- ad1_nonbucket_group_info %>%
  filter(sensor == "AD1-A") %>%
  arrange(feature) %>%
  pull(feature)

ad1_b_features <- ad1_nonbucket_group_info %>%
  filter(sensor == "AD1-B") %>%
  arrange(feature) %>%
  pull(feature)

ad1_c_features <- ad1_nonbucket_group_info %>%
  filter(sensor == "AD1-C") %>%
  arrange(feature) %>%
  pull(feature)

ad1_a_bucket_features <- ad1_bucket_group_info %>%
  filter(sensor == "AD1-A") %>%
  arrange(feature) %>%
  pull(feature)

ad1_b_bucket_features <- ad1_bucket_group_info %>%
  filter(sensor == "AD1-B") %>%
  arrange(feature) %>%
  pull(feature)

ad1_c_bucket_features <- ad1_bucket_group_info %>%
  filter(sensor == "AD1-C") %>%
  arrange(feature) %>%
  pull(feature)

# =========================================================
# CREATE NON-BUCKET PNGS
# =========================================================
saved_a <- tryCatch(
  plot_ad1_sensor_features(df, "AD1-A", ad1_a_features, outdir),
  error = function(e) {
    message("Skipped AD1-A plots: ", e$message)
    character(0)
  }
)

saved_b <- tryCatch(
  plot_ad1_sensor_features(df, "AD1-B", ad1_b_features, outdir),
  error = function(e) {
    message("Skipped AD1-B plots: ", e$message)
    character(0)
  }
)

saved_c <- tryCatch(
  plot_ad1_sensor_features(df, "AD1-C", ad1_c_features, outdir),
  error = function(e) {
    message("Skipped AD1-C plots: ", e$message)
    character(0)
  }
)

# =========================================================
# CREATE BUCKET PNGS
# =========================================================
saved_a_buckets <- tryCatch(
  plot_ad1_bucket_features(df, "AD1-A", ad1_a_bucket_features, outdir),
  error = function(e) {
    message("Skipped AD1-A bucket plots: ", e$message)
    character(0)
  }
)

saved_b_buckets <- tryCatch(
  plot_ad1_bucket_features(df, "AD1-B", ad1_b_bucket_features, outdir),
  error = function(e) {
    message("Skipped AD1-B bucket plots: ", e$message)
    character(0)
  }
)

saved_c_buckets <- tryCatch(
  plot_ad1_bucket_features(df, "AD1-C", ad1_c_bucket_features, outdir),
  error = function(e) {
    message("Skipped AD1-C bucket plots: ", e$message)
    character(0)
  }
)

# =========================================================
# SUMMARY
# =========================================================
all_saved <- c(
  saved_a,
  saved_b,
  saved_c,
  saved_a_buckets,
  saved_b_buckets,
  saved_c_buckets
)

all_saved <- all_saved[!is.na(all_saved) & nzchar(all_saved)]

cat("Saved", length(all_saved), "plots to:\n", outdir, "\n")