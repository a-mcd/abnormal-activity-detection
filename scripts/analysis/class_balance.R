library(tidyverse)
library(here)

# ----------------------------
# CONFIG
# ----------------------------
base_dir <- here("data", "raw")
dir_error <- file.path(base_dir, "adl_error")
dir_noerror <- file.path(base_dir, "adl_noerror")
outdir <- here("results", "analysis", "graphs")

# ----------------------------
# Extract activity from filename
# ----------------------------
parse_activity <- function(filename) {
  m <- stringr::str_match(filename, "(?i)(?:^|[._-])(t[1-5])(?:[._-]|$)")
  tolower(m[,2])
}

# ----------------------------
# Extract participant from filename
# ----------------------------
parse_participant <- function(filename) {
  m <- stringr::str_match(filename, "(?i)(p[0-9]+)")
  tolower(m[,2])
}

# ----------------------------
# Get episode metadata
# ----------------------------
get_episode_counts <- function(dir_path, dataset_label) {
  
  files <- list.files(dir_path, pattern = "\\.csv$", full.names = TRUE)
  
  tibble(
    filename = basename(files),
    activity = parse_activity(filename),
    participant = parse_participant(filename),
    dataset = dataset_label
  )
}

noerror_df <- get_episode_counts(dir_noerror, "adl_noerror")
error_df <- get_episode_counts(dir_error, "adl_error")

df <- bind_rows(noerror_df, error_df)

# ----------------------------
# Count episodes
# ----------------------------
episode_counts <- df %>%
  group_by(dataset, activity) %>%
  summarise(episodes = n(), .groups = "drop")


# ----------------------------
# Count participants
# ----------------------------
participant_counts <- df %>%
  group_by(dataset, participant) %>%
  summarise(sessions = n(), .groups = "drop")

df %>%
  filter(is.na(participant)) %>%
  select(filename) %>%
  print(n = Inf)

# ----------------------------
# Bar chart ep count
# ----------------------------
p <- ggplot(episode_counts,
       aes(x = activity, y = episodes, fill = dataset)) +
  
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.7) +
  
  geom_text(aes(label = episodes),
            position = position_dodge(width = 0.8),
            vjust = -0.3,
            size = 4) +
  
  labs(
    title = "Number of Episodes per Activity",
    x = "Activity",
    y = "Number of Episodes",
    fill = "Dataset"
  ) +
  
  scale_fill_manual(values = c("adl_noerror" = "#1f78b4",
                               "adl_error" = "#e31a1c")) +
  
  theme_minimal(base_size = 14)

print(p)

# Create output directory if it doesn't exist
if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}

ggsave(
  filename = file.path(outdir, "episodes_per_activity.png"),
  plot = p,
  width = 8,
  height = 6,
  dpi = 300
)



# ----------------------------
# Bar chart participant count
# ----------------------------
p_participant <- ggplot(participant_counts,
                        aes(x = participant, y = sessions, fill = dataset)) +
  
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.7) +
  
  geom_text(aes(label = sessions),
            position = position_dodge(width = 0.8),
            vjust = -0.3,
            size = 3) +
  
  labs(
    title = "Number of Sessions per Participant",
    x = "Participant",
    y = "Number of Sessions",
    fill = "Dataset"
  ) +
  
  scale_fill_manual(values = c("adl_noerror" = "#1f78b4",
                               "adl_error" = "#e31a1c")) +
  
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # rotate labels

print(p_participant)

ggsave(
  filename = file.path(outdir, "sessions_per_participant.png"),
  plot = p_participant,
  width = 10,
  height = 6,
  dpi = 300
)

