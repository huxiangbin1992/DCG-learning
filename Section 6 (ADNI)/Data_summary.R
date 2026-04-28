library(dplyr) 

df <- read.csv("All_Subjects_Table.csv", header = TRUE, stringsAsFactors = FALSE)

df$PTEDUCAT <- ave(df$PTEDUCAT, df$subject_id, FUN = function(x) {
  non_na_vals <- x[!is.na(x)]
  if (length(non_na_vals) == 0) {
    return(x)
  } else {
    min_val <- min(non_na_vals)
    return(ifelse(is.na(x), min_val, min_val))
  }
})

df$KEYMED <- ave(df$KEYMED, df$subject_id, FUN = function(x) {
  x <- trimws(x)
  x[x == ""] <- NA_character_
  tokens <- unlist(strsplit(paste(x[!is.na(x)], collapse = "|"), "\\|", fixed = FALSE))
  tokens <- trimws(tokens)
  tokens <- tokens[tokens != ""]
  tokens <- unique(tokens)
  if (length(tokens) == 0) {
    return(rep(NA, length(x)))
  }
  nums <- suppressWarnings(as.numeric(tokens))
  if (any(nums > 0, na.rm = TRUE)) {
    keep <- !(tokens == "0" | (!is.na(nums) & nums == 0))
    tokens <- tokens[keep]
    nums   <- nums[keep]
  }
  ord <- order(is.na(nums), nums, tokens)
  tokens_sorted <- tokens[ord]
  out <- paste(tokens_sorted, collapse = "|")
  rep(out, length(x))
})
ids <- unique(df$subject_id)
for (id in ids) {
  idx <- which(df$subject_id == id)
  if (all(is.na(df$KEYMED[idx]))) {
    meds <- tolower(trimws(df$CMMED[idx]))
    meds <- meds[!is.na(meds) & meds != ""]
    if (length(meds) == 0) next
    codes <- c()
    # 1 Aricept
    if (any(grepl("aricept|donepezil", meds))) codes <- c(codes, "1")
    # 3 Exelon
    if (any(grepl("exelon", meds))) codes <- c(codes, "3")
    # 4 Namenda
    if (any(grepl("namenda|memantine", meds))) codes <- c(codes, "4")
    # 5 Razadyne
    if (any(grepl("razadyne|galantamine", meds))) codes <- c(codes, "5")
    # 6 Anti-depressant
    if (any(grepl("sertraline|zoloft|lexapro|citalopram|prozac|trazodone", meds)))
      codes <- c(codes, "6")
    # 7 Other behavioral
    if (any(grepl("gabapentin|melatonin", meds)))
      codes <- c(codes, "7")
    if (length(codes) > 0) {
      code_str <- paste(sort(unique(codes)), collapse = "|")
      df$KEYMED[idx] <- code_str
    }
  }
}
df$KEYMED[is.na(df$KEYMED)] <- "0"
df$CMMED <- NULL

df$APOE4_count <- sapply(df$GENOTYPE, function(x) {
  if (is.na(x) || x == "") return(NA_integer_)
  alleles <- unlist(strsplit(x, "/"))
  sum(alleles == "4")
})
df$GENOTYPE <- NULL
df$APOE4_count <- ave(df$APOE4_count, df$subject_id, FUN = function(x) {
  non_na_vals <- x[!is.na(x)]
  if (length(non_na_vals) == 0) {
    return(rep(NA_integer_, length(x)))
  }
  uniq_vals <- unique(non_na_vals)
  if (length(uniq_vals) == 1) {
    return(rep(uniq_vals, length(x)))
  } else {
    max_val <- max(non_na_vals)
    return(rep(max_val, length(x)))
  }
})

df$diff_score <- NA_real_
visit2 <- trimws(tolower(as.character(df$visit)))
m_num <- suppressWarnings(as.integer(sub("^m0*", "", visit2)))
m_num[!grepl("^m0*[0-9]+$", visit2)] <- NA_integer_
ids <- unique(df$subject_id)
for (id in ids) {
  idx <- which(df$subject_id == id)
  idx_sc <- idx[visit2[idx] == "sc"]
  if (length(idx_sc) == 0) {         
    df$diff_score[idx] <- NA_real_
    next
  }
  sc_score <- df$MMSCORE[idx_sc[1]]
  if (is.na(sc_score)) {             
    df$diff_score[idx] <- NA_real_
    next
  }
  idx_m <- idx[!is.na(m_num[idx]) & !is.na(df$MMSCORE[idx])]
  if (length(idx_m) == 0) {
    diff_val <- 0
  } else {
    idx_minm <- idx_m[which.min(m_num[idx_m])]
    mmin_score <- df$MMSCORE[idx_minm]
    diff_val <- mmin_score - sc_score
  }
  df$diff_score[idx] <- diff_val
}

df$BCPREDX <- ave(df$BCPREDX, df$subject_id, FUN = function(x) {
  non_na_vals <- x[!is.na(x)]
  if (length(non_na_vals) == 0) {
    return(rep(NA, length(x)))
  }
  uniq_vals <- unique(non_na_vals)
  if (length(uniq_vals) == 1) {
    return(rep(uniq_vals, length(x)))
  }
  min_val <- min(non_na_vals)
  return(rep(min_val, length(x)))
})
all_na_id <- tapply(is.na(df$BCPREDX), df$subject_id, all)
ids_fix <- names(all_na_id)[all_na_id]
for (id in ids_fix) {
  idx <- which(df$subject_id == id)
  v <- tolower(df$visit[idx])
  idx_sc <- idx[v == "sc"]
  val <- NA
  if (length(idx_sc) > 0) {
    d <- df$DIAGNOSIS[idx_sc[1]]
    if (!is.na(d) && d != "") val <- d
  }
  if (is.na(val) || val == "") {
    idx_bl <- idx[v == "bl"]
    if (length(idx_bl) > 0) {
      d <- df$DIAGNOSIS[idx_bl[1]]
      if (!is.na(d) && d != "") val <- d
    }
  }
  if (!is.na(val) && val != "") {
    if (is.numeric(df$BCPREDX)) {
      df$BCPREDX[idx] <- suppressWarnings(as.numeric(val))
    } else {
      df$BCPREDX[idx] <- as.character(val)
    }
  }
}
df$DIAGNOSIS <- NULL

df <- df[df$visit == "sc", ]
df$visit <- NULL

load("ADNI_ROI_data.RData")
# overlap <- intersect( test_SubjectID ,df$subject_id)
overlap <- intersect(union(train_SubjectID, test_SubjectID),df$subject_id)
df_data <- df[df$subject_id %in% overlap, ]
df_data$PTGENDER[which(df_data$PTGENDER == 2)] <- 0 
minmax01 <- function(x){ (x - min(x, na.rm = TRUE)) /  (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)) }
df_data$entry_age <- minmax01(df_data$entry_age)
df_data$PTEDUCAT  <- minmax01(df_data$PTEDUCAT)
df_data$diff_score  <- minmax01(df_data$diff_score)
keymed_char <- as.character(df_data$KEYMED)
df_data$KEYMED_1 <- 0
df_data$KEYMED_2 <- 0
for (i in seq_along(keymed_char)) {
  x <- keymed_char[i]
  if (is.na(x) || trimws(x) == "" || x == "0") {next}
  meds <- as.numeric(unlist(strsplit(x, "\\|")))
  ad <- any(meds %in% c(1,3,4,5))
  other <- any(meds %in% c(6,7))
  if (ad && !other) {
    df_data$KEYMED_1[i] <- 1
    df_data$KEYMED_2[i] <- 0
  } else if (ad && other) {
    df_data$KEYMED_1[i] <- 0
    df_data$KEYMED_2[i] <- 1
  } else {
    df_data$KEYMED_1[i] <- 0
    df_data$KEYMED_2[i] <- 0
  }
}
df_data$KEYMED <- NULL
df_data$BCPREDX_1 <- ifelse(df_data$BCPREDX == 2, 1, 0)
df_data$BCPREDX_2 <- ifelse(df_data$BCPREDX == 3, 1, 0)
df_data$BCPREDX <- NULL
df_data$APOE4_count1 <- ifelse(df_data$APOE4_count == 1, 1, 0)
df_data$APOE4_count2 <- ifelse(df_data$APOE4_count == 2, 1, 0)
df_data$APOE4_count <- NULL
df_data$diff_score[df_data$subject_id == "016_S_4952"] <- 2

save(df_data, file = "ADNI_Tabular_data.RData")
