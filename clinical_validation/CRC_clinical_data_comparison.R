
### Library loads ###
library(readxl)
library(tidyverse)
library(seqinr)
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install(version = "3.20")
# BiocManager::install("Biostrings")
# BiocManager::install("pwalign")
library(Biostrings) 
library(pwalign)
library(ggpubr)
library(ggplot2)

### Setup ###
setwd("C:/Users/Alanna/Desktop/Research_Code/Desktop_research/crc_neoant_clinical_validation")  
# change to your working directory, or at least ensure that data files are in the same wd as this script

# You will need to download Supplementary Table 3 from the following paper:
# https://www.nature.com/articles/s41588-023-01499-4
# Mismatch repair deficiency is not sufficient to elicit tumor immunogenicity
# Westcott, P. et al 2023

# Load in TSNAdb CRC dataset:
crc_Neoant_File <- read_excel("41588_2023_1499_MOESM3_ESM.xlsx", sheet=3, #use supplementary_table3
                              skip = 2, col_names = TRUE, col_types = c("text", rep("skip",7), #chromosome
                                                              "text", rep("skip",2), #variant type
                                                              "text", rep("skip",2), #gene name
                                                              "text", rep("skip",3), #HLA allele
                                                              "text", #mutant epitope
                                                              "text", rep("skip",5), #wild-type epitope
                                                              "numeric", rep("skip",6), #tumor VAF
                                                              "numeric", #median mutant score (Kd, in units nM)
                                                              "numeric", rep("skip",4), #median wt mutant score (same)
                                                              "numeric", #CCF
                                                              "text", "skip", #patient ID
                                                              "text", #best response by RECIST
                                                              "numeric", "skip")) #days PFS
# Remove "NE" patient bc what even is that?
crc_Neoant_File <- crc_Neoant_File[crc_Neoant_File$response!='NE',]

# Filter mutation types s.t. we only look at mutations where we have a WT peptide for comparison
idx_of_NAs <- is.na(crc_Neoant_File[9]$Median.WT.Score)
crc_Neoant_File_trimmed <- crc_Neoant_File[!idx_of_NAs,]

# Remove mutations which have NA as their CCF:
idx_of_NAs <- is.na(crc_Neoant_File_trimmed$CCF)
crc_Neoant_File_trimmed <- crc_Neoant_File_trimmed[!idx_of_NAs,]

# Load IEDB epitope dataset:
epitope_table <- read_excel("../epitopes_tidy.xlsx", col_names = TRUE, col_types = c("text", "text", "text"))
epitopes_vec <- epitope_table$Description # epitope AA sequences
epitopes_vec[2586] <- gsub("l", "L", epitopes_vec[2586]) # fix IEDB accidental lower-case L
for (i in 1:length(epitopes_vec)) {
  epitopes_vec[i] <- gsub(" .*", "", epitopes_vec[i]) #remove spaces + additional information from IEDB AA seqs
}

recompute_A = FALSE  # Do you want to compute A from scratch?
recompute_R = FALSE  # Do you want to compute R from scratch?

### Compute or load A ###
if (recompute_A) {
  wt_affinity <- crc_Neoant_File_trimmed[9]$Median.WT.Score
  mt_affinity <- crc_Neoant_File_trimmed[8]$Median.MT.Score
  eps_over_L <- 1/3687 # value from Luksza et al. 2017
  A <- (wt_affinity)/((mt_affinity) * (1 + eps_over_L*(wt_affinity)))
  write.table(A, file='clin_A_data.txt', sep=' ', row.names=FALSE,col.names=FALSE) # write txt file of A values to disk
} else {
  Adf <- read.table(file='clin_A_data.txt', header=FALSE) # read txt file of A values
  A <- Adf$V1
}

### Compute or load R ###
if (recompute_R) {
  data(BLOSUM62) #load in BLOSUM62 sub matrix from Biostrings
  
  # Initialize values before aligning
  align_iter <- 1
  tot_aligns <- length(crc_Neoant_File_trimmed$Chrom)
  k = 4.87 # value from Luksza et al. (ref 15)
  a = 26   # value from Luksza et al. (ref 15)
  R = rep(0,tot_aligns)
  effective_score_nolog = rep(0,tot_aligns)
  fast_epitopes = AAStringSet(epitopes_vec)
  pts_vec <- unique(crc_Neoant_File_trimmed$id)
  num_pts <- length(pts_vec)
  
  # Begin alignment (this will take some time! There are 131,513 x 4058 alignments to be computed)
  for (pt in 1:num_pts) {
    idx <- which(crc_Neoant_File_trimmed$id == pts_vec[pt])
    pt_mut <- AAStringSet(crc_Neoant_File_trimmed$MT.Epitope.Seq[idx])
    num_mut <- length(idx)
    for (i in 1:num_mut) {
      scores = pairwiseAlignment(pattern = fast_epitopes, # pt_epitopes, # pt_epitopes,  # Align epitope sequences corresp. to THIS patient
                                 subject = pt_mut[i],     # to mutant (neoantigenic) sequence i
                                 substitutionMatrix = BLOSUM62, # using the BLOSUM62 substitution matrix
                                 scoreOnly=TRUE)       # and only output the numeric score.
      effective_score_nolog[align_iter] = sum(exp(-k*(a - scores)))
      Z = 1 + effective_score_nolog[align_iter]
      R[align_iter] = (1/Z)*effective_score_nolog[align_iter] # compute R as described in Luksza et al. (ref 15)
      align_iter <- align_iter + 1
      if (align_iter %% 700 == 0) {
        print(paste(round(align_iter/tot_aligns*100, 2), '% Complete', sep=''))
      }
    }
  }
  
  # Record values of R
  write.table(R, file='clin_R_data.txt', sep=' ', row.names=FALSE,col.names=FALSE) # write txt file of A values to disk
} else {
  Rdf <- read.table(file='clin_R_data.txt', header=FALSE) # read txt file of R values
  R <- Rdf$V1
}

### AxR Computation and Plotting ###
AxR = A*R

# Preparing labels containing mutation and gene information
allmuts = crc_Neoant_File_trimmed$Variant.Type   # Mutation description (point mutation)
allgenes = crc_Neoant_File_trimmed$Gene.Name      # In which gene
neoant_labs = rep(NA, length(allmuts))
for (i in 1:length(allmuts)) { 
  neoant_labs[i] = paste('mut',allmuts[i], 'gene', allgenes[i], sep='_')
}
# Set aside MOBSTER input
mobster_input = data.frame(axr = AxR, 
                           labs = neoant_labs,
                           pt = crc_Neoant_File_trimmed$id,
                           VAF = crc_Neoant_File_trimmed$Tumor.DNA.VAF) # bind all neoant scores, labels into one dataframe

# write.table(neoant_labs, 'neoantigen_labels.txt', sep=' ', row.names=FALSE, col.names=FALSE)
neoant_df = data.frame(A, R, AxR, neoant_labs,
                       pfs = crc_Neoant_File_trimmed$PFS_days,
                       ccf = crc_Neoant_File_trimmed$CCF,
                       id=crc_Neoant_File_trimmed$id,
                       response=crc_Neoant_File_trimmed$response) # bind all neoant scores, labels into one dataframe

# Plotting A, R, and AxR distributions (uncomment to plot!)
# pl1 = ggplot(data=neoant_df, aes(x = A)) +
#   geom_histogram(bins=50) +
#   scale_x_log10() +
#   labs(title='Histogram of A', x = "A", y = 'Count') +
#   geom_vline(aes(xintercept=median(A)),
#              color="red", linetype="dashed", linewidth=1)+
#   geom_vline(aes(xintercept=mean(A)),
#              color="blue", linetype="dashed", linewidth=1)
# pl1

# pl2 = ggplot(data=neoant_df, aes(x = AxR)) +
#   geom_histogram(bins=25) +
#   scale_x_log10() +
#   labs(title='Histogram of AxR', x = "AxR", y = 'Count') +
#   geom_vline(aes(xintercept=median(AxR)),
#              color="red", linetype="dashed", linewidth=1)+
#   geom_vline(aes(xintercept=mean(AxR)),
#              color="blue", linetype="dashed", linewidth=1)
# pl2
# 
# pl3 = ggplot(data=neoant_df, aes(x = R)) +
#   geom_histogram(bins=25) +
#   scale_x_log10() +
#   labs(title='Histogram of R', x = "R", y = 'Count') +
#   geom_vline(aes(xintercept=median(R)),
#              color="red", linetype="dashed", linewidth=1)+
#   geom_vline(aes(xintercept=mean(R)),
#              color="blue", linetype="dashed", linewidth=1)
# pl3
# 
# pl3 = ggplot(data=crc_Neoant_File_trimmed, aes(x = CCF)) +
#   geom_histogram(bins=25) +
#   scale_x_log10() +
#   labs(title='Histogram of CCF', x = "CCF", y = 'Count') +
#   geom_vline(aes(xintercept=median(CCF)),
#              color="red", linetype="dashed", linewidth=1)+
#   geom_vline(aes(xintercept=mean(CCF)),
#              color="blue", linetype="dashed", linewidth=1)
# pl3


# Strong neoantigens
summary(crc_Neoant_File_trimmed$CCF[which(AxR>=1)])
ind = which(AxR>=1 & crc_Neoant_File_trimmed$CCF>=0.1) # select strong neoantigens
crc_Neoant_File_trimmed_strong <- crc_Neoant_File_trimmed[ind,]
neoant_df <- neoant_df[ind,]
total_AxR <- AxR # set aside the total AxR values
AxR <- AxR[ind]

# pl4 = ggplot(data=neoant_df, aes(x = AxR)) +
#   geom_histogram() +
#   scale_x_log10() +
#   labs(title='Histogram of AxR>=1', x = "AxR", y = 'Count')
# pl4

# Per-patient mean weighted AxR:
pts <- crc_Neoant_File_trimmed_strong$id
unique_pts <- unique(pts)
num_pts <- length(unique(pts))
clin_df <- data.frame(pt_ID = unique_pts,
                      max_AxR = rep(0, num_pts),
                      mean_AxR = rep(0, num_pts),
                      response = rep('na', num_pts),
                      num_strong_neoant = rep(0, num_pts),
                      num_tot_neoant = rep(0, num_pts),
                      num_strong_clonal_neoant = rep(0, num_pts),
                      num_tot_clonal_neoant = rep(0, num_pts),
                      PFS_days = rep(0, num_pts))
# create data.frame where we have patient ID, mean weighted AxR, max neoant score, PFS days, and best response over course of therapy
for (i in 1:num_pts) {
  idx <- which(pts==unique_pts[i])
  cur_str_AxR <- AxR[idx]
  tot_idx <-which(crc_Neoant_File_trimmed$id==unique_pts[i]) 
  cur_tot_AxR <- total_AxR[tot_idx]
  clin_df$response[i] <- crc_Neoant_File_trimmed_strong$response[idx[1]]
  clin_df$num_tot_neoant[i] <- length(cur_tot_AxR)
  clin_df$num_tot_clonal_neoant[i] <- length(which(crc_Neoant_File_trimmed$CCF[tot_idx]==1))
  clin_df$num_strong_clonal_neoant[i] <- length(which(crc_Neoant_File_trimmed_strong$CCF[idx]==1))
  clin_df$num_strong_neoant[i] <- length(cur_str_AxR)
  clin_df$PFS_days[i] <- crc_Neoant_File_trimmed_strong$PFS_days[idx[1]]
}

clin_df = clin_df[clin_df$response != 'NE',]
neoant_df = neoant_df[neoant_df$response != 'NE',]
clin_df$response = factor(clin_df$response, levels=c('PD', 'SD', 'PR', 'CR'))
neoant_df$response = factor(neoant_df$response, levels=c('PD', 'SD', 'PR', 'CR'))

resp_fun <- function(recist_response) {
  if (recist_response %in% c('PD', 'SD')) {
    output <- 'NR'
  } else if (recist_response %in% c('PR', 'CR')) {
    output <- 'OR'
  }
  return(output)
}

neoant_df$response_group <- factor(sapply(neoant_df$response, resp_fun), 
                                           levels=c('NR', 'OR'))
clin_df$response_group <- factor(sapply(clin_df$response, resp_fun), 
                                 levels=c('NR', 'OR'))

## Let's reconstruct some clonal trees.
tot_tumor_size = 1e5

for (i in 1:num_pts) {
  idx <- which(pts==unique_pts[i])
  cur_AxR <- AxR[idx]
  pt_CCF <- crc_Neoant_File_trimmed_strong$CCF[idx]
  sort_inds = sort(pt_CCF, decreasing=FALSE, index.return=TRUE)$ix
  cur_AxR = cur_AxR[sort_inds]
  pt_CCF = pt_CCF[sort_inds]
  m = length(cur_AxR)
  ICs = c()
  AxRs = c()
  ccf_min <- 0
  while (sum(ICs)<tot_tumor_size && ccf_min <= 0.9) {
    start_ind <- which(pt_CCF>ccf_min)[1] # limit to CCF > tumor already accounted for, to ensure most conservative clonal reconstruction
    if (is.na(start_ind)) {
      break
    }
    max_ind <- which(cur_AxR[start_ind:m] == max(cur_AxR[start_ind:m]))
    if (length(max_ind)>1) {
      max_ind <- max_ind[length(max_ind)]  # if multiple, just pick the max. We already sorted by increasing CCF, so this will be the largest index.
    }
    cur_CCF <- pt_CCF[(start_ind:m)[max_ind]] # what's the CCF of this clone?
    ICs = c(ICs, round(tot_tumor_size*(cur_CCF-ccf_min),2)) # round to nearest cell
    AxRs = c(AxRs, cur_AxR[(start_ind:m)[max_ind]])
    ccf_min <- cur_CCF
  }
  clonal_inds <- which(pt_CCF == 1)
  remainder <- tot_tumor_size - sum(ICs)
  max_clonal_neoant_score <- 0
  if (length(clonal_inds)<1) {
    if (sum(ICs)<tot_tumor_size) {
      AxRs = c(0, AxRs) # place remaining founder cells to first spot for consistency
      ICs = c(remainder, ICs)
    }
    clin_df$num_clonal_neoant[i] <- 0
  } else {
    max_clonal_neoant_score <- max(cur_AxR[clonal_inds])
    if (sum(ICs)<tot_tumor_size) {
      AxRs = c(AxRs, max_clonal_neoant_score)
      ICs = c(ICs, remainder)
    }
  }
  clin_df$mean_AxR[i] <- sum(AxRs*ICs)/tot_tumor_size
  clin_df$max_AxR[i] <- max(AxRs)
  clin_df$max_clonal_neoant[i] <- max_clonal_neoant_score
  print(paste(c('Patient: ', unique_pts[i]), collapse = ' '))
  print(paste(c('Initial conditions:', ICs), collapse = ' '))
  print(paste(c('AxR Immunogenicity Scores: ', AxRs), collapse = ' '))
  print('                    ')
  print('                    ')
}

# SF (a)

pl6 = ggplot(data=clin_df, aes(x = response_group, y=max_clonal_neoant)) +
  geom_boxplot() +
  stat_compare_means() +
  labs(title='Strongest clonal neoantigen', x = "", y = 'AxR')
ggsave('SF3a_strongest_clonal.svg', height=4, width=4)
wt_a = wilcox.test(clin_df$max_clonal_neoant ~ clin_df$response_group)
wt_a$statistic
wt_a$p.value

pl6 = ggplot(data=clin_df, aes(x = response_group, y=mean_AxR)) +
  geom_boxplot() +
  stat_compare_means() +
  labs(title='Weighted mean antigenicity', x = "", y = 'AxR')
ggsave('SF3a_weightedmean.svg', height=4, width=4)
wt_b = wilcox.test(clin_df$mean_AxR ~ clin_df$response_group)
wt_b$statistic
wt_b$p.value

pl6 = ggplot(data=clin_df, aes(x = response_group, y=max_AxR)) +
  geom_boxplot() +
  stat_compare_means() +
  labs(title='Maximal neoantigen quality', x = "", y = 'AxR')
ggsave('SF3a_maxscore.svg', height=4, width=4)
wt_c = wilcox.test(clin_df$max_AxR ~ clin_df$response_group)
wt_c$statistic
wt_c$p.value

# SF 3 (b)
pl6 = ggplot(data=clin_df, aes(x = response_group, y = num_tot_clonal_neoant)) +
  geom_boxplot() +
  stat_compare_means() +
  #ylim(0, 3000) + 
  labs(title='Clonal neoantigenic heterogeneity', x = "", y = 'Number of clonal neoantigens')
ggsave('SF3b_numclonal.svg', height=4, width=4)

pl6 = ggplot(data=clin_df, aes(x = response_group, y = num_strong_clonal_neoant/num_tot_clonal_neoant)) +
  geom_boxplot() +
  stat_compare_means() +
  ylim(0, 0.07) + 
  labs(title='Fraction of strong clonal neoantigens', x = "", y = 'Strong / Total')
ggsave('SF3b_frac_clonal.svg', height=4, width=4)
wt_c = wilcox.test(clin_df$num_strong_clonal_neoant/clin_df$num_tot_clonal_neoant ~ clin_df$response_group)
wt_c$statistic
wt_c$p.value


# Responders:
OR_df <- clin_df[which(clin_df$response_group=='OR'),]
NR_df <- clin_df[which(clin_df$response_group=='NR'),]

# Total numbers of clonal neoant:
print('Total number of clonal neoants for OR:')
summary(OR_df$num_tot_clonal_neoant)
print('Total number of clonal neoants for NR:')
summary(NR_df$num_tot_clonal_neoant)

# Fraction of strong/total clonal neoant:
print('Clonal fraction (strong/total) for OR:')
summary(OR_df$num_strong_clonal_neoant/OR_df$num_tot_clonal_neoant)
print('Clonal fraction (strong/total) for NR:')
summary(NR_df$num_strong_clonal_neoant/NR_df$num_tot_clonal_neoant)


# Additional plots (not shown in paper)
pl6 = ggplot(data=clin_df, aes(x = mean_AxR, y=max_AxR)) +
  geom_smooth(method=lm) +
  geom_point() +
  labs(title='Correlation in neoantigenicity scoring', x = "weighted mean AxR", y = 'maximum AxR')
pl6
