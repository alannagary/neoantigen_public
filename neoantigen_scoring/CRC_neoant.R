### Library loads ###
library(readxl)
library(tidyverse)
library(seqinr)
library(Biostrings) 
# To get Biostrings, you should install BiocManager, then run BiocManager::install("Biostrings")
library(ggplot2)

### Setup ###
setwd("C:/Users/Alanna/Desktop/Research_Code/Desktop_research")  
# change to your working directory, or at least ensure that data files are in the same wd as this script

# Load in TSNAdb CRC dataset:
crc_Neoant_File <- read_excel("Colorectal_4_0_excel.xlsx",
                     col_names = TRUE, col_types = c("text", "text", "text", "text", "text","text",
                                                     "numeric", "numeric", "numeric", "numeric",
                                                     "text", "text",
                                                     "numeric", "numeric",
                                                     "text", "text",
                                                     'numeric','numeric',
                                                     "text",
                                                     "numeric",
                                                     "text"))

# Load in IEDB epitope set (see Methods for description of search criteria)
epitope_table <- read_excel("epitopes_tidy.xlsx", col_names = TRUE, col_types = c("text", "text", "text"))
crc_neoants <- crc_Neoant_File[,c(3, 13, 17)] 
epitopes_vec <- epitope_table$Description # epitope AA sequences
epitopes_vec[2586] <- gsub("l", "L", epitopes_vec[2586]) # fix IEDB accidental lower-case L
for (i in 1:length(epitopes_vec)) {
  epitopes_vec[i] <- gsub(" .*", "", epitopes_vec[i]) #remove spaces + additional information from IEDB AA seqs
}
data(BLOSUM62) #load in BLOSUM62 sub matrix from Biostrings

recompute_A = FALSE  # Do you want to compute A from scratch?
recompute_R = FALSE  # Do you want to compute R from scratch?

### Compute or load A ###
if (recompute_A) {
  eps_over_L <- 1/3687 # value from Luksza et al. 2017
  A <- (crc_neoants[,2])/((crc_neoants[,3]) * (1 + eps_over_L*(crc_neoants[,2])))
  A <- A$wild_Affinity # remove the "wild_affinity" label for A
  write.table(A, file='A_data.txt', sep=' ', row.names=FALSE,col.names=FALSE) # write txt file of A values to disk
} else {
  Adf <- read.table(file='A_data.txt', header=FALSE) # read txt file of A values
  A <- Adf$V1
}

### Compute or load R ###
if (recompute_R) {
  # Initialize values before aligning
  mut = AAStringSet(crc_Neoant_File$mut_peptide)    # mutated peptide sequences, AAString for fast computation
  fast_epitopes = AAStringSet(epitopes_vec)         # epitopes vec as AAStringSet for fast computations
  n = length(mut)         # number of neoantigen peptide sequences to check
  m = length(fast_epitopes)  # number of epitopes to align
  k = 4.87 # value from Luksza et al. (ref 15)
  a = 26   # value from Luksza et al. (ref 15)
  R = rep(0,n)
  effective_score_nolog = rep(0,n)
  
  # Begin alignment (this will take some time! There are 131,513 x 4058 alignments to be computed)
  for (i in 1:n) {
    scores = pairwiseAlignment(pattern = fast_epitopes,  # Align epitope sequences 1-4058
                               subject = mut[i],     # to mutant (neoantigenic) sequence i
                               substitutionMatrix = BLOSUM62, # using the BLOSUM62 substitution matrix
                               scoreOnly=TRUE)       # and only output the numeric score.
    effective_score_nolog[i] = sum(exp(-k*(a - scores)))
    Z = 1 + effective_score_nolog[i]
    R[i] = (1/Z)*effective_score_nolog[i] # compute R as described in Luksza et al. (ref 15)
    if (i %% 1000 == 0) {
      print(paste(round(i/n*100, 2), '% Complete', sep=''))
    }
  }
  
  # Record values of R
  write.table(R, file='R_data.txt', sep=' ', row.names=FALSE,col.names=FALSE) # write txt file of A values to disk
} else {
  Rdf <- read.table(file='R_data.txt', header=FALSE) # read txt file of R values
  R <- Rdf$V1
}

### AxR Computation and Plotting ###
AxR = A*R
# write.table(AxR_data, file='AxR_data.txt', sep=' ', row.names=FALSE,col.names=FALSE)

# Preparing labels containing mutation and gene information
allmuts = crc_Neoant_File$mutation   # Mutation description (point mutation)
allgenes = crc_Neoant_File$gene      # In which gene
neoant_labs = rep(NA, length(allmuts))
for (i in 1:length(allmuts)) { 
  neoant_labs[i] = paste('mut',allmuts[i], 'gene', allgenes[i], sep='_')
}
# write.table(neoant_labs, 'neoantigen_labels.txt', sep=' ', row.names=FALSE, col.names=FALSE)
neoant_df = data.frame(A, R, AxR, neoant_labs) # bind all neoant scores, labels into one dataframe

# Plotting
pl1 = ggplot(data=neoant_df, aes(x = A)) +
  geom_histogram(bins=50) +
  scale_x_log10() +
  labs(title='Histogram of A', x = "A", y = 'Count') +
  geom_vline(aes(xintercept=median(A)),
             color="red", linetype="dashed", linewidth=1)+
  geom_vline(aes(xintercept=mean(A)),
             color="blue", linetype="dashed", linewidth=1)
pl1

pl2 = ggplot(data=neoant_df, aes(x = AxR)) +
  geom_histogram(bins=25) +
  scale_x_log10() +
  labs(title='Histogram of AxR', x = "AxR", y = 'Count') +
  geom_vline(aes(xintercept=median(AxR)),
             color="red", linetype="dashed", linewidth=1)+
  geom_vline(aes(xintercept=mean(AxR)),
             color="blue", linetype="dashed", linewidth=1)
pl2

pl3 = ggplot(data=neoant_df, aes(x = R)) +
  geom_histogram(bins=25) +
  scale_x_log10() +
  labs(title='Histogram of R', x = "R", y = 'Count') +
  geom_vline(aes(xintercept=median(R)),
             color="red", linetype="dashed", linewidth=1)+
  geom_vline(aes(xintercept=mean(R)),
             color="blue", linetype="dashed", linewidth=1)
pl3

ind = which(AxR>=1)
strong_df = neoant_df[ind,]
num_mut_events = length(unique(strong_df$neoant_labs))


pl4 = ggplot(data=strong_df, aes(x = AxR)) +
  geom_histogram(bins=45) +
  scale_x_log10() +
  labs(title='Histogram of AxR>=1', x = "AxR", y = 'Count')
pl4

# Total number of unique mutations present in the subset of good neoants:
length(unique(neoant_labs[which(AxR>=1)])) # 2605

# Information about the neoantigens present in this group:
labeltable = table(neoant_labs[which(AxR>=1)])
sorted_labtab = sort(labeltable, decreasing=TRUE)
print(head(sorted_labtab)) # most frequent recurring strong neoantigens
print(sum(labeltable>1)/length(labeltable) * 100) # percent neoantigens occurring more than 1x
most_repeated_inds <- which(neoant_labs==rownames(sorted_labtab)[1])
vals <- neoant_df[most_repeated_inds,]
print(vals) # of the mutation appearing most frequently, what are its A, R values?
# A values can be very different depending on HLA (orders of magnitude)
# Why are R values different?
print(crc_Neoant_File$mut_peptide[most_repeated_inds])
# because some peptides include an additional AA at beginning/end of sequence.

### Look at indels, fusions from TSNAdb v2.0 ###
indel_raw_data = read.table('INDEL_Colorectal_modified_header.txt', header=TRUE)
fusion_raw_data = read.table('Fusion_Colorectal_modified_header.txt', header=TRUE)
fast_epitopes = AAStringSet(epitopes_vec)         # epitopes vec as AAStringSet for fast computations

recompute_indel_fusion_R = FALSE # Do you want to recompute R for fusion, indel mutations?

if (recompute_indel_fusion_R) {
  # Indel computation of R:
  indel = AAStringSet(indel_raw_data$Peptide)    # mutated peptide sequences, AAString for fast computation
  n = length(indel)         # number of neoantigen peptide sequences to check
  m = length(fast_epitopes)  # number of epitopes to align
  k = 4.87 # value from Luksza et al. (ref 15)
  a = 26   # value from Luksza et al. (ref 15)
  R_indel = rep(0,n)
  effective_score_nolog_indel = rep(0,n)
  for (i in 1:n) {
    scores = pairwiseAlignment(pattern = fast_epitopes,  # Align epitope sequences 1-4058
                               subject = indel[i],     # to mutant (neoantigenic) sequence i
                               substitutionMatrix = BLOSUM62, # using the BLOSUM62 substitution matrix
                               scoreOnly=TRUE)       # and only output the numeric score.
    effective_score_nolog_indel[i] = sum(exp(-k*(a - scores)))
    Z = 1 + effective_score_nolog_indel[i]
    R_indel[i] = (1/Z)*effective_score_nolog_indel[i] # compute R as described in Luksza et al. (ref 15)
    if (i %% 100 == 0) {
      print(paste(round(i/n*100, 2), '% Complete', sep=''))
    }
  }
  
  # Fusion computation of R:
  fusion = AAStringSet(fusion_raw_data$Peptide)    # mutated peptide sequences, AAString for fast computation
  n = length(fusion)         # number of neoantigen peptide sequences to check
  m = length(fast_epitopes)  # number of epitopes to align
  k = 4.87 # value from Luksza et al. (ref 15)
  a = 26   # value from Luksza et al. (ref 15)
  R_fusion = rep(0,n)
  effective_score_nolog_fusion = rep(0,n)
  for (i in 1:n) {
    scores = pairwiseAlignment(pattern = fast_epitopes,  # Align epitope sequences 1-4058
                               subject = fusion[i],     # to mutant (neoantigenic) sequence i
                               substitutionMatrix = BLOSUM62, # using the BLOSUM62 substitution matrix
                               scoreOnly=TRUE)       # and only output the numeric score.
    effective_score_nolog_fusion[i] = sum(exp(-k*(a - scores)))
    Z = 1 + effective_score_nolog_fusion[i]
    R_fusion[i] = (1/Z)*effective_score_nolog_fusion[i] # compute R as described in Luksza et al. (ref 15)
    if (i %% 10 == 0) {
      print(paste(round(i/n*100, 2), '% Complete', sep=''))
    }
  }
  write.table(R_indel, file='R_indel.txt', sep=' ', row.names=FALSE,col.names=FALSE)
  write.table(R_fusion, file='R_fusion.txt', sep=' ', row.names=FALSE,col.names=FALSE)
} else {
  R_indel <- read.table('R_indel.txt', header=FALSE)
  R_fusion <- read.table('R_fusion.txt', header=FALSE)
  R_indel <- R_indel$V1
  R_fusion <- R_fusion$V1
}


pl_indel = ggplot(data=data.frame(R_indel), aes(x = R_indel)) +
  geom_histogram(bins=25) +
  scale_x_log10() +
  labs(title='Histogram of R (indels only)', x = "R", y = 'Count') +
  geom_vline(aes(xintercept=median(R_indel)),
             color="red", linetype="dashed", linewidth=1)+
  geom_vline(aes(xintercept=mean(R_indel)),
             color="blue", linetype="dashed", linewidth=1)
pl_indel

pl_fusion = ggplot(data=data.frame(R_fusion), aes(x = R_fusion)) +
  geom_histogram(bins=25) +
  scale_x_log10() +
  labs(title='Histogram of R (fusion only)', x = "R", y = 'Count') +
  geom_vline(aes(xintercept=median(R_fusion)),
             color="red", linetype="dashed", linewidth=1)+
  geom_vline(aes(xintercept=mean(R_fusion)),
             color="blue", linetype="dashed", linewidth=1)
pl_fusion

# Are neoantigens arising from SNVs (TSNAdb 1.0) less neoantigenic (in T cell recognition, R) than indels and fusions (TSNAdb 2.0)?
R_value <- c(R,R_indel,R_fusion)
mut_type <- c(rep('SNV', length(R)), rep('Indel',length(R_indel)), rep('Fusion', length(R_fusion)))
R_df <- data.frame(factor(mut_type), R_value)

# Plot distribution comparisons
library(ggpubr)

pl_violin = ggplot(data=R_df, aes(x = mut_type, y=R_value)) +
  geom_violin() +
  scale_y_log10() +
  stat_compare_means() +
  labs(x='Mutation type', y='R')
pl_violin

pl_box = ggplot(data=R_df, aes(x = mut_type, y=R_value)) +
  geom_boxplot() +
  scale_y_log10() +
  stat_compare_means() +
  labs(x='Mutation type', y='R')
pl_box
