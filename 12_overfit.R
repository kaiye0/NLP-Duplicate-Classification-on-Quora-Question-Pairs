setwd("D:/University/Assignment/2019 - 2020/2019 Fall Semester/Nonlinear Optimization/Project")
A = read.table("feature_train.csv", header=TRUE, sep=",")
b = read.table("train_label.csv", header=TRUE)
label = b[c(1:404290),]
cwc_min = A[c(1:404290),1]
cwc_max = A[c(1:404290),2]
csc_min = A[c(1:404290),3]
csc_max = A[c(1:404290),4]
ctc_min = A[c(1:404290),5]
ctc_max = A[c(1:404290),6]
cww_min = A[c(1:404290),7]
cww_max = A[c(1:404290),8]
csw_min = A[c(1:404290),9]
csw_max = A[c(1:404290),10]
ctw_min = A[c(1:404290),11]
ctw_max = A[c(1:404290),12]
first_eq = A[c(1:404290),13]
lastt_eq = A[c(1:404290),14]
diff_len = A[c(1:404290),15]
mean_len = A[c(1:404290),16]
token_set_ratio = A[c(1:404290),17]
token_sort_ratio = A[c(1:404290),18]
fuzz_ratio = A[c(1:404290),19]
fuzz_partial_ratio = A[c(1:404290),20]
longest_substr_ratio = A[c(1:404290),21]
min_kcore = A[c(1:404290),22]
max_kcore = A[c(1:404290),23]
common_neighbor_count = A[c(1:404290),24]
common_neighbor_ratio = A[c(1:404290),25]
min_freq = A[c(1:404290),26]
max_freq = A[c(1:404290),27]
GLM = glm(label ~ cwc_min+cwc_max+csc_min+csc_max+ctc_min+ctc_max+cww_min+cww_max+csw_min+csw_max+ctw_min+ctw_max+first_eq+lastt_eq+diff_len+mean_len+token_set_ratio+token_sort_ratio+fuzz_ratio+fuzz_partial_ratio+longest_substr_ratio+min_kcore+max_kcore+common_neighbor_count+common_neighbor_ratio+min_freq+max_freq)
aov(GLM)
summary(GLM)