#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(zellkonverter)
  library(MuSiC)
  library(Biobase)
})

option_list <- list(
  make_option("--h5ad", type = "character", help = "Path to scRNA h5ad reference"),
  make_option("--bulk-csv", type = "character", help = "Bulk CSV exported from prepared data"),
  make_option("--truth-csv", type = "character", help = "Ground-truth proportions CSV"),
  make_option("--out-dir", type = "character", default = "DECONOMIX_MODELS/results/benchmarks/music"),
  make_option("--max-cells-per-ct", type = "integer", default = 2000)
)
opt <- parse_args(OptionParser(option_list = option_list))

out_dir <- opt$`out-dir`
bulk_csv <- opt$`bulk-csv`
truth_csv <- opt$`truth-csv`
h5ad_path <- opt$h5ad
max_cells_per_ct <- opt$`max-cells-per-ct`

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("[MuSiC] Loading scRNA reference")
sce <- readH5AD(h5ad_path)
if (!"cell_type" %in% colnames(colData(sce))) {
  stop("cell_type column not found in scRNA obs")
}

ct_vec <- as.character(colData(sce)$cell_type)
idx_list <- split(seq_len(ncol(sce)), ct_vec)
keep_cols <- unlist(lapply(idx_list, function(ix) if (length(ix) > max_cells_per_ct) sample(ix, max_cells_per_ct) else ix))
sce <- sce[, keep_cols, drop = FALSE]
message("[MuSiC] Using ", ncol(sce), " reference cells after capping")

bulk_df <- read.csv(bulk_csv, check.names = FALSE)
truth_df <- read.csv(truth_csv, check.names = FALSE)

assays_available <- assayNames(sce)
assay_name <- if ("counts" %in% assays_available) "counts" else if ("X" %in% assays_available) "X" else assays_available[1]
message("[MuSiC] Using assay=", assay_name)

ref_mat_all <- assay(sce, assay_name)
common_genes <- intersect(rownames(ref_mat_all), colnames(bulk_df))
if (length(common_genes) == 0) {
  stop("No overlapping genes between reference and bulk data")
}

ref_mat <- as.matrix(ref_mat_all[common_genes, , drop = FALSE])
pheno <- data.frame(cell_type = as.character(colData(sce)$cell_type))
rownames(pheno) <- colnames(ref_mat)
ref_eset <- ExpressionSet(assayData = ref_mat, phenoData = AnnotatedDataFrame(pheno))

bulk_mat <- t(as.matrix(bulk_df[, common_genes, drop = FALSE]))
rownames(bulk_mat) <- common_genes
bulk_eset <- ExpressionSet(assayData = bulk_mat)

message("[MuSiC] Running music_prop")
res <- music_prop(bulk.eset = bulk_eset, sc.eset = ref_eset, clusters = "cell_type", verbose = TRUE)
pred <- as.data.frame(t(res$Est.prop.weighted))

shared_cols <- intersect(colnames(truth_df), colnames(pred))
pred <- pred[, shared_cols, drop = FALSE]
truth <- truth_df[, shared_cols, drop = FALSE]

calc_spearman <- function(a, b) {
  out <- c()
  for (nm in colnames(a)) {
    rho <- suppressWarnings(cor(a[[nm]], b[[nm]], method = "spearman"))
    if (is.na(rho)) rho <- 0
    out <- c(out, rho)
  }
  out
}

spearman <- calc_spearman(truth, pred)
mae <- colMeans(abs(as.matrix(truth) - as.matrix(pred)))
metrics <- data.frame(
  cell_type = shared_cols,
  spearman = spearman,
  mae = mae,
  avg_prop = colMeans(truth)
)
metrics <- rbind(
  metrics,
  data.frame(
    cell_type = "AVERAGE",
    spearman = mean(metrics$spearman),
    mae = mean(metrics$mae),
    avg_prop = mean(metrics$avg_prop)
  )
)

write.csv(pred, file.path(opt$out_dir, "pred_props.csv"), row.names = FALSE)
write.csv(metrics, file.path(opt$out_dir, "performance.csv"), row.names = FALSE)
write.csv(pred, file.path(out_dir, "pred_props.csv"), row.names = FALSE)
write.csv(metrics, file.path(out_dir, "performance.csv"), row.names = FALSE)
message("[MuSiC] Saved results to ", out_dir)
