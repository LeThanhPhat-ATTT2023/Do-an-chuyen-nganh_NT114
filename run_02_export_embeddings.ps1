$PY = "D:\v\nt114\Scripts\python.exe"
$ROOT = $PSScriptRoot

& $PY -u -m graphslm_ids.offline_path.training.export_student_embeddings `
    --payload-npy   "data/interim/payload_dataset_14gb/payload_256.npy" `
    --checkpoint    "outputs/student_cnn/student_cnn_best.pt" `
    --output-path   "data/processed/student_embeddings_14gb.npy" `
    --batch-size    2048 `
    --dropout       0.1 `
    --device        auto `
    --l2-normalize
