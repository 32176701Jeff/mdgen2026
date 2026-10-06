# MDGen-2026 相對 base 的改動清單

本文件目前僅依據 `MDGEN2026_REPORT-r1.md`、`MDGEN2026_REPORT-r2.md` 與 `MDGEN2026_REPORT-r2.1.md` 整理，尚未使用 code diff 補充。基準版本為上游 commit `81482a4`。

整理原則：相同的 `file_path` 與 `block_name` 合併為同一列，只記錄 r2.1 的最終結果。r1 新增但在 r2 或 r2.1 完全撤銷的內容不列入，例如 dataset／wrapper／inference 的 position ID 傳遞、強制關閉 TF32、推論入口的 `highest` matmul precision，以及固定使用 float32 建立 NPY。

## 修改的既有檔案

| file_path | block_name | 最終改動 |
|---|---|---|
| `mdgen/dataset.py` | `data-preprocess-fixes` | 修正 frame 不足時的處理錯誤，以及最後一個合法時間窗無法被抽到的問題。 |
| `mdgen/model/latent_model.py` | `sdpa-route` | `LatentMDGenModel`、`IPALayer` 與 `LatentMDGenLayer` 的 `use_sdpa` 預設均為 `True`；內部 attention layer 接收上層指定的路徑。 |
| `mdgen/model/latent_model.py` | `sdpa-time-axis-mask` | time-axis attention 沿用相同的 mask 與 SDPA 轉換流程。 |
| `mdgen/model/latent_model.py` | `position-id-model-routing` | 保留 `position_ids=None` 的模型介面，並可將其傳至 residue-axis attention；正常訓練與推論不提供非 `None` 值。 |
| `mdgen/model/latent_model.py` | `position-id-residue-layout` | 保留將 `B,L` position ID 展開為 residue attention 使用的 `B×T,L` layout；time-axis attention 不使用。 |
| `mdgen/model/latent_model.py` | `no-rope-padding-mask` | 將有效位置 mask 轉為 boolean `key_padding_mask`，使有效 residue 為 `False`、padding 為 `True`，並讓 RoPE／no-RoPE、manual／SDPA 共用相同遮蔽語意。 |
| `mdgen/model/mha.py` | `sdpa-route` | 加入 manual／SDPA routing 與 fallback 記錄；`MultiheadAttention.use_sdpa` 預設為 `True`，`AttentionWithRoPE` 未明確指定時沿用此預設。 |
| `mdgen/model/mha.py` | `sdpa-scaling` | 保留原本的 `q *= self.scaling`，並令 SDPA 使用 `scale=1.0`，避免重複 scaling。 |
| `mdgen/model/mha.py` | `sdpa-mask` | 將 fairseq padding mask 反轉為 SDPA 的保留 mask；mask 沒有實際作用時傳入 `None`。 |
| `mdgen/model/mha.py` | `sdpa-bias-kv` | 保留 `add_bias_kv=True` 的 K／V token append 與 mask 延長流程。 |
| `mdgen/model/mha.py` | `sdpa-dropout` | training 使用原 dropout，evaluation 的 SDPA dropout 設為 `0.0`。 |
| `mdgen/model/mha.py` | `sdpa-layout-output` | SDPA 前將 tensor 轉成 `B,H,L,D`，完成後還原原 layout，並通過既有 `out_proj`。 |
| `mdgen/model/mha.py` | `position-id-rope-interface` | 保留 `position_ids=None` 介面與上游 RoPE 計算；任何非 `None` 值統一在 `MultiheadAttention.forward` 入口拋出 `NotImplementedError`，不受 RoPE 是否啟用影響。 |
| `mdgen/model/mha.py` | `sdpa-diagnostics` | 記錄各 attention module 最後實際使用的 manual／SDPA 路徑及 SDPA fallback reason。 |
| `mdgen/model/primitives.py` | `runtime-compatibility` | 對 legacy FlashAttention 加入 optional import guard，以支援未安裝該套件的 PyTorch 2.x 環境。 |
| `mdgen/tensor_utils.py` | `runtime-compatibility` | 將新版 PyTorch 不接受的 list indexing 改為 tuple indexing。 |
| `mdgen/wrapper.py` | `sdpa-route` | `NewMDGenWrapper.use_sdpa` 的建構預設改為 `True`。 |
| `mdgen/wrapper.py` | `checkpoint-runtime-args` | `save_hyperparameters` 只保存原有 `args`，避免 `use_sdpa` 等 runtime 選項成為 checkpoint 的建構需求。 |
| `mdgen/parsing.py` | `sdpa-route` | 將 `--use_sdpa` 改為 `--manual_attention`；未帶 flag 時使用 SDPA。 |
| `mdgen/parsing.py` | `fp32-training-precision` | 將既有 `--precision` 的允許值限制為 `32-true`。 |
| `mdgen/parsing.py` | `seed-deterministic-args` | `--train_seed` 預設為 `None`、deterministic 預設關閉；保留 seed、deterministic 與 cuDNN benchmark 的顯式 CLI 控制。 |
| `mdgen/parsing.py` | `deterministic-benchmark-guard` | 禁止 deterministic algorithms 與 cuDNN benchmark 同時啟用。 |
| `mdgen/parsing.py` | `runtime-args-separation` | 正式 `train.py` 只註冊一般訓練參數與 attention 路徑參數；backend profiler、peak memory 與 execution time 量測參數只由 `train-runtime.py` 註冊。 |
| `mdgen/parsing.py` | `model-output-dir` | 新增 `--model_dir`，可指定 training 輸出根目錄；未提供時維持 `workdir/<run_name>`。 |
| `mdgen/parsing.py` | `peak_memory` | 提供 runtime training peak-memory JSON 的輸出參數。 |
| `mdgen/parsing.py` | `execution_time` | 提供排除 warm-up 後 sec/step JSON 的輸出參數。 |
| `train.py` | `sdpa-route` | 由 `manual_attention` 反向計算 `use_sdpa`；預設走 SDPA，只有 `--manual_attention` 使用手寫 attention。 |
| `train.py` | `fp32-matmul-precision` | 正式訓練使用上游的 `torch.set_float32_matmul_precision("medium")`；模型參數與輸入輸出仍為 FP32。 |
| `train.py` | `seed-initialization` | 只有明確提供 `--train_seed` 時才呼叫 `pl.seed_everything`。 |
| `train.py` | `deterministic-execution` | 由 Lightning Trainer 的設定控制 deterministic execution。 |
| `train.py` | `deterministic-benchmark-guard` | 保留 deterministic 與 cuDNN benchmark 不可同時啟用的檢查。 |
| `train.py` | `runtime-args-separation` | 正式 training 入口不包含 backend profiler、peak-memory 與 execution-time instrumentation。 |
| `train.py` | `runtime-compatibility` | Trainer 的 validate／fit checkpoint 載入明確使用 `weights_only=False`，以支援可信任的舊 Lightning checkpoint。 |
| `sim_inference.py` | `sdpa-route` | 將 `--use_sdpa` 改為 `--manual_attention`；未帶 flag 時以 SDPA 載入 checkpoint。 |
| `sim_inference.py` | `seed-deterministic-args` | `--inference_seed` 預設為 `None`、deterministic 預設關閉，並保留顯式 CLI 控制。 |
| `sim_inference.py` | `seed-initialization` | 只有明確提供 inference seed 時才初始化 RNG。 |
| `sim_inference.py` | `deterministic-execution` | 依 CLI 設定呼叫 `torch.use_deterministic_algorithms`。 |
| `sim_inference.py` | `deterministic-benchmark-guard` | 禁止 deterministic algorithms 與 cuDNN benchmark 同時啟用。 |
| `sim_inference.py` | `fp32-inference-model` | checkpoint 載入後使用 `model.eval().float()`。 |
| `sim_inference.py` | `runtime-compatibility` | 以 `weights_only=False` 載入可信任的舊 Lightning checkpoint。 |
| `sim_inference.py` | `get_batch`（無獨立 marker） | 讀取 NPY 後檢查 residue 數量是否與 CSV 的 `seqres` 長度一致；不一致時明確拋出 `ValueError`。 |
| `scripts/prep_sims.py` | `data-preprocess-fixes` | 統一使用 `args.atlas_dir`，並讓 `--atlas_dir` 與舊 `--sim_dir` 成為同一個 argparse 選項的 alias。 |
| `scripts/prep_sims.py` | `npy-dtype-option` | 新增 `--dtype {float16,float32}`；預設 `float16`，並依選項決定輸出 NPY dtype。 |
| `tps_inference.py` | `sdpa-route` | 新增 `--manual_attention`，並將對應的 `use_sdpa` 傳入 checkpoint loader；預設使用 SDPA。 |
| `tps_inference.py` | `runtime-compatibility` | checkpoint 載入補上 `weights_only=False`。 |
| `design_inference.py` | `sdpa-route` | 新增 `--manual_attention`，並將對應的 `use_sdpa` 傳入 checkpoint loader；預設使用 SDPA。 |
| `design_inference.py` | `runtime-compatibility` | checkpoint 載入補上 `weights_only=False`。 |
| `upsampling_inference.py` | `sdpa-route` | 新增 `--manual_attention`，並將對應的 `use_sdpa` 傳入 checkpoint loader；預設使用 SDPA。 |
| `upsampling_inference.py` | `runtime-compatibility` | checkpoint 載入補上 `weights_only=False`。 |

## 新增的檔案

| file_path | block_name | 最終內容或用途 |
|---|---|---|
| `.gitignore` | `repository-artifacts` | 排除 checkpoint、workdir、wandb、暫存檔及 `*.npy`、`*.npz`、`*.pdb`、`*.pickle` 等大型產物。 |
| `scripts/prep-protein-csv.py` | `position-id-csv` | 從 PDB 建立 `seqres` 與 `position_ids` CSV；檔頭說明 position IDs 僅供未來使用，r2.1 模型不讀取該欄位。 |
| `train-runtime.py` | `sdpa-route` | 提供預設 SDPA、可用 `--manual_attention` 切換的 training benchmark 入口。 |
| `train-runtime.py` | `fp32-matmul-precision` | 與正式 training 一致使用 `medium` matmul precision。 |
| `train-runtime.py` | `seed-initialization` | 預設不設定 seed；需要固定 benchmark 輸入時可明確提供 `--train_seed`。 |
| `train-runtime.py` | `deterministic-execution` | 由 Lightning Trainer 設定控制 deterministic execution。 |
| `train-runtime.py` | `deterministic-benchmark-guard` | 保留 deterministic 與 cuDNN benchmark 不可同時啟用的檢查。 |
| `train-runtime.py` | `runtime-args-separation` | 集中註冊並使用 backend profiler、peak memory 與 execution time 的量測參數。 |
| `train-runtime.py` | `runtime-compatibility` | Trainer 的 validate／fit checkpoint 載入明確使用 `weights_only=False`。 |
| `train-runtime.py` | `sdpa-diagnostics` | 輸出實際 SDPA operator/backend，並逐一記錄 MHA 的 `last_attention_backend` 與 `last_sdpa_fallback_reason`。 |
| `train-runtime.py` | `peak_memory` | 成功或 OOM 時輸出 CUDA peak allocated／reserved、完成 step、matmul precision、deterministic 狀態與各 MHA backend。 |
| `train-runtime.py` | `execution_time` | 使用 CUDA Event 量測完整 training step，排除前五個 warm-up step後輸出統計值、matmul precision、deterministic 狀態與各 MHA backend。 |
| `test_sdpa.py` | `sdpa-equivalence-test` | 建立 manual／SDPA forward、input gradient 與 parameter gradient 等價性測試；固定 `highest` 並關閉 TF32；涵蓋 no-RoPE padding invariance，以及 RoPE／no-RoPE、manual／SDPA 對非 `None position_ids` 的拒絕；manual 與 SDPA 組均明確指定 `use_sdpa`。 |
| `test_sdpa.py` | `sdpa-upstream-reference` | RoPE residue／time 測試使用固定的上游 legacy manual 實作對照目前 SDPA。 |
| `test_sdpa.py` | `sdpa-time-full-row-padding` | 新增 time-axis 整列 padding case，驗證 output／gradient 誤差與 finite 狀態。 |
| `benchmark_l_scaling.py` | `sdpa-route` | 將 attention CLI 改為 `--manual_attention`；預設量測 SDPA。 |
| `benchmark_l_scaling.py` | `seed-deterministic-args` | `--seed` 預設為 `None`，新增預設關閉的 `--deterministic`。 |
| `benchmark_l_scaling.py` | `seed-initialization` | 只有明確提供 seed 時才初始化 RNG。 |
| `benchmark_l_scaling.py` | `deterministic-execution` | 依 CLI 設定 deterministic algorithms，並將實際狀態寫入 JSON。 |
| `benchmark_l_scaling.py` | `sdpa-diagnostics` | 逐長度記錄實際 attention path、SDPA backend／fallback 與 CUDA peak memory。 |
| `benchmark_l_scaling.py` | `a4-l-scaling` | L-scaling JSON 記錄 matmul precision、deterministic algorithms 實際狀態，以及各 MHA 的 `last_attention_backend` 與 `last_sdpa_fallback_reason`。 |
| `benchmark_l_scaling.py` | `runtime-compatibility`（無獨立 marker） | checkpoint loader 明確使用 `weights_only=False`，以支援可信任的舊 Lightning checkpoint。 |
| `scripts/analyze_ensemble.py` | `third-party-provenance` | 保留 AlphaFlow ensemble 分析工具，並記錄固定來源 commit、改名說明、原作者 copyright 與 MIT License。 |
| `scripts/print.py` | `third-party-provenance` | 保留 AlphaFlow analysis pickle 彙整工具，並記錄固定來源 commit、改名說明、原作者 copyright 與 MIT License。 |
| `mdgen/model/mha_legacy.py` | `sdpa-upstream-reference` | 保存上游 commit `642b95b4740ef889167f433a9155d6ed34ee6a70` 的 manual MHA 實作作為固定 regression baseline；正式模型不匯入。 |
| `my_new/environment-mdgen2026-cu126.yml` | `r1-historical-environment` | 保存 r1 的 cu126 環境，檔頭標記「r1 歷史環境，勿用」。 |
| `my_new/environment-mdgen2026-cu130.yml` | `r1-historical-environment` | 保存 r1 的 cu130 環境，檔頭標記「r1 歷史環境，勿用」。 |
| `my_new/environment-mdgen2026-cu126-pyemma.yml` | `runtime-environment` | 以 cu126 環境加入 PyEMMA 及固定的相關依賴；環境名稱為 `mdgen2026-r2-a6-cu126`，並作為正式環境鎖定檔。 |
| `my_new/environment-mdgen2026-cu130-pyemma.yml` | `runtime-environment` | 以 cu130 環境加入相同的 PyEMMA dependency set；環境名稱為 `mdgen2026-r2-a6-cu130`，供 eigh／Blackwell 追蹤。 |
| `my_new/download_atlas.sh` | 檔案整體 | 新增 ATLAS 資料下載腳本。 |

## 刪除的檔案

無。三份報告只記載部分新增 block 後續遭撤銷，沒有記載相對 base 刪除完整檔案。
