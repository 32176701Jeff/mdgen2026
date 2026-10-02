# MDGen-2026 r1 Review Reply

- Reviewed tag: `mdgen2026-r1`
- Reviewed commit: `c0ee7cb6726c4dc1c40a15203127ef64cbf13528`
- Target tag: `mdgen2026-r2`

## 4. 詳細回覆與驗證證據

### M1 — position_ids

處置：程式修改完成，cluster runtime 驗證待執行。本輪不實作 explicit `position_ids` 的 RoPE 語意；正常訓練與推論不再把 CSV 的 `position_ids` 傳入模型。模型介面維持 `position_ids=None`，若 RoPE 收到非 `None` 值則明確拋出 `NotImplementedError`，避免輸入被靜默忽略。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/dataset.py` | `position-id-dataset`（r1 已存在；r2 刪除） | 不再解析、驗證或回傳 CSV 的 `position_ids`。CSV 可保留該欄位，但本輪 dataset 會忽略。 |
| `mdgen/dataset.py` | `position-id-crop-padding`（r1 已存在；r2 刪除） | 移除 position ID 的 crop 與 padding；residue、mask 與其他既有資料處理維持不變。 |
| `mdgen/wrapper.py` | `position-id-wrapper`（r1 已存在；r2 刪除） | 移除三處由 batch 傳入 `model_kwargs` 的 `position_ids`；模型使用預設值 `None`。 |
| `sim_inference.py` | `position-id-inference-input`（r1 已存在；r2 刪除） | 不再從 inference CSV 解析 `position_ids`。 |
| `sim_inference.py` | `position-id-inference-routing`（r1 已存在；r2 刪除） | 移除函式參數、batch 欄位與 inference model routing；保留 NPY residue 數量與 `seqres` 長度驗證。 |
| `mdgen/model/mha.py` | `position-id-rope-interface`（r1 已存在；r2 修改並保留） | 保留 `position_ids=None` 介面與上游 RoPE 計算；收到非 `None` 時拋出 `NotImplementedError`。 |
| `scripts/prep-protein-csv.py` | `position-id-csv`（r1 已存在；r2 保留） | 保留 CSV 產生功能，於檔頭新增 r2 說明：position IDs 僅供未來使用，MDGen-2026 r2 模型不會讀取。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；r2 修改並保留） | 新增 manual 與 SDPA 收到非 `None` position IDs 時都必須拋出 `NotImplementedError` 的 regression case；`None` 路徑維持上游 RoPE 行為。 |

### M2 — matmul precision

處置：程式修改完成，C2／C5／C6 的 sec/step 待 cluster 重測。正式訓練恢復上游 MDGen 2024 的 `medium` matmul precision；推論與 A4 L-scaling 不再由入口程式覆寫全域 matmul／TF32 設定；SDPA 數值等價測試則維持 `highest` 與 TF32 關閉。

| file_path | block_name | 說明改動 |
|---|---|---|
| `train.py` | `fp32-matmul-precision`（r1 已存在；r2 修改並保留） | 將 `torch.set_float32_matmul_precision` 由 `highest` 改回上游的 `medium`。模型參數、輸入與輸出 dtype 仍為 FP32。 |
| `train.py` | `fp32-disable-tf32`（r1 已存在；r2 刪除） | 刪除 `cuda.matmul.allow_tf32=False` 與 `cudnn.allow_tf32=False`，不再抵消上游 `medium` 設定。 |
| `train-runtime.py` | `fp32-matmul-precision`（r1 已存在；r2 修改並保留） | 與正式訓練入口一致，固定使用 `medium`，使 runtime benchmark 代表正式訓練設定。 |
| `train-runtime.py` | `fp32-disable-tf32`（r1 已存在；r2 刪除） | 刪除兩個強制關閉 TF32 的 backend flag。 |
| `sim_inference.py` | `fp32-matmul-precision`（r1 已存在；r2 刪除） | 移除推論入口的 `highest` 全域覆寫，恢復上游推論行為。 |
| `sim_inference.py` | `fp32-disable-tf32`（r1 已存在；r2 刪除） | 移除推論入口的兩個 TF32 backend flag。 |
| `benchmark_l_scaling.py` | 無獨立 block name（r1 直接設定；r2 刪除） | 移除 `highest` 與兩個 TF32 backend flag，使 A4 不額外注入 precision policy。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；r2 保留不改） | 固定 `highest` 並關閉 TF32，排除內部低精度路徑，讓 manual／SDPA 的數值等價測試維持最高精度基準。 |

### M3 — `--no_rope` padding mask

處置：程式修改完成，cluster runtime 驗證待執行。MDGen 2024 在 `--no_rope`＋manual attention 時會進入 PyTorch `F.multi_head_attention_forward`；原本傳入的 float `1 - mask` 會被解讀為加到 attention score 的 additive mask，而不是 padding 遮蔽。MDGen-2026 r2 在 attention caller 統一建立 boolean `key_padding_mask`，使 manual 與 SDPA 在 `--no_rope` 下具有相同的遮蔽語意。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/model/latent_model.py` | `no-rope-padding-mask`（r2 新增） | 將有效位置 mask 轉為 `~mask.to(torch.bool)`，再傳入 `key_padding_mask`。有效 residue 為 `False`，padding 為 `True`；RoPE／no-RoPE、manual／SDPA 共用相同語意。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；r2 修改並保留） | 新增經過 `AttentionWithRoPE` caller 的 no-RoPE＋padding case，比較 manual／SDPA output、input gradient 與 parameter gradient。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；r2 修改並保留） | 將 padding 位置輸入加上 `100.0`，確認兩條路徑的有效 residue output 均不改變。 |

這是與 MDGen 2024 的刻意行為差異：MDGen 2024 在 `--no_rope` 且有 padding 時沒有真正遮蔽 padding key；MDGen-2026 r2 改為正確遮蔽。影響範圍僅限 `--no_rope` 且 batch 含 padding 的訓練／推論；預設啟用 RoPE 的 ATLAS 設定與 `atlas.ckpt` 不受影響。使用 MDGen 2024 `--no_rope` 訓練的 checkpoint，在含 padding 的輸入上可能與 r2 行為不同。

驗證計畫：在 mdgen2026 conda／cluster 環境執行 `test_sdpa.py`，要求 no-RoPE case 的 manual path 為 `torch_mha`、SDPA path 為 `sdpa`，output／gradient 相對誤差低於既有門檻，且兩條路徑的 padding invariance error 均低於 `1e-6`。

### M5 — deterministic／seed 預設關閉

處置：程式修改完成，cluster runtime 驗證待執行。一般訓練、推論與 L-scaling 預設不主動設定 random seed，亦不啟用 deterministic algorithms；需要可重現的 benchmark／測試時，才明確傳入 seed 與 `--deterministic`。`--benchmark` 介面及其與 deterministic 不可同時開啟的檢查均保留。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/parsing.py` | `seed-deterministic-args`（r1 已存在；r2 修改並保留） | `--train_seed` 預設由 `137` 改為 `None`，`--deterministic` 預設由開啟改為關閉；仍可由 CLI 明確指定。 |
| `train.py` | `seed-initialization`（r1 已存在；r2 修改並保留） | 僅在明確提供 `--train_seed` 時呼叫 `pl.seed_everything`；Trainer 繼續依 `--deterministic` 與 `--benchmark` 設定執行。 |
| `train-runtime.py` | `seed-initialization`（r1 已存在；r2 修改並保留） | 與正式訓練入口採用相同預設；需要固定 benchmark 輸入時可傳 `--train_seed 137`，正式速度測量則使用預設非 deterministic 或明確傳入 `--no-deterministic`。 |
| `sim_inference.py` | `seed-deterministic-args`、`seed-initialization`（r1 已存在；r2 修改並保留） | `--inference_seed` 預設改為 `None`、deterministic 預設關閉，只有明確提供 seed 時才初始化 RNG。 |
| `benchmark_l_scaling.py` | `seed-deterministic-args`、`seed-initialization`、`deterministic-execution`（r2 新增） | `--seed` 預設改為 `None`，新增預設關閉的 `--deterministic`；輸出 JSON 記錄實際設定。需要重現 A4 時明確傳入 `--seed 137 --deterministic`。 |

`CUBLAS_WORKSPACE_CONFIG`回覆：先前未主動設定此環境變數，repo 與既有執行指令亦未設定；先前執行未遇到要求該設定的 cuBLAS deterministic 錯誤。r2 的一般執行預設不啟用 deterministic algorithms，因此不要求設定；未來若明確執行 deterministic CUDA 測試，將於啟動 Python 前設定並記錄實際值。

### M6 — NPY dtype 選項

處置：程式修改完成，cluster runtime 驗證待執行。前處理輸出的 NPY 預設恢復為上游使用的 float16，使 ATLAS 資料量化方式與 `atlas.ckpt` 的訓練條件一致；需要較大座標範圍的 Piezo1、膜蛋白或 ligand 系統可明確選擇 float32。這只調整 NPY 的儲存 dtype；training dataset 現有程式會在讀取後轉為 float32，模型運算 dtype 不變。

| file_path | block_name | 說明改動 |
|---|---|---|
| `scripts/prep_sims.py` | `fp32-data-pipeline`（r1 已存在；r2 刪除） | 移除固定使用 `np.float32` 建立 NPY 的設定。 |
| `scripts/prep_sims.py` | `npy-dtype-option`（r2 新增） | 新增 `--dtype {float16,float32}`，預設 `float16`；依選項決定輸出 NPY dtype。 |
| `sim_inference.py` | `fp32-data-pipeline`（r1 已存在；r2 刪除） | 移除無條件將完整輸入複製成 float32 的處理，恢復上游行為：一般 inference 複製第一個 frame 並轉為 float32，TPS 路徑保留原始 memmap 與 dtype。 |

驗證計畫：分別以預設設定與 `--dtype float32` 產生小型 NPY，確認輸出 dtype 為 float16／float32；再各執行一次一般 inference 與 TPS smoke test，確認一般 inference 使用 float32 複本，而 TPS 不複製完整 trajectory。M6 依老師要求獨立 commit。

### M8 — SDPA 改為預設 attention

處置：程式修改完成，五個正式入口的 cluster runtime 驗證待執行。使用者未指定 attention flag 時預設使用 SDPA；需要上游手寫 attention 做 A/B 比較時，才明確傳入 `--manual_attention`。`atlas.ckpt` 的 state_dict 不需轉換，manual 與 SDPA 共用相同參數，差異僅為 runtime attention 路徑。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/parsing.py` | `sdpa-route`（r1 已存在；r2 修改並保留） | 將 `--use_sdpa` 改為 `--manual_attention`；預設不帶 flag 時使用 SDPA。 |
| `train.py`、`train-runtime.py` | `sdpa-route`（r1 已存在；r2 修改並保留） | 由 `manual_attention` 反向計算 `use_sdpa`；預設 SDPA，只有 `--manual_attention` 使用手寫 attention。 |
| `mdgen/wrapper.py` | `sdpa-route`（r1 已存在；r2 修改並保留） | `NewMDGenWrapper` 的 `use_sdpa` 建構預設由 `False` 改為 `True`。 |
| `mdgen/model/latent_model.py` | `sdpa-route`（r1 已存在；r2 修改並保留） | `LatentMDGenModel` 的 `use_sdpa` 建構預設由 `False` 改為 `True`；內部 layer 仍接收上層明確傳入的值。 |
| `sim_inference.py` | `sdpa-route`（r1 已存在；r2 修改並保留） | 將 `--use_sdpa` 改為 `--manual_attention`；checkpoint 預設以 SDPA 路徑載入。 |
| `tps_inference.py`、`design_inference.py`、`upsampling_inference.py` | `sdpa-route`（r2 新增） | 三個入口新增 `--manual_attention`，並把對應的 `use_sdpa` 值傳入 `NewMDGenWrapper.load_from_checkpoint()`。 |
| `tps_inference.py`、`design_inference.py`、`upsampling_inference.py` | `runtime-compatibility`（r2 新增） | checkpoint 載入補上 `weights_only=False`，支援新版 PyTorch 載入舊 Lightning checkpoint。 |
| `benchmark_l_scaling.py` | `sdpa-route`（r2 新增） | 將 `--use_sdpa` 改為 `--manual_attention`；預設量測 SDPA，加 flag 才量測 manual。 |

A/B 指令規則同步更新：SDPA 組不帶 attention flag；manual 組明確加入 `--manual_attention`。因此 C2／C6 等 SDPA 指令移除 `--use_sdpa`，C5 等 manual 指令新增 `--manual_attention`；A4 manual 組加 flag、SDPA 組不帶 flag；A6 manual inference 加 flag、SDPA inference 不帶 flag。`test_sdpa.py` 仍在程式內明確建立 manual 與 SDPA 兩組，不依賴 CLI 預設值，因此不需修改。

驗證計畫：分別執行 `train.py`、`sim_inference.py`、`tps_inference.py`、`design_inference.py`、`upsampling_inference.py`，確認均可載入既有 checkpoint，且未帶 `--manual_attention` 時實際 backend 為 SDPA；再以 `--manual_attention` smoke test 確認手寫路徑仍可用。

### M9 — `--sim_dir` alias

處置：程式修改完成。`scripts/prep_sims.py` 以 `--atlas_dir` 作為目前的主要參數名稱，同時恢復舊版 README 使用的 `--sim_dir` alias；兩種寫法均解析至同一個 `args.atlas_dir`，後續資料讀取邏輯不需分支。

| file_path | block_name | 說明改動 |
|---|---|---|
| `scripts/prep_sims.py` | `data-preprocess-fixes`（r1 已存在；r2 修改並保留） | `--atlas_dir` 與舊 `--sim_dir` 共用同一個 argparse 參數，並統一儲存於 `args.atlas_dir`。 |

驗證方式：執行 `python scripts/prep_sims.py --help`，確認同一選項列出 `--atlas_dir ATLAS_DIR, --sim_dir ATLAS_DIR`；實際前處理可使用任一名稱，未提供時仍沿用原本的預設路徑。

### M10 — 第三方檔案來源與授權

處置：保留兩個 AlphaFlow ensemble 分析工具，並補齊可重現的來源與 MIT 授權聲明。修改前已將本機檔案與 AlphaFlow commit `0408d7c89dac444a43a9089d7427ce470b0a5e67` 的對應原檔逐位元比較；兩者內容完全相同，本 repo 只有檔名不同。M10 僅新增檔頭註解，不修改分析邏輯。

| file_path | block_name | 說明改動 |
|---|---|---|
| `scripts/analyze_ensemble.py` | `third-party-provenance`（r2 新增） | 記錄原始檔 `scripts/analyze_ensembles.py` 的固定 commit URL、僅改名說明、原作者 copyright 與 MIT License。 |
| `scripts/print.py` | `third-party-provenance`（r2 新增） | 記錄原始檔 `scripts/print_analysis.py` 的固定 commit URL、僅改名說明、原作者 copyright 與 MIT License。 |

來源 repository：`https://github.com/bjing2016/alphaflow`。原始授權為 MIT License，copyright 為 `2024 Bowen Jing, Bonnie Berger, Tommi Jaakkola`。驗證方式為移除新增的 `third-party-provenance` 檔頭並忽略檔尾換行後，分別與固定 commit 的原始檔比較，並對兩個本機檔案執行 Python 語法編譯檢查。
