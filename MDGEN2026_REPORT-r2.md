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
