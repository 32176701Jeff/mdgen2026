# MDGen-2026 r1 Review Reply

- Reviewed tag: `mdgen2026-r1`
- Reviewed commit: `c0ee7cb6726c4dc1c40a15203127ef64cbf13528`
- Target tag: `mdgen2026-r2`

## 4. 詳細回覆與驗證證據

### M1 — position_ids

處置：程式修改完成，cluster runtime 驗證待執行。本輪不實作 explicit `position_ids` 的 RoPE 語意；正常訓練與推論不再把 CSV 的 `position_ids` 傳入模型。模型介面維持 `position_ids=None`，若 RoPE 收到非 `None` 值則明確拋出 `NotImplementedError`，避免輸入被靜默忽略。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/dataset.py` | `position-id-dataset`（r1，刪除） | 不再解析、驗證或回傳 CSV 的 `position_ids`。CSV 可保留該欄位，但本輪 dataset 會忽略。 |
| `mdgen/dataset.py` | `position-id-crop-padding`（r1，刪除） | 移除 position ID 的 crop 與 padding；residue、mask 與其他既有資料處理維持不變。 |
| `mdgen/wrapper.py` | `position-id-wrapper`（r1，刪除） | 移除三處由 batch 傳入 `model_kwargs` 的 `position_ids`；模型使用預設值 `None`。 |
| `sim_inference.py` | `position-id-inference-input`（r1，刪除） | 不再從 inference CSV 解析 `position_ids`。 |
| `sim_inference.py` | `position-id-inference-routing`（r1，刪除） | 移除函式參數、batch 欄位與 inference model routing；保留 NPY residue 數量與 `seqres` 長度驗證。 |
| `mdgen/model/mha.py` | `position-id-rope-interface`（r1，修改並保留） | 保留 `position_ids=None` 介面與上游 RoPE 計算；收到非 `None` 時拋出 `NotImplementedError`。 |
| `scripts/prep-protein-csv.py` | `position-id-csv`（r1，保留） | 保留 CSV 產生功能，於檔頭新增 r2 說明：position IDs 僅供未來使用，MDGen-2026 r2 模型不會讀取。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1，修改並保留） | 新增 manual 與 SDPA 收到非 `None` position IDs 時都必須拋出 `NotImplementedError` 的 regression case；`None` 路徑維持上游 RoPE 行為。 |
