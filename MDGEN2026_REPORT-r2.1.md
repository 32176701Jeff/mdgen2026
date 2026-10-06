# MDGen-2026 r2.1 Review Reply

本文件記錄 `mdgen2026-r2` 至 `mdgen2026-r2.1` 的修改、驗證與回覆。

## R2 — C19 conda 環境建立 log

目的：使用兩份包含 PyEMMA 的環境鎖定檔，分別建立全新的 CUDA 12.6 與 CUDA 13.0 conda 環境，並保存完整的環境建立輸出，作為環境可重現性的證據。

```bash
conda env create \
  --name mdgen2026-r2-a6-cu126 \
  --file my_new/environment-mdgen2026-cu126-pyemma.yml \
  2>&1 | tee my_new/r2/C19_M12-cu126-conda-create.log

conda env create \
  --name mdgen2026-r2-a6-cu130 \
  --file my_new/environment-mdgen2026-cu130-pyemma.yml \
  2>&1 | tee my_new/r2/C19_M12-cu130-conda-create.log
```
預期產物：
(a) `my_new/r2/C19_M12-cu126-conda-create.log`
(b) `my_new/r2/C19_M12-cu130-conda-create.log`

## R3 — 大型產物移出 Git 追蹤
處置：將 `*.npy`、`*.npz`、`*.pdb`、`*.pickle` 加入 `.gitignore`，並以 `git rm --cached` 取消既有大型產物的 Git 追蹤。此操作只修改 Git index，不刪除工作目錄中的實體檔案；JSON、log、txt、code 與 CSV 均繼續保留於 repository。
取消追蹤清單摘要：
(a) `.npy`：52 個
(b) `.npz`：26 個
(c) `.pdb`：2 個
(d) `.pickle`：1 個
原始檔保存位置：81 個實體檔目前均保留於執行工作的原始工作目錄；正式外部保存位置待補。
執行狀態：Git index 整理與實體檔保留檢查已完成；Git 已不再追蹤上述四種大型產物，工作目錄仍保有 52 個 NPY、26 個 NPZ、2 個 PDB 與 1 個 pickle。待補正式外部保存位置及 R3 獨立 commit。
### C40 — R3 repository 大型產物整理

```bash
git ls-files -z -- '*.npy' '*.npz' '*.pdb' '*.pickle' \
  | xargs -0 git rm --cached --

git add .gitignore
git ls-files -- '*.npy' '*.npz' '*.pdb' '*.pickle'
git status --short
git commit -m "R3: stop tracking generated artifacts"
```

## R4 — position_ids 檢查移到 MHA 入口

處置：將非 `None` 的 `position_ids` 檢查由 RoPE 實作移至 `MultiheadAttention.forward` 入口。RoPE／no-RoPE、manual／SDPA 收到非 `None` 的 `position_ids` 時，現在都會拋出 `NotImplementedError`，不再於 no-RoPE 路徑靜默忽略輸入。

| file_path | block_name | 說明改動 |
|---|---|---|
| `mdgen/model/mha.py` | `position-id-rope-interface`（r1 已存在；r2.1 修改並保留） | 將非 `None` 的 `position_ids` 拒絕檢查移到 `MultiheadAttention.forward` 入口，使檢查不再依賴 RoPE 是否啟用。 |
| `test_sdpa.py` | `sdpa-equivalence-test`（r1 已存在；r2.1 修改並保留） | 在既有 RoPE manual／SDPA rejection case 之外，新增 no-RoPE manual／SDPA case；四條路徑收到非 `None` 的 `position_ids` 時都必須拋出 `NotImplementedError`。 |

### C41 — R4 position_ids CPU regression

```bash
conda activate mdgen2026-r2-a6-cu126
mkdir -p my_new/r2.1

python test_sdpa.py \
  --device cpu \
  --seed 137 \
  --output_dir my_new/r2.1/C41_R4-position-ids \
  2>&1 | tee my_new/r2.1/C41_R4-position-ids.log
```
