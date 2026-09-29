## 2026-09-25 mdgen2026
> 建立日期：2026-09-24 ｜ 最後更新：2026-09-24  
> 作者：王皇文  
> github帳號：`32176701Jeff`  mail:`r12521230@ntu.edu.tw`

## 主要改進
1. torch版本更新
1. SDPA,FP32
1. positionID傳遞到ROPE
1. seed/deterministic設定

## 交付內容
1. T1-電腦與環境
1. T2-原始code議題確認
1. T3-code_modification
1. T4-等價性測試
1. T5-Performance/Batchsize/residue_length/GPU_memory

## T1-電腦與環境

### T1.1-電腦規格
| 項目 | 規格 |
|---|---|
| 主機名稱 | m5-cluster2 |
| 作業系統 | Ubuntu 24.04.5 LTS |
| 系統架構 | x86_64 |
| Linux kernel | 7.0.0-31-generic |
| CPU | AMD Ryzen 9 9950X |
| CPU核心／執行緒 | 16 cores／32 threads |
| 系統記憶體 | 30 GiB |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU記憶體 | 24,564 MiB |
| NVIDIA driver | 595.71.05 |
| Driver最高支援CUDA版本 | 13.2 |

### T1.2-Conda與PyTorch環境
cu126與cu130環境使用相同的Python及MDGen相依套件，主要差異只有PyTorch使用的CUDA runtime版本。
共同套件版本：
| 項目 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch Lightning | 2.6.5 |
| NumPy | 1.26.4 |
| fair-esm | 2.0.0 |
| torchdiffeq | 0.2.5 |

環境差異：
| 項目 | cu126環境 | cu130環境 |
|---|---|---|
| 環境設定檔 | [environment-mdgen2026-cu126.yml](./my_new/environment-mdgen2026-cu126.yml) | [environment-mdgen2026-cu130.yml](./my_new/environment-mdgen2026-cu130.yml) |
| Conda環境名稱 | `mdgen2026-cu126` | `mdgen2026-cu130` |
| PyTorch | 2.12.1+cu126 | 2.12.1+cu130 |
| PyTorch CUDA runtime | 12.6 | 13.0 |
| Torchvision | 0.27.1+cu126 | 0.27.1+cu130 |

### T1.3-相依性檢查
| 檢查項目 | 結果 |
|---|---|
| PyTorch Lightning 是否阻擋 PyTorch 2.12.1 | 否；實際安裝 PyTorch Lightning 2.6.5 |
| torchaudio | 未安裝，MDGen 未使用 |
| pip openfold | 未安裝 |
| flash-attn | 非必要相依；本次使用 PyTorch 內建 SDPA |


## T2-原始code議題確認

T2檢查原始MDGen在導入SDPA前需要確認的實作細節；對應的MDGen-2026修改方式記錄於T3。

原始MDGen：[https://github.com/bjing2016/mdgen](https://github.com/bjing2016/mdgen)

### T2.1-SDPA前置確認

| 編號 | 確認項目 | 原始code確認結果 | 對應T3項目 |
|---|---|---|---|
| Q1 | Scaling | Q已在外部乘上`head_dim**-0.5` | Scaling |
| Q2 | Padding mask | Dataset、MHA與SDPA使用的mask極性不同 | Mask極性與fast path |
| Q3 | add_bias_kv | K/V會增加一個learned token，mask也會同步延長 | bias K/V |
| Q4 | Time-axis mask | Time-axis attention有傳入mask | Time-axis mask |
| Q5 | Dropout | Manual attention只在training啟用dropout | Dropout |
| Q6 | 輸出比較位置 | MHA在`out_proj`後的共同輸出為`attn` | Tensor layout與輸出 |

#### Q1-Scaling位置

確認結果：

SDPA內部會自動執行scaling。原始MDGen已在外部對Q乘上`head_dim**-0.5`，因此導入SDPA時需要避免重複scaling。

原始code：

```
# mdgen/model/mha.py, line 102
self.scaling = self.head_dim**-0.5

# mdgen/model/mha.py, line 263
q *= self.scaling

# mdgen/model/mha.py, line 359
attn_weights = torch.bmm(q, k.transpose(1, 2))
```

#### Q2-Padding

確認結果：

Dataset的mask中，1代表有效residue，0代表padding。原始MDGen使用`1 - mask`產生`key_padding_mask`，因此有效位置變成0，padding位置變成1；在manual attention中，`key_padding_mask=True`代表需要遮蔽。PyTorch SDPA的boolean mask語意剛好相反，True代表允許參與attention，False代表遮蔽，因此傳入SDPA前必須對`key_padding_mask`取反。
| 位置 | Dataset mask | key_padding_mask | SDPA boolean mask |
|---|---:|---:|---:|
| 有效residue | 1 | False／0 | True／1 |
| Padding | 0 | True／1 | False／0 |

Padding mask的作用是：所有位置先計算Q、K、V；完成QKᵀ後，padding key對應的attention score會在softmax前設為`-inf`，softmax後權重變成0，因此對應的V不會貢獻到最終output。

原始code：

[1] Dataset建立mask
```
# mdgen/dataset.py, line55-56
L = frames.shape[1]
mask = np.ones(L, dtype=np.float32)

# mdgen/dataset.py, line80–89, L<cropping window
mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])
```

[2] 轉換成key_padding_mask
```
# mdgen/model/latent_model.py，line 458–461
mask.reshape(B * T, L)

# mdgen/model/latent_model.py, line 325–327
x, _ = self.attn(
    query=x,
    key=x,
    value=x,
    key_padding_mask=1 - mask
)
```

[3] Manual attention套用mask
```
# mdgen/model/mha.py，line 251–263
q = self.q_proj(query)
k = self.k_proj(key)
v = self.v_proj(value)
q *= self.scaling

# mdgen/model/mha.py，line 359
attn_weights = torch.bmm(q, k.transpose(1, 2))

# mdgen/model/mha.py，line 370–375
if key_padding_mask is not None:
    attn_weights = attn_weights.view(
        bsz, self.num_heads, tgt_len, src_len
    )
    attn_weights = attn_weights.masked_fill(
        key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
        float("-inf")
    )

# mdgen/model/mha.py，line 381
attn_weights_float = utils_softmax(
    attn_weights, dim=-1, onnx_trace=self.onnx_trace
)

# mdgen/model/mha.py，line 383–389
attn_probs = F.dropout(
    attn_weights_float.type_as(attn_weights),
    p=self.dropout,
    training=self.training,
)
attn = torch.bmm(attn_probs, v)
```

#### Q3-add_bias_kv

確認結果：

`bias_k`和`bias_v`是兩個可以訓練的額外向量。假設原本序列有L個token，啟用`add_bias_kv`後，程式會在K和V的尾端各加上一個learned token。Q長度仍為L，K和V長度變成L+1，padding mask也必須同步延長為L+1。

Bias token不是padding，所以原始`key_padding_mask`在尾端補0，代表不遮蔽。SDPA本身沒有`add_bias_kv=True`參數，只接收已經準備好的Q、K、V。

原始code：

```
# mdgen/model/latent_model.py, line 343, IPA
self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

# mdgen/model/latent_model.py, line 401, LatentMDGenLayer
self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)
```

#### Q4-Time-axis attention有沒有使用mask

確認結果：

Time-axis attention有使用mask。

原始code：

```
# mdgen/dataset.py，line 80–89
mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

# mdgen/wrapper.py，line 353–364, 原始 batch['mask'] 的形狀是 B × L。這裡沿著時間軸展開成 B × T × L
'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1)

# mdgen/model/latent_model.py，line 472–475
x = self.mha_t(
    x.transpose(1, 2).reshape(B * L, T, C),
    mask=mask.transpose(1, 2).reshape(B * L, T)
)
```

#### Q5-Dropout

確認結果：

原始manual attention透過`training=self.training`控制dropout；SDPA需要由呼叫端明確設定dropout機率。

原始code：

```
# mdgen/model/mha.py，line 381–389
attn_weights_float = utils_softmax(
    attn_weights,
    dim=-1,
    onnx_trace=self.onnx_trace
)
attn_probs = F.dropout(
    attn_weights_float.type_as(attn_weights),
    p=self.dropout,
    training=self.training,
)
attn = torch.bmm(attn_probs, v)
```

#### Q6-輸出比較位置

確認結果：

Manual attention與SDPA皆以`out_proj`後的`attn`作為數值比較位置。比較時另外區分diffusion time、layer及residue／frame／IPA attention。

原始code：

```
mdgen/model/mha.py，line 397，variable_name = attn
```

### T2.2-Data representation與cropping順序

確認結果：

原始MDGen先計算frame與torsion，再執行crop。

原始code：

```
# mdgen/dataset.py, line79-84
torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
torsion_mask = torsion_mask[0]
if self.args.atlas:
    if L > self.args.crop:
```

## T3-code_modification

### T3.0-New files

MDGen-2026新增以下四個檔案。此處先說明新增原因與檔案來源，實作及使用方式仍依功能放在後續對應章節。

| 新增檔案 | 新增原因 | 詳細說明 |
|---|---|---|
| `scripts/prep-protein-csv.py` | 原始MDGen沒有從PDB產生`seqres`與`position_ids` CSV的工具 | T3.4、T3.6 |
| `mdgen/attn_capture.py` | 集中處理attention output與metadata輸出，供manual attention與SDPA進行數值比較 | T3.2、T4 |
| `scripts/analyze_ensemble.py` | 從AlphaFlow下載的evaluation檔案，用於計算ensemble的RMSD、RMSF、PCA、Wasserstein distance、contact與SASA等評估結果 | T4 |
| `scripts/print.py` | 從AlphaFlow下載的evaluation檔案，用於彙整evaluation輸出的pickle結果並列印各項統計指標 | T4 |

#### block_name標記規則

另外，程式碼中的修改區塊會以`# <block_name>`標記，可在對應檔案中搜尋`# <block_name>`定位實作。<br>

`# <block_name>`的標記方式如下：

1. 完整新增 function/class：定義前使用 # block_name
1. 既有 function 的連續多行修改：使用 # block_name:start／# block_name:end
1. 單行修改：行尾使用 # block_name
1. 單一新增 argument：argument 行尾使用 # block_name
1. 不連續修改區域：分別標記，不共用一組 start/end
1. start/end 不跨越 function 或 class
1. 不將未修改的原始 code 包入標記範圍
1. 只使用報告已定義的 block name

### T3.1-新增args

#### Runtime行為

1. Training與inference分別提供SDPA開關；未使用`--use_sdpa`時走原本的manual attention，使用後才切換到SDPA。
1. PyTorch SDPA會依照GPU、dtype、mask與tensor shape，在執行時選擇FLASH_ATTENTION、EFFICIENT_ATTENTION、CUDNN_ATTENTION或MATH backend。

#### 輸出與比較

1. 使用`--print_sdpa_backend <path>`可以輸出實際使用的backend；由於profiler會影響效能，因此backend確認與正式performance測量分開執行。
1. 使用`--attn_to_npy <folder_path>`可以輸出第一次model evaluation中最高層的residue、frame與IPA MHA結果，供manual attention與SDPA進行數值誤差比較。

#### Checkpoint相容性

1. 以上參數都是MDGen-2026新增的runtime選項，不會寫入checkpoint；沒有提供參數時不影響原本的training或inference流程。  

#### Args總覽

| 類別 | Args | 功能 |
|---|---|---|
| SDPA | `--use_sdpa` | 啟用PyTorch SDPA；未提供時使用原本的manual attention。Training與inference皆支援。 |
| SDPA | `--print_sdpa_backend <path>` | 將實際使用的SDPA backend與attention路徑輸出至指定檔案。 |
| SDPA | `--attn_to_npy <folder_path>` | 將最高層的residue、frame與IPA MHA輸出存成NPY，供manual與SDPA數值比較。 |
| FP32 | `--precision 32-true` | Training固定使用完整FP32；此參數原本已存在，MDGen-2026將可選值限制為`32-true`。 |
| Deterministic | `--deterministic`／`--no-deterministic` | 控制是否要求使用deterministic演算法；預設開啟。 |
| Deterministic | `--benchmark`／`--no-benchmark` | 控制cuDNN benchmark autotuner；預設關閉，且不能與deterministic同時開啟。 |
| Seed | `--train_seed <int>` | 設定training使用的seed，預設為137。 |
| Seed | `--inference_seed <int>` | 設定inference使用的seed，預設為137。 |
| 其他 | `--model_dir <path>` | 指定training輸出目錄；未提供時使用原本的`workdir/<run_name>`。 |



### T3.2－SDPA實作與T2問題處理

MDGen-2026將原本的`bmm → mask → softmax → dropout → bmm`改為`F.scaled_dot_product_attention`，其餘Q/K/V projection、RoPE、bias K/V與output projection維持原本語意。  

| 項目 | T2確認結果 | MDGen-2026處理方式 | 修改檔案,block_name |
|---|---|---|---|
| Scaling(Q1) | 原始code已執行`q *= self.scaling` | 保留原始scaling，呼叫SDPA時設定`scale=1.0`，避免重複scaling | `mdgen/model/mha.py`, `sdpa-scaling` |
| Mask極性與fast path(Q2) | `key_padding_mask=True`代表遮蔽，但SDPA中`True`代表保留 | 使用`~key_padding_mask`反轉極性；全部為True時改傳`None` | `mdgen/model/mha.py`, `sdpa-mask` |
| bias K/V(Q3) | `add_bias_kv=True`，bias K/V實際有使用 | 保留原本K/V append與mask延長流程，再將處理完成的K/V交給SDPA | `mdgen/model/mha.py`, `sdpa-bias-kv` |
| Dropout(Q5) | manual attention只在training啟用dropout | SDPA使用`self.dropout if self.training else 0.0` | `mdgen/model/mha.py`, `sdpa-dropout` |
| Tensor layout與輸出(Q6) | 原始MHA合併batch與head維度 | SDPA前轉為`B,H,L,D`；完成後還原排列並通過原本的`out_proj` | `mdgen/model/mha.py`, `sdpa-layout-output` |
| Time-axis mask(Q4) | `mha_t`呼叫端有傳入mask | frame-axis MHA沿用相同的SDPA與mask轉換流程 | `mdgen/model/latent_model.py`, `sdpa-time-axis-mask` |
| SDPA route | 原始model沒有SDPA runtime開關、eligibility檢查與fallback流程 | 將`use_sdpa`傳入各attention layer；每次forward依執行條件選擇SDPA或manual attention，並記錄實際路徑與fallback原因 | `mdgen/wrapper.py`+`mdgen/model/latent_model.py`+`mdgen/model/mha.py`, `sdpa-route` |
| SDPA diagnostics(Q6) | 原始code沒有backend確認與中間attention輸出功能 | 使用profiler輸出實際SDPA backend與module-level路徑，並保存最高層residue、frame及IPA MHA output與metadata供數值比較 | `train.py`+`sim_inference.py`+`mdgen/model/mha.py`+`mdgen/attn_capture.py`, `sdpa-diagnostics` |

### T3.3-FP32

MDGen-2026全程使用FP32，不啟用bf16、AMP或TF32，以便進行嚴格的數值等價測試。<br>

| 項目 | 原始code狀態 | MDGen-2026處理方式 | 修改檔案與block_name |
|---|---|---|---|
| Training precision | `--precision`可使用其他精度設定 | 限制為`32-true`，避免啟用mixed precision | `mdgen/parsing.py`, `fp32-training-precision` |
| Matmul計算精度 | Training設定為`medium`，可能使用較低的內部計算精度 | Training與inference皆設定為`highest` | `train.py`+`sim_inference.py`, `fp32-matmul-precision` |
| TF32 | 原始code未明確關閉TF32 | 明確關閉CUDA matmul與cuDNN的TF32 | `train.py`+`sim_inference.py`, `fp32-disable-tf32` |
| Inference模型dtype | 載入模型後未明確指定dtype | 使用`model.eval().float()`，確保模型以FP32執行 | `sim_inference.py`, `fp32-inference-model` |
| Data pipeline dtype | `prep_sims.py`將座標以float16存入NPY，且TPS inference路徑未統一轉為float32 | 預處理階段直接將座標以float32存入NPY，避免先經float16量化；inference所有路徑載入後皆明確複製為float32 | `scripts/prep_sims.py`+`sim_inference.py`, `fp32-data-pipeline` |

### T3.4-positionID

MDGen-2026將PDB residue number(目前是以ATLAS提供的pdb為版本，如果有其他pdb格式，需要更改的事`scripts/prep-protein-csv.py`)轉換成`position_ids`，並從CSV一路傳遞至RoPE介面。現階段RoPE仍沿用原本的連續位置計算，因此不改變既有模型數值<br>
| 項目 | 原始code狀態 | MDGen-2026處理方式 | 修改檔案與block_name |
|---|---|---|---|
| CSV產生 | 原始CSV只有protein name與`seqres` | 從PDB residue number產生以0為起點且保留gap的`position_ids`，以JSON格式寫入CSV | `scripts/prep-protein-csv.py`, `position-id-csv` |
| Training資料讀取 | Dataset沒有`position_ids` | CSV有`position_ids`時讀取；沒有時使用`arange(L)`，並檢查長度 | `mdgen/dataset.py`, `position-id-dataset` |
| Crop與padding | 未處理position資訊 | Crop時同步裁切；padding時同步補齊`position_ids` | `mdgen/dataset.py`, `position-id-crop-padding` |
| Inference資料讀取 | Inference只讀取`seqres` | CSV有`position_ids`時讀取；沒有時使用`arange(L)` | `sim_inference.py`, `position-id-inference-input` |
| Inference內部傳遞 | Inference batch與rollout沒有position資訊 | 檢查`seqres`、`position_ids`及NPY residue數量一致，並經由`get_batch`、rollout與expanded batch將`position_ids`傳入model | `sim_inference.py`, `position-id-inference-routing` |
| Model kwargs傳遞 | Wrapper未傳遞position資訊 | Training、validation與inference皆將`position_ids`加入`model_kwargs` | `mdgen/wrapper.py`, `position-id-wrapper` |
| Model內部傳遞 | Latent model與各attention layer沒有`position_ids`參數 | 將`position_ids`傳入prepend IPA與各層residue-axis MHA | `mdgen/model/latent_model.py`, `position-id-model-routing` |
| Residue-axis排列 | Residue MHA使用`B×T`作為batch維度 | 將`(B,L)`展開為`(B×T,L)`；time-axis MHA不使用residue位置 | `mdgen/model/latent_model.py`, `position-id-residue-layout` |
| MHA與RoPE介面 | MHA直接呼叫`self.rot_emb(q, k)` | MHA與RoPE新增`position_ids=None`介面；目前仍使用原本RoPE計算，保持數值不變 | `mdgen/model/mha.py`, `position-id-rope-interface` |

## T3.5-seed_and_deterministic

MDGen-2026為training與inference加入可設定的seed與deterministic控制。預設使用seed 137、開啟deterministic並關閉cuDNN benchmark，讓manual attention與SDPA能在相同條件下比較。<br>

| 項目 | 原始code狀態 | MDGen-2026處理方式 | 修改檔案與block_name |
|---|---|---|---|
| Runtime參數 | Training只有註解掉的seed設定；inference沒有seed與deterministic參數 | 新增`--train_seed`、`--inference_seed`、`--deterministic`與`--benchmark`，預設seed為137、deterministic開啟、benchmark關閉 | `mdgen/parsing.py`+`sim_inference.py`, `seed-deterministic-args` |
| Seed初始化 | 原始code沒有實際執行統一seed設定 | 使用`seed_everything(..., workers=True)`固定Python、NumPy、PyTorch與DataLoader worker的seed | `train.py`+`sim_inference.py`, `seed-initialization` |
| Deterministic執行 | 原始code未明確要求deterministic演算法 | Training將設定交給Lightning Trainer；inference使用`torch.use_deterministic_algorithms` | `train.py`+`sim_inference.py`, `deterministic-execution` |
| cuDNN benchmark | 原始code未明確控制benchmark | 將benchmark設定傳入training與inference；禁止同時開啟deterministic與benchmark | `mdgen/parsing.py`+`train.py`+`sim_inference.py`, `deterministic-benchmark-guard` |

## T3.6-scripts/prep-protein-csv.py
原始MDGen需要從CSV讀取蛋白質名稱與胺基酸序列，其中seqres使用單字母胺基酸代號；這些代號及其對應關係定義於mdgen/residue_constants.py。但原始MDGen沒有提供從PDB檔案產生所需CSV的程式，因此MDGen2026新增scripts/prep-protein-csv.py，將PDB中的三字母胺基酸名稱轉換為單字母序列，並輸出可供Dataset讀取的CSV。
此腳本同時從PDB residue number產生position_ids。第一個residue的位置會正規化為0，後續位置保留原始PDB residue number的間距。例如原始編號為10, 11, 14時，輸出的position_ids為[0, 1, 4]，因此能保留缺失residue所形成的gap。
```
name,seqres,position_ids
1a62_A,MNLTELKNTPVSELITLGENMGLENLARMRKQDIIFAILKQHAKSGEDIFGDGVLEILQDGFGFLRSADSSYLAGPDDIYVSPSQIRRFNLRTGDTISGKIRPPKEGERYFALLKVNEVNFDKPENARNK,"[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129]"
```

## T3.7-other

以下項目不屬於SDPA、FP32、position ID、seed/deterministic或`prep-protein-csv.py`的主要功能修改，主要是原始code修正與新版環境相容性處理。

| 項目 | 原始code狀態 | MDGen-2026處理方式 | 修改檔案與block_name |
|---|---|---|---|
| Data preprocess修正 | Frame window抽樣未包含最後合法位置，且`prep_sims.py`的CLI參數名稱與實際讀取欄位不一致 | 修正frame數檢查與抽樣邊界，並將ATLAS目錄參數統一為`--atlas_dir` | `mdgen/dataset.py`+`scripts/prep_sims.py`, `data-preprocess-fixes` |
| Runtime相容性 | 新版FlashAttention可能缺少legacy API，且`batched_gather`使用list形式indexing | 對FlashAttention legacy API加入optional import guard，並將index list轉為tuple以相容新版執行環境 | `mdgen/model/primitives.py`+`mdgen/tensor_utils.py`, `runtime-compatibility` |

# T4-等價性測試

## T4.1-data_preprocess與執行流程

### T4.1.1-測試蛋白質與atlas.ckpt
本次測試使用`1a62_A`與`1bkp_A`兩個蛋白質，並將training的`crop`設定為256。兩個蛋白質分別短於及長於cropping window，可同時測試padding與cropping兩種資料處理路徑。
| 蛋白質 | Residue數量 | 與crop=256的關係 | Dataset處理方式 | 測試目的 |
|---|---:|---|---|---|
| `1a62_A` | 130 | 小於256 | 保留完整序列並padding至256 | 驗證padding mask及SDPA mask轉換 |
| `1bkp_A` | 278 | 大於256 | 從原始序列crop出256個residues | 驗證cropping |

下載方式，下載後的檔案會存在./data/pdbxtc中
```
bash ./my_new/download_atlas.sh
```

atlas.ckpt在https://huggingface.co/bjing-mit/mdgen下載 並存入./ckpt中

#### making csv
```
python scripts/prep-protein-csv.py \
  --input_dir data/pdbxtc \
  --output_csv data/proteins-mdgen2026.csv
```

#### making npy
```
python scripts/prep_sims.py \
  --split data/proteins-mdgen2026.csv \
  --atlas_dir data/pdbxtc \
  --outdir data/npy \
  --num_workers 4 \
  --stride 40 \
  --atlas
```

### T4.1.2-原始MDGen
MDGen-2026以原始MDGen為基礎進行擴充，並保留原本的manual attention執行路徑。測試時不加入`--use_sdpa`即可執行原始attention流程，作為MDGen baseline；其餘設定保持相同，再加入`--use_sdpa`進行SDPA比較。


#### train
```
python train.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --model_dir ckpt \
  --batch_size 1 \
  --prepend_ipa \
  --crop 256 \
  --atlas \
  --run_name origin-mdgen \
  --val_repeat 25 \
  --epochs 10 \
  --ckpt_freq 9 \
  --num_frames 250 \
  --grad_checkpointing
```

#### inference
```
python model/mdgen2026/sim_inference.py \
--sim_ckpt ckpt/atlas.ckpt \
--data_dir data/atlas/npy \
--num_frames 250 \
--num_rollouts 1 \
--suffix _R1 \
--split data/proteins-mdgen2026.csv \
--out_dir data/inference/origin-mdgen \
--xtc
```

## T4.2-測試環境
先使用cu126的conda版本進行測試 cu130的問題會在T4.6進行說明

## T4.3-數值等價性
## T4.4-執行時間
## T4.5-GPU peak memory
## T4.6-cu130 OOM觀察
