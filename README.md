# hf-model-mirror

依一份 TOML 清單，把 Hugging Face 上的開源模型批次下載到指定路徑（例如外接硬碟）。

- **換清單、換路徑只要改設定檔**：`models.toml` 裡寫目標路徑和模型清單，也可以用 `--root` 臨時覆寫路徑。
- **固定版本**：`plan` 會把每個模型固定在某個 commit，並記下每個檔案的大小與 SHA256。
- **去重**：不同模型裡 SHA256 相同的檔案只存一份，其他模型用硬連結共用。例如原版和微調版共用的分片，只下載一次。
- **可中斷**：隨時暫停或關機，重跑 `download` 時，未完成的模型會先校驗已下載的檔案、跳過它們，再接著下載剩下的。已完成的模型每次會快速檢查檔案是否都在、大小是否相符（不算 hash），有缺才補（內容庫有就直接連結）；要完整重新校驗請用 `verify`。
- **保守**：checksum 不符、空間不足、磁碟換了，就停止並保留所有檔案；不覆蓋、也不刪除任何檔案。

## 安裝

需要 Python 3.11 以上。

```powershell
git clone https://github.com/ChiJiun/hf-model-mirror.git
cd hf-model-mirror
pip install -e .
hf auth login      # 選填：下載需要授權的模型時才需要
```

## 使用

```powershell
copy models.example.toml models.toml   # 改成自己的清單與路徑
hfmirror plan        # 查詢 commit 與檔案清單，寫入 models.lock.json（每個設定檔各有一份 <檔名>.lock.json）
hfmirror status      # 看各模型大小、去重後還要下載多少
hfmirror download    # 開始下載（Ctrl+C 或 hfmirror pause 可停止，重跑即接續）
hfmirror verify      # 離線重新校驗（不下載）
```

- 結束碼（exit code）：`0` 全部完成；`1` 下載或校驗出錯（已停止並保留檔案）；`2` 設定檔、鎖定檔或參數錯誤；`3` 其他模型都完成，但有模型被略過（`plan` 時標為 blocked，例如需授權的 repo）；`130` 暫停或 Ctrl+C。
- 其他設定檔：`hfmirror -c D:\lists\llm.toml plan`
- 臨時換路徑：`hfmirror --root F:\models download`（相對路徑以目前工作目錄為準）
- 同一個資料夾可以放多份清單（例如 `llm.toml`、`embed.toml`），各自產生 `llm.lock.json`、`embed.lock.json`，互不影響。
- 設定檔的區塊名稱或欄位打錯（例如 `[[model]]`），會直接報錯；`--model` 指定清單裡沒有的 repo 也會報錯。
- 只處理某個模型：`hfmirror download --model BAAI/bge-m3`
- 改了清單（新增、刪除模型，或改 `revision`、`include`、`exclude`、`enabled`、`category`）之後要重跑 `plan`；沒重跑的話，`download` 會提醒你。
- 重跑 `plan` 時，設定沒變的模型會沿用已固定的 commit，不會因為上游更新而重新下載。要全部更新到最新版，請用 `hfmirror plan --update`。
- 欄位型別寫錯會直接報錯，例如 `include = "*.gguf"` 必須寫成 `include = ["*.gguf"]`；不會默默變成下載整個 repo。

### 設定檔

```toml
[settings]
root = "D:/AI-Models"     # 目標路徑
reserve_gb = 50           # 每個檔案下載前，磁碟至少要留下的空間
# volume_guid = "..."     # 選填（Windows）：磁碟卷 GUID 不符就停止

[[models]]
repo = "BAAI/bge-m3"                        # 必填
revision = "main"                           # 分支、tag 或 commit
category = "embedding"                      # 放在 root 底下的子資料夾
include = ["*.safetensors", "*.json"]       # 只下載符合的檔案（glob）；不填 = 全部
exclude = ["onnx/*"]
enabled = true
```

清單順序就是下載順序。完整範例見 [`models.example.toml`](models.example.toml)。GGUF repo 通常一次放多種量化，請用 `include` 指定要的那一種，例如 `["*Q4_K_M.gguf"]`。

## 架構

```text
<root>\
  <category>\<owner>--<name>--<commit>\   完整模型目錄，檔名與 HF 相同，推論引擎可直接讀
    .cache\huggingface\download\          未完成的下載（續傳用）
  blobs\sha256\<前兩碼>\<SHA256>         內容庫：已驗證的完整 LFS 檔，與模型目錄裡的檔案是同一份資料（硬連結）
  .hfmirror\
    state.json / PROGRESS.md              進度（機器讀／人讀）
    links.jsonl                           每次建立硬連結的紀錄
    run.lock / pause                      單一實例鎖／暫停要求
```

```text
hfmirror\
  config.py   讀 TOML、檢查欄位
  plan.py     向 HF API 查 commit 與檔案（含 LFS SHA256），套用 include/exclude，寫鎖定檔
  store.py    算 checksum、登記內容庫、建立硬連結
  runner.py   download / status / verify 主流程、暫停、單一實例鎖、空間與磁碟檢查
  cli.py      命令列
```

`download` 依清單順序處理每個檔案：

1. 檔案已存在：重新校驗；相符就跳過，不符就停止（不覆蓋）。
2. 檔案不存在，但內容庫已有同一個 SHA256：先驗證內容庫那份，再建立硬連結，不需要下載。
3. 都沒有：用 `huggingface_hub` 下載，校驗後登記到內容庫，讓後面的模型共用。

小檔（config、tokenizer 等）用 Git blob SHA1 校驗，大檔用 LFS SHA256 校驗。

## 注意

- **硬連結不是獨立副本**：改到其中一個路徑的內容，所有共用這份資料的模型都會一起被改到。不要直接在備份目錄裡量化、微調或編輯權重，要用請先複製出去。
- 硬連結只能在同一個磁碟卷內建立；`root` 底下必須是同一個卷（NTFS、ext4、APFS 等都支援；exFAT／FAT32 不支援，`download` 開始前會檢查並停止）。
- Linux／macOS 上沒有磁碟卷 GUID 檢查：如果外接碟沒掛載，`root` 會建在系統碟上。下載前請先確認已經掛載。
- Windows 路徑超過 260 字元又沒有開啟長路徑支援時，校驗可能找不到檔案而停止（不會損壞檔案）。目標路徑盡量設短一點。
- 檔案總管顯示的資料夾大小會重複計算硬連結；空間夠不夠請看磁碟可用空間，或看 `hfmirror status`。
- 複製到別顆硬碟時，請用能保留硬連結的工具，否則共用的檔案會被展開成多份。複製後執行 `hfmirror verify --relink`，可以從內容庫補回缺少的連結。
- 單一檔案的續傳交給 `huggingface_hub` 處理；已校驗完成的檔案一定會跳過。
- 需要授權的模型（gated）如果還沒申請到權限，`plan` 會把它標成 blocked，下載時略過，不影響其他模型。

## 開發

```powershell
pip install -e .[dev]
pytest
```

測試用假的 HF 端，涵蓋：硬連結共用、重跑不重新下載、損壞檔案不覆蓋、暫停後接續、暫時性錯誤重試、blocked repo 略過、清單變更要求重跑 plan、補回硬連結、空間不足時停止。
