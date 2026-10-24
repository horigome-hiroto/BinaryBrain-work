# NIST SD19 数字サイズ比較実験

**ダウンロード:** [実験プログラム一式（ZIP）](nist_sd19_program.zip?raw=true)。リンク先でダウンロードし、解凍後の `はじめに.txt` を開いてください。ZIPには追加モジュールも含まれます。元の7ファイルだけでは実行できません。

添付7ファイルを複製して拡張した独立した実験ディレクトリです。既存MNISTファイル・過去の結果・BinaryBrain本体は変更していません。

**実装と数値検証は別です。この環境では公式SD19の取得と実データ生成を確認しました。ただしBinaryBrain拡張のCPUビルド障害により、実SD19での学習動作は未確認です。長時間学習は実行していません。** `validation_report.json` を参照してください。

## 維持した処理／変更した処理

| 項目 | 方針 |
|---|---|
| DenseAffine→層ごとのLUT蒸留→順次LUT変換・tail学習 | 添付ドライバ・ループを再利用 |
| 学習・検証・テスト枚数 | 4096 / 128 / 1024 |
| batch / Dense epoch / 蒸留 epoch / 変換後 epoch | 16 / 128 / 各層2 / 各層4 |
| optimizer / 学習率 | Adam / 全体・蒸留・tailとも0.001 |
| frame modulation / BIN_MODE | 5 / True |
| 背景教師保持率 | train 0.8 / val・test 1.0（画像余白の割合とは別） |
| 教師 | 元コード同様、事前生成したuint8 11ch One-Hot。バッチでfloat32へ変換 |
| バッチ内処理・評価頻度 | 元コード維持。毎バッチの指標計算、各epochのtrain_eval・validationも維持 |
| グラフ・best保存・CSV/JSON・RTL | 元コード維持。グラフ頻度は既存オプションで変更可能 |
| データ入力 | MNISTのtorchvision読込をSD19 PNGに変更。3×3合成を廃止し単一数字にする |
| サイズ | 22 / 28 / 32 / 44 / 64 / 128。28固定の教師・出力検査を一般化 |
| パディング | 添付ファイルのSame/replicateを維持。全3×3、Dense・LUT両方に指定 |
| 評価範囲 | 初期値は入力全域。`--eval-hw` 指定時のみ中央正方形。Lossは常にモデル出力全域 |
| 蒸留学習率のCLI指定 | 添付では未使用だった値をAdamに渡す。初期値0.001は変わらない |
| 計測 | 元のepoch更新時間に加え、関数単位の評価・グラフ・保存・構築・RTL時間を記録 |

`provenance.json` に添付原本のSHA256と、変更していない学習・評価5関数のASTハッシュがあります。テストで比較します。全体の書き直しや別モデルへの置き換えはしていません。

## ネットワーク

- FPGAマガジン: 初期DW/PWは32/32ch、中間256/32ch、最後256/64ch、最終PWは36→11ch。DW候補`random`とPW`serial`を維持。`--fpga-dw-pw-blocks` は任意の正整数（初期値36）。
- **1ブロックの場合はユーザー指定に従い先頭32chを優先**。中間・最終DW/PWは存在せず、32→36→11となります。2以上の場合の最終64chは変更しません。
- 卒業研究: 空間29層＋PW2層。36chとhidden 36/72/144/288ch、random/depthwise/serial接続、最終72→11chを維持。これらの層数は変更対象にしていません。
- `--dry-run` は構成記述のみで、BinaryBrainのForward/Backward検証ではありません。

## データ生成と公平性

公式SD19 `by_class/30`〜`39`（ASCII数字の16進表現）の**元128×128 PNG**のみ読込みます。大文字小文字は含めません。`hsf_*` があればその中だけを使用し、重複した`train_*`集合を避けます。さらにグレースケール画素のSHA256で重複を除外します。

ソート済み候補パスをseedで並べ替え、必要枚数まで画素重複を除外して非重複のtrain/val/testを固定し、`sources_*.json` に選択元パス・ラベル・ファイルSHA256・画素SHA256を保存します。同じデータルート・seed・枚数・commonディレクトリなら、全サイズ・両ネットワークで同じ元画像を使います。**筆者(writer)単位の独立分割ではありません**。クラス均等数を強制する変更も加えていません。

前景閾値でbboxを切り出し、縦横比を保って長辺を`occupancy × 画像サイズ`にし中央配置します。初期値は0.8。元画像を反転して白い数字・黒い背景へ統一します。二値化の閾値は0.5、補間はbilinear。deskew・回転・拡張は行いません。占有率は長辺比で、前景面積比ではありません。縮小後に前景が消えた画像はエラーにします。

train背景だけを固定乱数で間引き、教師の全chが0の画素を無視画素とします。val/testの背景はすべて教師にします。入力画像の背景は間引きません。教師マスクは一度生成して固定し、epochごとに変えません。

`config.py` を編集、または `--occupancy` / `--threshold` / `--polarity dark|light` / `--interpolation nearest|bilinear|bicubic` / `--bg-keep-train` / `--bg-keep-eval` で変更できます。

前処理と分割からキャッシュIDを生成し、サイズ間でキャッシュを混同しません。初回にfloat32入力・uint8教師を`.npy`へ生成し、以後はmmap。元コードと同じDataLoader（workers=0、shuffle用seed固定）を利用します。毎バッチのPNG読込・resize・One-Hot生成はありません。128で教師trainだけで約738 MBあるため、サイズごとに事前生成してください。

`dataset_*/preview.png` の上段=入力、中段=クラス教師（-1無視、0〜9数字、10背景）、下段=有効教師領域。`metadata.json`、`sources.json` と一緒に確認してください。初回の候補一覧作成・選択画像のハッシュ計算・ZIP展開は重い前処理で、学習時間とは分離します。共有キャッシュの同時再生成は避け、先に順番にprepareしてください。

## 準備

```bash
cd /workspace/BinaryBrain-work
source /workspace/.venvs/binarybrain/bin/activate
pip install -r samples/python/nist_sd19/requirements.txt
export PYTHONPATH=/workspace/BinaryBrain-work/python${PYTHONPATH:+:$PYTHONPATH}
cd samples/python/nist_sd19
```

GPU実験では、別途CUDA対応PyTorchとこのリポジトリのBinaryBrainを用意してください。PyTorchをGPUのForwardエンジンに置き換えることはありません。現環境の`validated-environment.txt`はデータ系テスト時のバージョン一覧で、GPU実験の推奨構成ではありません。

データがなければ自動取得します。取得先: `https://s3.amazonaws.com/nist-srd/SD19/by_class.zip`。大きなZIPを保存後、数字クラスのみ展開します。TLS検証・展開時CRCを維持し、ローカルSHA256を記録します。このSHAは配布者公表の照合値ではありません。手元の展開済みデータがあれば `--sd19-root /path/to/by_class` を指定できます。

このクラウドではHEAD確認は403でしたが、実際のGET取得・数字クラス展開・CRC検証は成功しました。データは `/workspace/.deps/sd19-download-check` に保存し、この実験の `data/sd19/raw` から参照しています。将来の実行用に `s3.amazonaws.com` の許可追加も設定下書きに保存済みです。設定の公開・反映を済ませたという意味ではありません。

```bash
# 明示的な取得だけ
python download_sd19.py --root ./data/sd19/raw
# 44の準備だけ。BinaryBrain不要。両モデルで同じキャッシュ。
python run.py --model fpga_magazine --size 44 --prepare-dataset-only
# 再生成: 前の生成物をarchiveへ退避。元画像分割manifestは維持。
python run.py --model fpga_magazine --size 44 --prepare-dataset-only --force-dataset
```

## 最初の短時間検証（本格学習の前に）

```bash
python smoke.py --model fpga_magazine --size 44 --report smoke_fpga.json
python smoke.py --model g_research --size 44 --report smoke_gresearch.json
python -m pytest tests -q
```

`smoke.py` はDenseとLUTのshape・Forward/Backward、固定分割の先頭最大2画像のDense更新1回、Loss・validation指標を検査します。蒸留全層を通した長時間実験ではありません。実データを人工画像に自動置き換えません。`--dry-run`で数値検証を代用しません。GPUがある環境でも通常設定の128epochを実行する前にこれを通してください。

## 学習・再開・再評価

以下は**これから実行するコマンド**で、本環境で実学習確認済みではありません。

```bash
python run.py --model fpga_magazine --size 44 --fpga-dw-pw-blocks 36 --seed 0 --run-id fpga44_b36_s0
python run.py --model g_research --size 44 --seed 0 --run-id grad44_s0
# 12条件の例（長時間学習。今回未実行）
for size in 22 28 32 44 64 128; do
  python run.py --model fpga_magazine --size "$size" --seed 0
  python run.py --model g_research --size "$size" --seed 0
done
# Dense bestからLUT変換を再開
python run.py --model fpga_magazine --size 44 --fpga-dw-pw-blocks 36 --seed 0 --run-id fpga44_b36_s0 --resume-from-cnv 0 --allow-existing-run-dir
# cnv6完了時点からcnv7へ（元と同じ設定を付ける）
python run.py --model g_research --size 44 --seed 0 --run-id grad44_s0 --resume-from-cnv 7 --allow-existing-run-dir
# 保存した設定とbest_dense/best_lutで再評価。通常は元の評価範囲を維持。
python run.py --model fpga_magazine --run-id fpga44_b36_s0 --retest-final
# 中央40評価は別のre_evaluationsへ保存。全域44と混ぜない。
python run.py --model fpga_magazine --run-id fpga44_b36_s0 --retest-final --eval-hw 40
```

run ID省略時は構成・サイズ・DW/PW数・seed・評価幅・全設定ハッシュで保存先を分離します。明示IDは既存ディレクトリがあれば停止します。再開は設定一致を検査し、データ再生成と併用できません。

**再開の制限は元コードと同じ**: Dense学習途中／蒸留途中／epoch途中の正確な再開ではありません。完了済み層のcheckpointから再開し、shuffleの進行を復元しますが、全乱数状態・Adam内部状態の厳密な復元はしません。途中中断した層はその層の開始からやり直してください。

初期学習条件は`config.py`管理。`--batch-size`、`--dense-epochs`、`--distill-epochs`、`--cnv-epochs`、3種の`--*-learning-rate`、3種の`--num-*-tiles`、`--frame-modulation-size`、`--bin-mode/--no-bin-mode`等で明示変更できます。比較実験では同じ条件を使ってください。`--eval-hw`は入力以下で偶数差の中央正方形です。Accuracy/mIoU/FG mIoU/Class IoUは指定範囲、Lossは全域です。背景の存在しないクラスのIoUは元コード通りNaNとして平均から除きます。

## 保存・速度

各runのconfig/architecture、checkpoints（best_dense/best_lut/各cnv）、logs（metrics・class_iou・training_time・distillation・phase_timings）、graphs、rtlを保存します。Dense後とLUT後のtest精度は別ステージで保存。グラフ・class IoU棒グラフ、CSV/JSON出力、RTL書出し処理は引き継いでいます。

- `training_time.csv`: 元コードと同じtrain_update／distillation所要時間。
- `phase_timings.jsonl`: 学習epoch・評価・plot・checkpoint保存・構築・RTLのホスト経過秒。GPU専用カーネル時間ではなく、非同期処理の同期位置にも影響されます。入れ子の時間を単純加算しないでください。
- データ生成時間: datasetのmetadata。図は初回生成だけ。mmapの初回page faultとウォーム時を区別してください。
- `python benchmark_loader.py --run-dir ./data/sd19/runs/fpga44_b36_s0 --output loader_speed.json`: 同じデータ・DataLoader・変換で、従来形式のin-memory配列とmmapを比較。**ネットワーク学習速度の比較ではありません**。
- 詳細なPython側時間は必要時だけ `python -m cProfile -o smoke.prof smoke.py ...`。デフォルトのバッチ処理にprofilerは追加していません。
- 空間演算量は概ね画素数比で増えます（128対44は約8.46倍）。Same Paddingはvalidで途中縮小する構成より後段の演算量が多くなります。添付2ファイル自体は既にSame指定でした。ネットワーク構造・評価頻度・dtypeを勝手に省略する高速化はしていません。

`--no-plot`、`--plot-every`、`--dense-class-iou-bar-every`、`--no-rtl`を使う場合、旧コードとの速度比較でも同じ値に揃えてください。

## カメラとRTL

```bash
python camera.py --run-dir ./data/sd19/runs/fpga44_b36_s0 --checkpoint best_lut --output ./data/camera44
```

640×132の合成入力・教師・推論マスク・比較図・条件JSONを出力します。元の正方形モデルのshapeとパラメータを維持するため、入力サイズのパッチに分割して推論します。端パッチはreplicateで補います。パッチ境界を越えるコンテキストは失われ、全画像一括推論とは一致しません。halo重ね合わせは未実装です。配置は正方形単位なので、28では高さ132に完全な28パッチは4個です（研究目的の「約5個」は概数）。

一括640×132はConvolution2dの内部shape／Switcher／モデル読込の再設定検証が未完了のため、動作可能と断定しません。カメラ機能・RTL生成とも実モデル未構築のため実行未確認です。RTL出力は既存exporterを利用し、replicate境界・クロック／stream動作・KV260上の一致はRTLシミュレーションと合成で別途検証が必要です。未検証RTLをカメラ評価の正解とは扱わないでください。

## 新規ファイル

- `run.py`: 構成選択入口。
- `fpga_magazin_network_samepadding.py` / `g_research_miou_replicate_fixed.py`: 添付ドライバを必要箇所だけ拡張。
- `config.py` / `experiment_config.py`: SD19設定と元import名の互換入口。
- `experiment_dataset.py` / `download_sd19.py`: 公式PNG取得、分割、前処理、教師、キャッシュ、可視化。
- `experiment_common.py` / `experiment_metrics.py` / `experiment_lut.py`: 元の学習・グラフ・指標・構成記述を再利用、サイズ・計測等を拡張。
- `smoke.py` / `camera.py` / `benchmark_loader.py`: 短時間数値検証、分割カメラ推論、読込ベンチ。
- `tests/test_sd19.py`: データ・指標・元ループ保持・記録用構築検査。
- `requirements.txt` / `validated-environment.txt` / `provenance.json` / `validation_report.json` / `.gitignore` / 本書。

## 今回の確認結果

- 公式SD19 ZIPの取得・数字PNG展開が成功（TLS・読み出したZIPメンバのCRCを検証）。ZIP SHA256は`validation_report.json`に記録。
- 全6サイズで実SD19の4096/128/1024枚を生成。全サイズのmanifest ID一致、元画像画素ハッシュの分割間非重複、入力と教師のshape、前景ラベル、有効画素数、val/test背景教師の全保持を確認。
- 44のtrain背景教師保持率は **0.7999355598**、val/testは **1.0**。実際の入力・教師プレビューを目視確認。
- 自動テスト **9 passed**。Matplotlib/pyparsingの非推奨警告14件のみ。依存関係チェックも成功。
- 44で4096枚を読むloader-only比較のウォーム3回中央値: 従来in-memory配列 **0.05120秒**、mmap **0.05121秒**。教師float32変換までで、ネットワーク・GPU処理は含めない。単一環境の小さな差から高速化を主張しない。
- 実際の44の両構成スモークコマンドはどちらも **binarybrain.core読込で停止**。元リポジトリのCPUビルドエラー`Concatenate.h:264`が前提障害。Forward/Backward、BinaryBrain Loss、短時間学習、実学習速度、実カメラ推論、RTL実行は未検証。
- 記録用コンストラクタによる構築チェックは、全6サイズ・両ネットワーク（FPGAは1/36ブロック）で一致。これは数値演算テストとは区別している。

実データプレビュー: `data/sd19/common/dataset_878c074885add55bda67/preview.png`。
