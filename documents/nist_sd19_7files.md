# NIST SD19 実験：7ファイル版

[7ファイルだけのZIPをダウンロード](../downloads/nist_sd19_7files.zip?raw=true)

[GitHubで7ファイルを見る](../samples/python/nist_sd19_7files)

ZIPの中身は以下の7ファイルだけです。すべて同じフォルダに置いてください。

1. `fpga_magazin_network_samepadding.py` — FPGAマガジン構成の実行
2. `g_research_miou_replicate_fixed.py` — 卒業研究構成の実行
3. `experiment_config.py` — 設定。旧`config.py`を統合
4. `experiment_dataset.py` — データ生成。旧`download_sd19.py`を統合
5. `experiment_common.py` — 学習・可視化・保存・RTL・短時間テスト・カメラ・速度比較
6. `experiment_metrics.py` — 指標
7. `experiment_lut.py` — 構成情報

旧版`nist_sd19`は残していますが、今回ダウンロードするのは **`nist_sd19_7files.zip`** です。旧版からファイルを追加する必要はありません。`run.py`、`config.py`、`download_sd19.py`、`smoke.py`、`camera.py`、`benchmark_loader.py`は不要です。

## 準備

以前の実験が動いた、BinaryBrainとCUDAを使用できるPython環境を利用してください。ZIPにはPythonやBinaryBrain本体、SD19画像は含めていません。必要な外部パッケージは `numpy`、`Pillow`、`torch`、`matplotlib`、`tqdm` とBinaryBrainです。必要ならその環境にインストールしてください。

```bash
python -m pip install "numpy<2" Pillow "matplotlib<3.10" tqdm
# PyTorchとBinaryBrainは使用するCUDA環境に合わせて導入してください。
```

ZIP内の`nist_sd19_7files`フォルダを、BinaryBrainリポジトリの`samples/python/`内に置きます。

```bash
cd BinaryBrain-work/samples/python/nist_sd19_7files
```

リポジトリのフォルダ名が`BinaryBrain`なら読み替えてください。

## 最初に44×44を短時間確認

データ未配置時は公式SD19を自動ダウンロードします（ZIP約984MB、展開・生成にさらに容量が必要）。既存データがある場合は `--sd19-root /path/to/by_class` を付けます。

```bash
# データ生成・入力画像と教師画像の可視化だけ（学習しない）
python fpga_magazin_network_samepadding.py --size 44 --prepare-dataset-only

# 両構成の短時間テスト（本格学習しない）
python fpga_magazin_network_samepadding.py --size 44 --smoke-test
python g_research_miou_replicate_fixed.py --size 44 --smoke-test
```

`--smoke-test`は両ルートのForward/Backwardと、固定データから最大2枚のDense更新1回、Lossと検証指標を確認します。通常の全層蒸留は実行しません。正常に終了したら次へ進みます。

## 本格学習

```bash
# FPGAマガジン構成：44×44、DW/PW36ブロック
python fpga_magazin_network_samepadding.py --size 44 --fpga-dw-pw-blocks 36 --seed 0 --run-id fpga44_b36_seed0

# 卒業研究構成：44×44
python g_research_miou_replicate_fixed.py --size 44 --seed 0 --run-id grad44_seed0
```

`--size` は22/28/32/44/64/128。保存先は `data/sd19/runs/指定run-id/` です。

初期値は学習4096枚・検証128枚・テスト1024枚、batch16、Dense128epoch、蒸留各層2epoch、変換後各層4epoch、Adam・各学習率0.001、frame modulation5・BIN_MODE=True、背景教師保持率train0.8／val・test1.0です。設定は **`experiment_config.py`** で変更します。元のCLI上書きも維持しています。

全6サイズ・両構成で同じ元画像分割を使います。全域評価が初期値で、`--eval-hw`を指定した場合だけ中央領域評価です。Lossの領域は全域です。背景教師の間引きと入力画像の余白は別の設定です。

## 再開・再評価・データ再生成

```bash
# Dense bestからLUT変換を再開。元と同じ設定を付けてください。
python fpga_magazin_network_samepadding.py --size 44 --fpga-dw-pw-blocks 36 --seed 0 --run-id fpga44_b36_seed0 --resume-from-cnv 0 --allow-existing-run-dir

# 完了済みcnv6のcheckpointからcnv7を開始
python g_research_miou_replicate_fixed.py --size 44 --seed 0 --run-id grad44_seed0 --resume-from-cnv 7 --allow-existing-run-dir

# 学習済みモデルの再評価。元の設定・評価幅を引き継ぐ。
python fpga_magazin_network_samepadding.py --run-id fpga44_b36_seed0 --retest-final

# データ再生成。旧生成物はarchiveへ退避。
python fpga_magazin_network_samepadding.py --size 44 --prepare-dataset-only --force-dataset
```

再開は元コード同様に**層の変換開始時点から**で、epoch途中・Adam状態まで厳密に復元する方式ではありません。

## 補助機能も同じ2つの実行ファイルから

```bash
# 読込と教師float32変換だけの速度比較。学習はしない。
python fpga_magazin_network_samepadding.py --size 44 --benchmark-loader --benchmark-report loader44.json

# 保存済みモデルで640×132カメラ合成・分割推論
python fpga_magazin_network_samepadding.py --camera-run-dir ./data/sd19/runs/fpga44_b36_seed0 --camera-checkpoint best_lut --camera-output ./data/sd19/camera44
```

卒業研究モデルのカメラ推論は同じオプションを `g_research_miou_replicate_fixed.py` に付けます。パッチ境界のコンテキストが失われるため、全画像一括推論と同等ではありません。グラフ・class IoU・checkpoint・CSV/JSON・RTLの機能は従来どおりです。

## 今回の統合と確認範囲

- 学習ループ・評価ループ・ネットワーク構築・通常実験ドライバは統合前とAST比較で一致。
- ZIPは**7ファイルだけ**。ZIPから隔離ディレクトリへ展開して既存9テストを再実行し成功。
- 自動ダウンロードは小さいテストZIPと模擬HTTP応答で展開・数字クラス選択・SHA256・再利用を確認。
- 追加オプションが通常の長時間学習に入らず、それぞれの補助機能に分岐することを確認。
- 実SD19の44×44生成キャッシュを7ファイル版から正常に読込。
- 統合前には実SD19の全6サイズで4096/128/1024枚の生成と分割共有を確認済み。
- このクラウドのBinaryBrain CPUビルド障害は未解決です。**実Forward/Backward・学習・RTL・カメラ数値推論の成功は未確認**です。今回も長時間学習は実行していません。

統合前の詳しい研究条件・検証記録は[旧版の説明](../samples/python/nist_sd19/README.md)を参照できます。ただし実行時の追加ファイルや旧`config.py`は今回の7ファイル版には不要です。
