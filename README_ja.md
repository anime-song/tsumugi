# tsumugi

![tsumugi — あらゆる楽器を MIDI に採譜](resources/tsumugi.svg)

**あらゆる楽器を MIDI に採譜。** Neural Semi-CRF を基盤とする、単一の楽器非依存モデルです。

[English README](README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/tsumugi/blob/main/Colab_Inference.ipynb) | [MIT License](LICENSE)

| [![採譜例](https://img.youtube.com/vi/3pCAjQuhzDA/0.jpg)](https://youtu.be/3pCAjQuhzDA)<br>**採譜例** | [![元動画](https://img.youtube.com/vi/JuVu-AoC5M0/0.jpg)](https://www.youtube.com/watch?v=JuVu-AoC5M0)<br>**元動画** |
| --- | --- |

どちらの画像もクリックすると YouTube の動画が開きます。

---

## 最新アップデート（2026-11-23）

採譜した MIDI を一度 FluidSynth で鳴らして元の音源と聴き比べながら仕上げる後処理を追加し、[SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined) も 0.5.2 に更新しました（`Colab_Inference.ipynb` ではすべてデフォルト ON です）。

- **楽器の選び直し** — 候補の GM 音色を実際に鳴らして元のステムと比べ、いちばん近い音色を選びます。
- **途切れた持続音の修正** — ストリングスや管楽器などで 1 つの音が細切れになってしまう箇所をつなぎ直します（本来の連打はそのまま残します）。
- **リズムの補正** — 検出したビートに合わせてタイミングを整えます。3 連符やスライドなど、揃えると不自然になる箇所は元のタイミングを保ちます。
- **音量バランスと左右の定位** — 元の音源に近づくようノートごとの強弱・トラック音量（CC#7）・パン（CC#10）を調整します。左右に振られたバッキングと中央のソロが混ざるギターなどは別トラックに分けます。
- **SheetSage2-Refined 0.5.2 対応とボーカル改善** — ビート推定を 0.5.2 に更新し、ボーカルの主旋律にも SheetSage2-Refined を使うようにしました。分離したボーカル音声を使って楽器メロディの混入を防ぎ、tsumugi で採譜したハモリと組み合わせて、主旋律とハモリの取り違えやタイミングのズレを抑えています（スウィングにも対応）。
- **Colab での使いやすさ改善** — FluidSynth と SoundFont（`MuseScore_General.sf2` / MIT ライセンス）を自動で用意します。各処理の ON/OFF や速度設定（`fast` / `balanced` / `quality`）もフォームから選べます。

<details>
<summary><b>更新履歴</b></summary>

| 日付 | 更新内容 |
| --- | --- |
| 2026-10-17 | 🥁 `beat_source="sheetsage2"` を SheetSage2-Refined 0.2.0 のビートトラッカーに切り替え。デコーダーを使わずにビート・ダウンビート・小節ごとの拍子を推定します。変拍子の曲でダウンビート F が 0.835→0.883、小節の検出率が 0.661→0.752 に向上し、GPU で 1 曲数秒で動きます。セクションはデコーダーの出力だったため、`Predicted Sections` トラックは出力しなくなりました。 |
| 2026-10-14 | 🥁 ビート・ダウンビート・拍子の推定に [SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined) を追加し、デフォルトに設定（`beat_source="sheetsage2"`。`"tsumugi"` で従来のビートモデル）。ビートの推定精度が向上し、変拍子や拍子の変化にもより対応できるようになりました。曲のセクション（intro、verse、chorus など）も `Predicted Sections` トラックに出力。 |
| 2026-09-01 | 🥁 Drum model v1.5（`--type drums_v1_5`）を追加し、Colab のドラムステムのデフォルトに設定。実音源評価セットで exact F1 は Drum Kit が 0.6157→0.6890（+7.3ポイント）、All Percussions が 0.1879→0.4044（+21.7ポイント）に向上。 |
| 2026-08-25 | 🎤 Vocal harmony model v1.5（`--type vocal_harmony_v1_5`）を追加し、Colab の `vocals` ステムのデフォルトに設定。MIR-ST500 のホールドアウト分割で COnP が 0.6052 から 0.6814 に向上。 |
| 2026-08-20 | ⚡ uv と PyTorch 2.13 へ移行。MPS 推論、AMP、regional compile の制御を追加し、デバイス同期、一時コピー、ステムの重複読み込みを削減。意図的に出力が変わる変更が 2 点あり、CUDA で attention を暗黙に低精度化しないように変更（FP32 がデフォルト）し、V1 のウィンドウバッチでデコード状態をウィンドウ順に伝播するように変更。 |
| 2026-08-19 | 🎻 Other-instrument model v1.5（`--type other_v1_5`）を追加し、Colab の `other` ステムのデフォルトに設定。実音源評価セットで COnP が 0.7318 から 0.7701 に向上。 |
| 2026-08-09 | 🎹 Instrument Refinement model を追加。分離ステムオーディオを使って楽器クラスを再割り当てし、曲中で楽器分類が頻繁に切り替わる問題を抑制。RWC-I の overall top-1 は 71.3% から 74.5% に向上する一方、楽器ごとには改善と悪化の両方があるため、[RWC-I ベンチマーク](instrument_agnostic_amt/instrument_refinement/RWC_BENCHMARK.md)を参照。 |
| 2026-07-30 | パイプラインにビート、コード、キーの推論モデルを追加。デフォルトでは無効。 |
| 2026-07-24 | 専用のベロシティ予測モデルを追加し、Colab のステムワークフローでデフォルト有効化。 |
| 2026-07-22 | 🎸 Guitar model v1.5（`--type guitar_v1_5`）を追加し、Colab のギターステムのデフォルトに設定。 |
| 2026-07-16 | 🐛 ノートタイミングがずれるデータ拡張のバグを修正し、`bass_v2` を再学習。ノート検出を改善し、持続音の過分割を解消（`bass_v2` のみ）。 |
| 2026-07-15 | 🎸 スラップベースの分類を改善した更新版ベースモデル（`--type bass_v2`）を追加。 |
| 2026-07-12 | 🎯 ステムごとの楽器クラス選択を追加。確率計算前に不適切な楽器を除外。 |
| 2026-06-24 | 🥁 実験的なドラム特化モデル（`--type drums`）を追加。 |
| 2026-06-05 | 🎻 その他の楽器向けモデル（`--type other`）を追加。 |
| 2026-05-31 | 🎤 Vocal harmony model と `vocal_harmony` クラスを追加。🧩 重複するノート区間のための Pitch Slot を追加。 |
| 2026-05-20 | 🎸 ギター特化モデル（`--type guitar`）を追加。 |
| 2026-05-18 | 📦 ピッチシフト／タイムストレッチの前処理スクリプトを追加。 |
| 2026-05-17 | 🎤 ボーカル特化モデル（`--type vocal`）を追加。 |
| 2026-05-16 | 🎸 ベース特化モデル（`--type bass`）を追加。 |
| 2026-05-09 | 🔧 ウィンドウをまたぐノート結合を修正。ビート／コード学習と Colab のステムワークフローを追加。 |
| 2026-05-06 | 🥁 ドラム検出を改善し、新しいデータ拡張を追加。 |
| 2026-05-05 | ✨ EMA、楽器損失のマスキング、ディレクトリ一括推論を追加。 |
| 2026-05-03 | 🚀 初回リリース。 |

</details>

---

## クイックスタート

最も手軽なのは [`Colab_Inference.ipynb`](Colab_Inference.ipynb) です。曲をアップロードしてセルを実行し、MIDI をダウンロードするだけです。環境構築やチェックポイント探しは必要ありません。

ローカルで実行する場合：

```bash
git clone https://github.com/anime-song/tsumugi.git
cd tsumugi
uv sync --locked
uv run python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav
```

初回実行時にチェックポイントが Hugging Face から自動的にダウンロードされます。

---

## 概要

オーディオを MIDI に変換する、**楽器非依存の自動採譜（Automatic Music Transcription: AMT）** モデルです。

[Basic Pitch](https://github.com/spotify/basic-pitch) と同じように、演奏している楽器を問いません。ピアノ、ギター、ベース、ボーカル、弦楽器、金管楽器など、音程があれば採譜できます。すべてを 1 つのモデルで扱います。

アーキテクチャは、Yujia Yan 氏らによる [**Transkun**](https://github.com/Yujia-Yan/Transkun) で、もともとピアノ向けに設計された Neural Semi-CRF の手法を拡張しています。本プロジェクトでは、これをあらゆる楽器へ一般化しています。

### 得意な音源と苦手な音源

クリーンで持続音が多く、ステムごとには単音である素材で最も高い採譜品質が得られます。学習データが少ない領域では精度が下がります。

- **エレキギター**、特にディストーションをかけた音は汎化が難しく、採譜精度が低くなります。
- **民族楽器**（三味線、シタールなど）は学習データが少なく、同様に精度が低くなります。
- **速いボーカルフレーズやスウィングのフレージング**では、音高やノート境界が不安定になります。
- 1 つの持続音が**複数の短いノートに分割される**ことがあります（過分割）。
- **ステム分離ワークフロー**では、ステム間に小さなタイミングのずれが生じ、MIDI のマージ後に同期ずれとして現れることがあります。

**楽器分類とマルチトラック出力**は動作しますが、分類精度にはまだ制約があります。スラップベースやシンセベースは `electric bass` にまとめられやすく、アコースティックピアノとエレクトリックピアノは頻繁に混同されます。また、分離したギターステムに混入したシタールやバンジョーは誤ったクラスへ分類されることがあります。**ドラムモデル**（`--type drums_v1_5` および従来の `--type drums`）も実験段階で、今後挙動が変わる可能性があります。中核機能である楽器非依存の音高検出は実験機能ではありません。

---

## 特徴

- 🎹 **あらゆる有音程楽器** — ピアノ、ギター、ベース、ボーカル、弦楽器、管楽器などを 1 つのモデルで採譜
- 🎛️ **実音比較による MIDI 仕上げ** — FluidSynth で鳴らした音を元音源と比べ、楽器選定・途切れた持続音の修正・リズム補正・音量やパンの調整を自動で実行
- 🎚️ **ノート単位のベロシティ** — 専用の予測モデル、または実音比較による音量調整で強弱を再現
- 🎯 **ビート、コード、キー** — SheetSage2-Refined や内蔵モデルでビート・拍子・コード・キーを推定
- 🧪 **［実験的］楽器分類** — 36 クラスの分類と実音比較による GM 音色選定で楽器別トラックを出力

---


## インストール

**必要なもの**

- Python 3.10～3.14
- [uv](https://docs.astral.sh/uv/)
- PyTorch 2.13.0 / torchaudio 2.11.0（`uv.lock` から自動インストール）
- NVIDIA GPU（VRAM 12 GB 以上を推奨）、Apple Silicon Mac、または CPU

Linux と Windows では CUDA 13.0 wheel が解決されます。Apple Silicon macOS ではプラットフォーム用 wheel が解決されます。PyTorch 2.13 は Intel macOS 用 wheel を提供していません。

```bash
git clone https://github.com/anime-song/tsumugi.git
cd tsumugi

# uv.lock に固定された推論用コア依存パッケージ
uv sync --locked

# 用途別のオプション依存パッケージ
uv sync --locked --extra stem        # ステム分離推論
uv sync --locked --extra evaluation  # 評価スクリプト
uv sync --locked --extra training    # 学習
```

`uv sync` は `.venv/` を作成します。`source .venv/bin/activate` で有効化するか、コマンドの前に `uv run` を付けてください。

`.python-version` は開発時のデフォルトとして Python 3.12 を選択しますが、サポート範囲は 3.10～3.14 のままです。3.12 がない場合、ダウンロードが無効化されているかオフラインでない限り、uv が管理対象の CPython をダウンロードします。

**動作確認済み：** Apple Silicon（M4 Pro、macOS / MPS）および CUDA（Colab Tesla T4）。

**テスト：**

```bash
uv sync --locked --all-extras
uv run pytest
```

MPS が利用できない環境では MPS 固有のテストをスキップします。compile の回帰テストはオプトインです。

```bash
RUN_ACCELERATOR_COMPILE_TEST=1 uv run pytest tests/test_mps_inference.py
```

---

## 推論

```bash
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav
```

`--checkpoint` を指定しない場合、モデルは Hugging Face からダウンロードされます。

### デバイス選択

`--device` のデフォルトは `auto` で、**CUDA → MPS → CPU** の順に利用可能な最初のバックエンドを選びます。バックエンドを明示的に指定した場合、利用できなければ暗黙にフォールバックせずエラーになります。

```bash
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav                # auto
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --device mps   # Apple Silicon GPU
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --device cpu
```

### regional compile（任意）

```bash
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --compile
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --compile --compile-mode reduce-overhead
```

初回のウィンドウでコンパイルのコストがかかるため、短い曲を 1 回だけ採譜する場合は遅くなる可能性があります。
読み込んだモデルで多くの曲を続けて処理する場合に効果があります。`--compile-mode` には `default`、`reduce-overhead`、`max-autotune-no-cudagraphs` も指定できます。

### ステム分離ワークフロー

[`Colab_Inference.ipynb`](Colab_Inference.ipynb) では、曲をステムに分離してそれぞれ採譜したあと、楽器選定・持続音の修正・リズム補正・音量やパンの調整をまとめて行って 1 本の MIDI に仕上げます。

ミックス全体を 1 回で処理するより遅いものの、通常は高い精度が得られます。各ステムの音響が単純になり、楽器の重なりが減るためです。差が特に大きいのは、密度の高いミックス、バンド録音、持続和音にメロディを重ねたアレンジです。

各後処理は個別に ON/OFF でき、`render_compare_speed`（`"fast"` / `"balanced"` / `"quality"`）で速度と精度のバランスを選べます。Colab では FluidSynth が自動で入り（ローカル環境では Ubuntu/Debian: `apt-get install -y fluidsynth`、macOS: `brew install fluid-synth`、Windows: 公式バイナリ）、`soundfont_path` や `TSUMUGI_SOUNDFONT_PATH` を指定しない場合は `MuseScore_General.sf2`（MIT ライセンス）が自動でダウンロードされます。

`beat_source` と `vocal_melody_source` の既定値は `"sheetsage2"` です。[SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined) 0.5.2（デコーダーなし）でビート・拍子とボーカル主旋律を推定し、ボーカルは分離ステム音声で楽器メロディを除いたうえで tsumugi のハモリと統合します。内蔵モデルだけを使いたい場合は `"tsumugi"` を指定してください（コードとキーは常に tsumugi の推定です）。`uv sync --locked --extra stem --extra sheetsage` でインストールできます。SheetSage2-Refined の重み（2.7 GB）は CC BY-NC 4.0（非商用のみ）です。

### 楽器再判定の単体実行

既存 MIDI の各ノートについて、採譜元のステムを使って楽器を再分類します。タイミングとピッチは維持し、トラックのプログラム番号と名前だけを変更します。

```bash
python -m instrument_agnostic_amt.instrument_refinement.cli.infer \
  --audio separated_stems/song_other.wav \
  --midi stem_midis/song_other.mid \
  --stem-name other \
  --output-midi song_other_refined.mid
```

`--stem-name` は、そのステムに妥当なクラスだけに候補を制限します。`--mode cluster`（デフォルト）は音色埋め込みに基づいてノートをグループ化し、グループごとにラベルを付けます。`--mode single` はステム全体に 1 つの楽器を割り当てます。

### ベロシティ予測の単体実行

AMT とは別の後処理モデルです。MIDI ファイルとステムを入力し、トラック、ピッチ、Note On/Off のタイミングを維持したまま、固定ベロシティをノートごとに予測した強弱へ置き換えます。

```bash
python -m instrument_agnostic_amt.velocity.cli.infer_velocity \
  --midi output.mid \
  --stems-dir separated_stems \
  --output-midi output_velocity.mid
```

ステムディレクトリには、`vocals.wav`、`bass.wav`、`drums.wav`、`other.wav` のようにステム名を付けたファイルを置いてください。`--compile-velocity` は、コア AMT の `--compile` とは独立してベロシティバックボーンを regional compile します。学習とデータ準備については [`instrument_agnostic_amt/velocity/README.md`](instrument_agnostic_amt/velocity/README.md) を参照してください。

### 主な引数

| 引数 | デフォルト | 説明 |
| --- | --- | --- |
| `--audio` | （必須） | 入力オーディオのパス |
| `--output-midi` | `<audio>.mid` | 出力 MIDI のパス |
| `--checkpoint` | （自動） | 学習済みモデル。省略時は Hugging Face からダウンロード |
| `--type` | `default` | モデルの種類。下表を参照 |
| `--device` | `auto` | `auto`（CUDA → MPS → CPU）、`cuda`、`mps`、または `cpu` |
| `--amp` | `false` | CUDA または MPS で混合精度を有効化 |
| `--amp-dtype` | デバイスのデフォルト | `fp16` または `bf16` |
| `--compile` | `false` | バックボーンの Transformer ブロックを regional `torch.compile` |
| `--compile-mode` | `default` | `default`、`reduce-overhead`、`max-autotune`、`max-autotune-no-cudagraphs` |
| `--window-ms` | 学習時の値 | 推論ウィンドウサイズ（ms） |
| `--stride-ms` | `window-ms / 2` | ウィンドウのストライド |
| `--window-batch-size` | `1` | 1 バッチあたりのウィンドウ数。小さくするとピークメモリを削減。バッチサイズ間でバイト単位の同一出力は保証されない |
| `--merge-gap-ms` | 1 hop | 小さなノート間隔を結合する閾値 |
| `--merge-onset-ms` | `50.0` | ほぼ同時の onset を結合する閾値 |
| `--max-midi-melodic-instruments` | `15` | 楽器トラックの最大数 |
| `--allowed-instruments` | 全クラス | 楽器分類の候補。選択したクラス内で softmax を再正規化 |
| `--silence-gate-rms-dbfs` | `-72` | 無音ウィンドウをスキップする RMS 閾値 |

**モデルの種類（`--type`）**

| 値 | 備考 |
| --- | --- |
| `default` | 全楽器向け |
| `bass` / `bass_v2` | `bass_v2` が更新版 |
| `guitar` / `guitar_v1_5` | `guitar_v1_5` が Colab のデフォルト |
| `vocal` | リードボーカル向けにファインチューニング |
| `vocal_harmony` / `vocal_harmony_v1_5` | `v1_5` は Pitch Slot が 1 つのため、1 パート内の同時発音を予測しない |
| `other` / `other_v1_5` | `other_v1_5` が Colab のデフォルト |
| `drums` / `drums_v1_5` | `drums_v1_5` が Colab のデフォルト。`drums` は従来チェックポイント（**実験的**） |

---

## データ準備

ステムと対応する MIDI を次の構成で配置します。

```
stems/          # オーディオ（.wav / .flac）
  ├── song1__piano.wav
  ├── song1__guitar.wav
  ├── song2__vocal.wav
stem_midis/     # 対応する MIDI
  ├── song1__piano.mid
  ├── song1__guitar.mid
  ├── song2__vocal.mid
```

命名規則は `<song_name>__<instrument_name>.wav` です。曲名と楽器名を 2 つのアンダースコアで区切ります。同じ曲名を持つステムは 1 つの曲として扱われます。

```bash
python preprocess/prepare_dataset.py \
  --stems_dir ./stems \
  --midis_dir ./stem_midis \
  --npz_dir ./stem_npz \
  --manifest_path ./manifest.csv
```

このコマンドは、前処理済みノート配列（開始、終了、音高、ベロシティ、楽器 ID）を格納する `stem_npz/` と、データセットの索引である `manifest.csv` を生成します。

オーディオが 22050 Hz でない場合は、コンテナ形式とサンプルサブタイプを維持したまま上書きリサンプリングします。

```bash
python -m preprocess.resample_only --input ./stems --resample-rate 22050
```

---

## 学習

```bash
python -m recipes.amt.train \
  --manifest_path manifest.csv \
  --batch_size 8 \
  --lr 5e-4 \
  --epochs 3000 \
  --save_dir checkpoints \
  --wandb
```

すべてのデータ拡張を有効にする場合：

```bash
python -m recipes.amt.train \
  --dataset_config dataset_config.yaml \
  --batch_size 8 --lr 5e-4 --warmup_steps 1000 --epochs 3000 \
  --ir_folder ./IRs --noise_folder ./noise --drum_folder ./drum_stems \
  --p_augment 1.0 --p_intra_drop 0.3 --p_cross_mix 0.5 --p_drum_mix 0.1 \
  --sa_p 0.5 --sa_freq_max 10 --sa_time_max 20 --sa_num_freq 2 --sa_num_time 2 \
  --wandb --project_name tsumugi
```

### 主な引数

| 引数 | デフォルト | 説明 |
| --- | --- | --- |
| `--dataset_config` | `dataset_config.yaml` | 重み付きマルチデータセット設定 |
| `--batch_size` | `8` | バッチサイズ |
| `--lr` | `5e-4` | 学習率（AdamW） |
| `--warmup_steps` | `1000` | 学習率のウォームアップステップ数 |
| `--window_ms` | `8000` | 入力ウィンドウ長（ms） |
| `--p_intra_drop` | `0.3` | 同じ曲のステムをドロップする確率 |
| `--p_cross_mix` | `0.5` | 別の曲のステムをミックスする確率 |
| `--p_augment` | `1.0` | オーディオ拡張を適用する確率 |
| `--init-from` | `None` | 重み初期化に使うチェックポイント |
| `--no_amp` | `false` | 混合精度を無効化 |

### マルチデータセット設定

`dataset_config.yaml` は、異なる重みで複数のデータセットを混合します。

```yaml
datasets:
  - name: main
    manifest: manifest.csv
    weight: 0.2
    use_for_cross_aug: true

  - name: maestro
    manifest: other_db/maestro_manifest.csv
    weight: 0.05
    use_for_cross_aug: true

  - name: musicnet
    manifest: other_db/musicnet_manifest.csv
    weight: 0.5
    use_for_cross_aug: false  # クロスステムミキシングには使用しない
```

同じ曲からレンダリングしたステムを別々の manifest に保存している場合は、`group` を使用します。

```yaml
datasets:
  - name: rendered_piano
    group: single_stems
    manifest: piano_stem_manifest.csv
    allow_multi_stem_same_song: true

  - name: rendered_strings
    group: single_stems
    manifest: strings_stem_manifest.csv
    allow_multi_stem_same_song: true
```

同じ `group` と CSV の `song_name` を持つエントリは 1 つの仮想的な曲になるため、`allow_multi_stem_same_song: true` では複数の manifest にまたがってステムを選択できます。重み、データ拡張設定、クロスデータ拡張の対象可否はエントリごとに維持されます。`group` を省略すると `name` が使われ、従来どおり個別に扱われます。

### データ拡張

**ステムレベル** — 同じ曲のステムをドロップして疎なアレンジを再現、別の曲のステムを混ぜて新しい組み合わせを作成、ドラムのないミックスへランダムにドラムを追加。

**オーディオレベル** — 7 バンド EQ、±0.2 半音のマイクロピッチシフト、実際のインパルス応答を使った IR リバーブ、ガウスノイズと背景音、チャンネル入れ替えとランダムパンニング、ステムごとに ±6 dB のゲイン。

**スペクトログラムレベル** — CQT 特徴量への SpecAugment の時間／周波数マスキング、および基音を常に残しながら倍音チャンネルをランダムに落とす harmonic dropout。

---

## ビート、コード、キー

```bash
# MIDI からビートを事前学習
python -m recipes.beat_chord.pretrain_beat \
  --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis

# ビート／コードを joint 学習
python -m recipes.beat_chord.train --midi_dir midi_dataset/merged

# ビート／コードを推論
python -m instrument_agnostic_amt.beat_chord.cli.infer --checkpoint path/to/checkpoint.pth --midi_path song.mid
```

---

## プロジェクト構成

```
instrument_agnostic_amt/
├── amt/                        # AMT 本体のモデル・推論・データ・CLI
├── taxonomy/                   # 楽器クラスと GM の対応
├── beat_chord/                 # ビート／コード推定、SheetSage2 連携、ボーカル統合
├── instrument_refinement/      # 楽器再判定と GM 音色選定
├── quantization/               # 持続音の結合とビート量子化
├── velocity/                   # ベロシティ予測・音量バランス調整・ステレオパン
├── soundfont.py                # SoundFont の自動ダウンロードと FluidSynth レンダリング
└── cli/                        # 複数モデルを束ねるステム推論

recipes/
├── amt/                        # AMT の train、Dataset、target、loss
├── beat_chord/                 # beat/chord の Dataset と学習
├── instrument_refinement/      # refinement のデータ準備と学習
├── velocity/                   # velocity のデータ準備と学習
└── common/                     # 学習専用の共通 augmentation

preprocess/                     # 独立したデータ準備ツール
```

---

## 謝辞

Neural Semi-CRF の定式化とデコード設計の多くは、Yujia Yan 氏らによる [Transkun](https://github.com/Yujia-Yan/Transkun) に基づいています。本プロジェクトでは、その成果をピアノから任意の楽器へ拡張しています。

## ライセンス

[MIT](LICENSE)
