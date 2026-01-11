# Eyelid Segmentation Inference

顔画像から眼瞼・虹彩・瞳孔をセグメンテーションし、楕円近似を行う推論パイプライン。

## 概要

2段階の推論パイプライン:
1. **Stage1 (Detection)**: 顔画像から両眼（Right_eye / Left_eye）を検出
2. **Stage2 (Segmentation)**: 検出した眼領域のROIに対してセグメンテーション + 楕円近似

## 必要なモデル

| モデル | パス | 用途 |
|--------|------|------|
| Detection | `YOLO11l-detect.pt` | 両眼検出 |
| Segmentation | `YOLO11n-seg.pt` | 眼領域セグメンテーション |

## セグメンテーションクラス

| クラス名 | 説明 |
|----------|------|
| conj | 結膜（白目の可視部分） |
| caruncle | 涙丘 |
| iris_vis | 虹彩（可視部分） |
| iris_occ | 虹彩（眼瞼で隠れた部分） |
| pupil_vis | 瞳孔（可視部分） |
| pupil_occ | 瞳孔（眼瞼で隠れた部分） |

## 使い方

### 1. 環境セットアップ

```bash
# 依存パッケージ（Colabでは!をつけてコマンドを入れる）
!pip install ultralytics
```

### 2. 推論の実行

`inference.ipynb` を開いて上から順にセルを実行:
パスは実際のフォルダに合わせて変えてください

1. **入力画像の設定** (cell-3)
   ```python
   IMAGE_PATH = r"C:\Users\CorneAI\Eyelid_seg_inference\your_image.png"
   ```

2. **Stage1: 両眼検出** (cell-4〜7)
   - 顔画像から Right_eye / Left_eye を検出
   - 検出結果を `{image_name}_result.jpg` に保存

3. **ROI抽出** (cell-9〜12)
   - 検出したBBoxから25%マージンを付けて512x512のROIを抽出
   - ROI画像を `{image_name}_{eye_name}_roi.png` に保存

4. **Stage2: セグメンテーション** (cell-14〜17)
   - ROI画像に対してセグメンテーション推論
   - 結果を `{image_name}_{eye_name}_seg.png` に保存

5. **楕円近似 & Eyelid輪郭** (cell-19〜21)
   - Eyelid（黄）: conj + iris_vis + pupil_vis の結合マスク輪郭
   - Iris（緑）: iris_vis + iris_occ の楕円近似
   - Pupil（青）: pupil_vis + pupil_occ の楕円近似

## 出力

| ファイル | 内容 |
|----------|------|
| `{name}_result.jpg` | 両眼検出結果 |
| `{name}_{eye}_roi.png` | 抽出したROI画像 (512x512) |
| `{name}_{eye}_seg.png` | セグメンテーション結果 |

## 楕円パラメータ

楕円近似の結果は以下の形式で出力:
```
{eye_name} Iris: center=(cx, cy), axes=(a, b), angle=theta
{eye_name} Pupil: center=(cx, cy), axes=(a, b), angle=theta
```

- `center`: 楕円中心座標 (px)
- `axes`: 長軸・短軸の長さ (px)
- `angle`: 回転角度 (度)

## パラメータ設定

### ROI抽出
- `roi_size`: 512 (出力サイズ)
- `expansion_ratio`: 0.25 (BBoxの25%拡張)

### 推論
- `conf`: 0.25 (信頼度閾値)
- `retina_masks`: True (高解像度マスク)
