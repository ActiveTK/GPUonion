# GPUonion

Tor v3 onion service の vanity アドレスを CUDA で探索するツール。
[mkp224o](https://github.com/cathugger/mkp224o) /
[onion-vanity-address](https://github.com/offset/onion-vanity-address) と同系統の
スカラーインクリメント方式を GPU 上で実装したもの。

## アルゴリズム

- ランダムな clamped スカラー `a0` を起点に、スレッド `t` の第 `i` 候補を
  `a = a0 + 8*(t + i*T)`（`T` = 総スレッド数）とする
- GPU 上では Curve25519 (Montgomery) の x-only 差分加算 1 回（3M+2S）で
  次候補へ進む（フルスカラー倍算は初期化時の1回のみ）。Edwards の y は
  双有理写像 y = (U−W)/(U+W) から復元し、分母はバッチ逆元で償却
- インクリメントを 8 の倍数に保つことで clamp 構造が維持され、結果はそのまま
  Tor の expanded secret key (`hs_ed25519_secret_key`) として有効
- 発見した鍵は必ず CPU 側で再計算・照合してから出力（GPU/CPU で同一の
  `__host__ __device__` コードを共用）

## ビルド

### Linux (H100 などの本番環境)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

デフォルトで sm_60〜sm_90 の実バイナリ + compute_90 の PTX を同梱するため、
Pascal 以降のほぼ全 GPU（Blackwell 以降は JIT）で動作する。
特定 GPU のみで良ければ高速にビルドできる:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90   # H100 のみ
```

### Windows (開発環境)

VS 2022 + CUDA Toolkit 12.x:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## 使い方

```bash
./gpuonion <prefix> [options]
./gpuonion -b [options]
  <prefix>       探索する base32 プレフィックス (a-z 2-7)
  -b, --bench    約20秒のベンチマークを実行（prefix不要）
  -d <spec>      CUDA デバイス指定: "all"（デフォルト、全GPU使用）/ 単一番号 / "0,1,2" のようなカンマ区切り
                 指定したGPUはそれぞれ独立したホストスレッドで並行して動作する
  -n <count>     この個数見つけたら終了 (default 1)
  -o <dir>       出力ディレクトリ (default ./found)
  --blocks <n>   ブロック数 (default: SM数 * 16)
  -t <threads>   ブロックあたりスレッド数 (default 128)
  -i <iters>     1回の起動でスレッドあたり試行する候補数 (default 1024)
  -B <batch>     Montgomery バッチ逆元のバッチサイズ、2べき 8..1024 (default 128)
  --m51          5x51 limb 版カーネルを使用（A/B 比較用）
  --ext          Edwards 拡張座標版カーネルを使用（A/B 比較用、最も遅い）
  --selftest     内部テストのみ実行
```

高速化の経緯と各手法の効果は [Optimize.md](Optimize.md) を参照。

例:

```bash
./gpuonion claude
```

ベンチマーク（マッチしない16文字prefixで実測し、prefix長ごとの期待所要時間も表示）:

```bash
./gpuonion -b
```

出力は `found/<address>/` に Tor がそのまま読める形式で保存される:

- `hostname`
- `hs_ed25519_secret_key`（`HiddenServiceDir` に配置して使用）
- `hs_ed25519_public_key`

## 性能の目安

| GPU | 速度 |
|---|---|
| RTX 4070 (sm_89) | ~2.7 GKey/s (持続) |
| H100 (sm_90) | ~5-6 GKey/s (INT スループット比からの見積もり) |

期待試行回数は `32^len`。RTX 4070 の実測では 6文字 prefix が平均 ~0.4 秒、
7文字が ~13 秒、8文字が ~7 分、9文字が ~3.6 時間。

実装は Montgomery x-only 差分加算 (3M+2S/候補) + バッチ逆元 +
4×64bit limb の遅延簡約 (PTX madc 連鎖)。詳細は [Optimize.md](Optimize.md)。

## 注意

- prefix に使えるのは base32 文字 (`a-z`, `2-7`) のみ。`0 1 8 9` は不可
- 秘密鍵ファイルの取り扱いに注意（`found/` はコミットしないこと）
