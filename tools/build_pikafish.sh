#!/bin/sh
# Build Pikafish cho Jetson Nano (aarch64) TRÊN MÁY CÓ MẠNG (Colab / laptop Ubuntu), rồi chép sang robot.
# Robot không cần mạng, không cần trình biên dịch: file chạy được link tĩnh (-static).
#
#   sh tools/build_pikafish.sh            -> dist/pikafish_jetson.tar.gz (pikafish + pikafish.nnue)
#   sh tools/build_pikafish.sh x86        -> bản cho chính máy đang chạy (Colab / laptop Linux)
#   sh tools/build_pikafish.sh windows    -> dist/pikafish_windows.zip (pikafish.exe + pikafish.nnue) cho laptop Windows
#
# Trên robot:  tar xzf pikafish_jetson.tar.gz -C ~/   ->  ~/pikafish/pikafish, ~/pikafish/pikafish.nnue
set -e
TARGET=${1:-jetson}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/pikafish-build}
OUT="$ROOT/dist/pikafish_$TARGET"
mkdir -p "$WORK" "$OUT"

if [ ! -d "$WORK/Pikafish" ]; then
  git clone --depth 1 https://github.com/official-pikafish/Pikafish "$WORK/Pikafish"
fi
cd "$WORK/Pikafish/src"
make net                                   # tải pikafish.nnue (cần mạng) — phải cùng thời điểm với mã nguồn

if [ "$TARGET" = "jetson" ]; then
  command -v aarch64-linux-gnu-g++ >/dev/null 2>&1 || \
    (sudo apt-get update -q && sudo apt-get install -y -q g++-aarch64-linux-gnu) || \
    (apt-get update -q && apt-get install -y -q g++-aarch64-linux-gnu)
  make clean >/dev/null
  # armv8: Cortex-A57 của Jetson Nano có NEON nhưng không có dotprod
  # NNUE_EMBEDDING_OFF: không nhúng mạng vào file chạy -> đọc pikafish.nnue lúc chạy (file chạy nhỏ, đổi mạng dễ)
  make -j"$(nproc)" build ARCH=armv8 COMP=gcc CXX=aarch64-linux-gnu-g++ \
       EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF EXTRALDFLAGS=-static
elif [ "$TARGET" = "windows" ]; then
  # Laptop Windows (demo): biên dịch chéo bằng MinGW, link tĩnh -> 1 file .exe chạy ngay, không cần cài gì.
  # sse41-popcnt: chạy được trên gần như mọi CPU x86-64 từ 2010 trở đi.
  command -v x86_64-w64-mingw32-g++-posix >/dev/null 2>&1 || \
    (sudo apt-get update -q && sudo apt-get install -y -q mingw-w64) || \
    (apt-get update -q && apt-get install -y -q mingw-w64)
  make clean >/dev/null
  make -j"$(nproc)" build ARCH=x86-64-sse41-popcnt COMP=mingw CXX=x86_64-w64-mingw32-g++-posix \
       EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF EXTRALDFLAGS=-static
else
  make clean >/dev/null
  make -j"$(nproc)" build ARCH=x86-64-avx2 EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF
fi

if [ "$TARGET" = "windows" ]; then
  cp pikafish.exe pikafish.nnue "$OUT/"
  git log -1 --format='Pikafish %H (%cd)' > "$OUT/VERSION.txt"
  cd "$ROOT/dist" && rm -f "pikafish_windows.zip" && \
    python3 -c "import shutil; shutil.make_archive('pikafish_windows', 'zip', '.', 'pikafish_windows')"
  echo "Xong: $ROOT/dist/pikafish_windows.zip"
  exit 0
fi

cp pikafish pikafish.nnue "$OUT/"
git log -1 --format='Pikafish %H (%cd)' > "$OUT/VERSION.txt"
cd "$ROOT/dist" && tar czf "pikafish_$TARGET.tar.gz" -C "$OUT/.." "pikafish_$TARGET"
file "$OUT/pikafish" 2>/dev/null || true
echo "Xong: $ROOT/dist/pikafish_$TARGET.tar.gz"
