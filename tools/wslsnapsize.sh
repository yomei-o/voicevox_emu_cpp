#!/bin/sh
# What the saved session weighs, and what compressing it would save.
#
# The question was whether it fits under 100 MB.  Answering it with a number
# beats answering it with an opinion, and the number depends on what the bytes
# are: model weights are floats, and floats compress badly.
set -e
SNAP=${VVSNAP:-$HOME/vv/session.state}
[ -f "$SNAP" ] || { echo "no $SNAP - run tools/wslresume.sh first"; exit 1; }
raw=$(wc -c < "$SNAP")
printf 'raw        %10.1f MB\n' "$(echo "$raw" | awk '{print $1/1048576}')"
for level in 1 6 9; do
    n=$(gzip -c -"$level" "$SNAP" | wc -c)
    printf 'gzip -%s    %10.1f MB   (%.0f%%)\n' "$level" \
        "$(echo "$n" | awk '{print $1/1048576}')" \
        "$(echo "$n $raw" | awk '{print $1*100/$2}')"
done
if command -v zstd > /dev/null 2>&1; then
    for level in 3 19; do
        n=$(zstd -c -"$level" -T0 "$SNAP" 2>/dev/null | wc -c)
        printf 'zstd -%-2s   %10.1f MB   (%.0f%%)\n' "$level" \
            "$(echo "$n" | awk '{print $1/1048576}')" \
            "$(echo "$n $raw" | awk '{print $1*100/$2}')"
    done
else
    echo "zstd: not installed"
fi
if command -v xz > /dev/null 2>&1; then
    n=$(xz -c -3 -T0 "$SNAP" | wc -c)
    printf 'xz -3      %10.1f MB   (%.0f%%)\n' \
        "$(echo "$n" | awk '{print $1/1048576}')" \
        "$(echo "$n $raw" | awk '{print $1*100/$2}')"
fi
