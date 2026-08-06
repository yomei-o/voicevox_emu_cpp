#!/bin/sh
# Expands the payload that is committed compressed.  Run by setup.sh and by the
# run scripts; doing it twice is free.
#
# Only the Open JTalk dictionary needs this.  It is 107 MB expanded, of which
# sys.dic alone is 98 MB - close enough to GitHub's 100 MB per-file limit to be
# uncomfortable, and it is distributed upstream as a 23 MB tar.gz anyway, so
# that is what the repository carries: the pristine download, unmodified.
set -e
cd "$(dirname "$0")"

if [ ! -f guest/open_jtalk_dic_utf_8-1.11/sys.dic ]; then
    echo "unpacking the Open JTalk dictionary (107 MB)"
    tar xzf guest/open_jtalk_dic_utf_8-1.11.tar.gz -C guest
fi
# Eight files, sys.dic among them; a short count means a truncated archive.
n=$(ls guest/open_jtalk_dic_utf_8-1.11 | wc -l)
[ "$n" -ge 8 ] || { echo "dictionary incomplete ($n files)"; exit 1; }
