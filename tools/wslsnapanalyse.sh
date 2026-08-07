#!/bin/sh
# Break the last snapshot down and see what would make it smaller.
cd "$(dirname "$0")/.."
python3 tools/snapanalyse.py "$HOME/vvsession.snap"
