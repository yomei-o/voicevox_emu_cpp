#!/bin/sh
# Build vvcudaemu under WSL with the locally fetched g++.
#
# A thin wrapper so the invocation carries no shell metacharacters: the path
# from a Windows shell through wsl.exe eats variable assignments and
# redirections, and a script file does not go through it.  The log goes to
# $HOME rather than /tmp because WSL shuts the VM down between calls and takes
# /tmp with it.
cd "$(dirname "$0")/.."
LOG=$HOME/vvbuild.log
CXX=$HOME/gpp/bin/g++ CC=gcc sh build_cudaemu.sh > "$LOG" 2>&1
status=$?
echo "== first 40 lines"
head -40 "$LOG" | cut -c1-300
echo "== last 10 lines"
tail -10 "$LOG" | cut -c1-300
exit $status
