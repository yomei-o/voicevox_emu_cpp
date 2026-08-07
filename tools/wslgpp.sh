#!/bin/sh
# What ~/gpp/bin/g++ actually is.
ls -l "$HOME/gpp/bin/"
echo "== file"
file "$HOME/gpp/bin/g++" 2>/dev/null
echo "== if it is a script, its contents"
head -20 "$HOME/gpp/bin/g++" 2>/dev/null | cat -v | head -20
