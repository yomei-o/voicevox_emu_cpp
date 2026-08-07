#!/bin/sh
# Does this g++ accept an object file as input at all?
#
# The build stopped with the compiler *parsing* a .o as C++ source, which is
# not something a working driver does - so the question is whether the driver
# is working, not whether the command was right.
cd /tmp || exit 1
printf 'int probe_value(void) { return 42; }\n' > probe_a.c
printf 'extern "C" int probe_value(void);\nint main(void) { return probe_value() == 42 ? 0 : 1; }\n' > probe_b.cpp

for cxx in "$HOME/gpp/bin/g++" g++ c++; do
    command -v "$cxx" >/dev/null 2>&1 || { echo "$cxx: not present"; continue; }
    echo "== $cxx"
    "$cxx" --version | head -1
    gcc -c probe_a.c -o probe_a.o 2>&1 | head -3
    "$cxx" probe_b.cpp probe_a.o -o probe_out 2>&1 | head -5
    if [ -x probe_out ]; then
        ./probe_out && echo "  links objects: yes"
    else
        echo "  links objects: NO"
    fi
done
