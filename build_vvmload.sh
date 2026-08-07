#!/bin/sh
# Builds src/vvmload.cpp - the .vvm loading sample - on whichever host this is.
#
# On Windows this calls cl.exe directly and passes the include and library
# directories as flags.  vcvars is not used: sourcing it hangs on this machine,
# and the flags are the only thing that has worked reliably here.  Override the
# three paths if your Visual Studio is elsewhere:
#
#   VC=... SDK=... SDKVER=... sh build_vvmload.sh
set -e
cd "$(dirname "$0")"
mkdir -p build

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
        VC=${VC:-C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.31.31103}
        SDK=${SDK:-C:/Program Files (x86)/Windows Kits/10}
        SDKVER=${SDKVER:-10.0.19041.0}
        CL="$VC/bin/Hostx64/x64/cl.exe"
        [ -x "$CL" ] || { echo "no cl.exe at $CL - set VC="; exit 1; }
        export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1 VSLANG=1033
        # /D_CRT_SECURE_NO_WARNINGS because fopen is fopen; /utf-8 because the
        # comments are not all ASCII and MSVC otherwise reads them as the ANSI
        # code page.
        "$CL" /nologo /std:c++17 /O2 /W3 /EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS \
            "/I$VC/include" "/I$SDK/Include/$SDKVER/ucrt" \
            "/I$SDK/Include/$SDKVER/um" "/I$SDK/Include/$SDKVER/shared" \
            /Isrc src/vvmload.cpp \
            /Fo:build/ /Fe:build/vvmload.exe \
            /link "/LIBPATH:$VC/lib/x64" "/LIBPATH:$SDK/Lib/$SDKVER/ucrt/x64" \
            "/LIBPATH:$SDK/Lib/$SDKVER/um/x64"
        echo "   build/vvmload.exe"
        ;;
    *)
        CXX=${CXX:-g++}
        $CXX -std=c++17 -O2 -Wall -Wextra -Isrc -o build/vvmload src/vvmload.cpp -ldl
        echo "   build/vvmload"
        ;;
esac
