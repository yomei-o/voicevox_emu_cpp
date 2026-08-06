// analyze.c - text in, accent phrases out, without any inference.
//
// The expensive half of VOICEVOX is the neural network: decrypting the model
// and initialising its ONNX sessions takes minutes under the emulator, and
// generating audio takes longer still.  Text analysis is the other half, and it
// is cheap - Open JTalk reads its dictionary and works out the moras, the accent
// positions and the phonemes with no model involved at all.
//
// So this is the part of the pipeline that a browser can run interactively: the
// real Open JTalk, inside the emulator, answering in seconds.  It is the same
// libvoicevox_core.so and the same published function the full pipeline calls.
//
//     analyze <dict-dir> <text>
//
// Prints the accent phrase JSON on stdout and nothing else, so a caller can
// parse it directly.
#include <stdio.h>
#include <stdlib.h>

#include "voicevox_core.h"

int main(int argc, char** argv) {
    const char* dict = argc > 1 ? argv[1] : "/opt/vv/open_jtalk_dic_utf_8-1.11";
    const char* text = argc > 2 ? argv[2] : "ずんだもんなのだ";

    OpenJtalkRc* ojt = NULL;
    VoicevoxResultCode r = voicevox_open_jtalk_rc_new(dict, &ojt);
    if (r != VOICEVOX_RESULT_OK) {
        fprintf(stderr, "open_jtalk_rc_new: %d %s\n", (int)r,
                voicevox_error_result_to_message(r));
        return 1;
    }

    char* json = NULL;
    r = voicevox_open_jtalk_rc_analyze(ojt, text, &json);
    if (r != VOICEVOX_RESULT_OK) {
        fprintf(stderr, "open_jtalk_rc_analyze: %d %s\n", (int)r,
                voicevox_error_result_to_message(r));
        return 1;
    }

    fputs(json, stdout);
    fputc('\n', stdout);
    fflush(stdout);

    voicevox_json_free(json);
    voicevox_open_jtalk_rc_delete(ojt);
    return 0;
}
