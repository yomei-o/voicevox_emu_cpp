// x86emu with the CUDA shim behind it.
//
// The same emulator, plus one thing: `Emulator::on_host_call` is wired to
// src/cudahost.cpp, so a guest that issues the reserved syscall gets its
// arithmetic done natively instead of interpreted.  Without that wiring the
// syscall answers ENOSYS and the guest's stand-ins fall back to computing
// nothing, which is a legible failure rather than a wrong answer.
//
//   vvcudaemu --sysroot DIR <program> [guest args...]
//
// The option set is deliberately small.  x86emu's own main.cpp is the place for
// tracing and dumps; this exists to answer one question - what does the
// arithmetic cost when it is not interpreted - and every option that is not
// about that is a distraction from it.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "emulator.h"

int64_t vv_host_call(x86emu::Emulator& e, uint64_t id, uint64_t args);
// The shim's own saved bookkeeping - the device arena's free list and the cuDNN
// descriptor table.  Implemented in cudahost.cpp, where those live.
bool vv_restore_shim(x86emu::Emulator& e, const std::string& path);
extern "C" int vvstub_timing;

int main(int argc, char** argv) {
    x86emu::Emulator::Options opt;
    std::string program;
    std::vector<std::string> guest_args;
    // A Linux guest is given only PATH by default - deliberately, so that a
    // Windows host's variables do not reach it - so anything the guest is meant
    // to read has to be said here.  That default cost this project two wrong
    // explanations before it was noticed.
    std::vector<std::string> guest_env;
    // A state to carry on from instead of starting the program.  The program is
    // still loaded first: that is what rebuilds the hook table, the module list
    // and the shim's own boundary, none of which is in the file.
    std::string resume;

    int i = 1;
    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--sysroot" || arg == "-r") && i + 1 < argc) {
            x86emu::FileTable::set_sysroot(argv[++i]);
        } else if (arg == "--trace-calls" || arg == "-c") {
            opt.trace_calls = true;
        } else if (arg == "--map" || arg == "-m") {
            opt.dump_map = true;
        } else if (arg == "--env" && i + 1 < argc) {
            guest_env.push_back(argv[++i]);
        } else if (arg == "--resume" && i + 1 < argc) {
            resume = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::printf("vvcudaemu [--sysroot DIR] [--env NAME=VALUE] [-c] [-m]"
                        " [--resume FILE] <program> [args...]\n");
            return 0;
        } else if (!arg.empty() && arg[0] == '-' && arg != "-") {
            std::fprintf(stderr, "vvcudaemu: unknown option %s\n", arg.c_str());
            return 2;
        } else {
            program = arg;
            ++i;
            break;
        }
    }
    if (program.empty()) {
        std::fprintf(stderr, "vvcudaemu: no program\n");
        return 2;
    }
    guest_args.push_back(program);
    for (; i < argc; ++i) guest_args.push_back(argv[i]);

    x86emu::Emulator emu(opt);
    emu.on_host_call = vv_host_call;
    if (!guest_env.empty()) {
        std::vector<std::pair<std::string, std::string>> env;
        env.emplace_back("PATH", "/usr/local/bin:/usr/bin:/bin");
        for (const std::string& e : guest_env) {
            size_t eq = e.find('=');
            if (eq == std::string::npos || eq == 0) {
                std::fprintf(stderr, "vvcudaemu: --env wants NAME=VALUE, got %s\n",
                             e.c_str());
                return 2;
            }
            env.emplace_back(e.substr(0, eq), e.substr(eq + 1));
        }
        emu.set_environment(std::move(env));
    }
    if (const char* t = std::getenv("VVSTUB_TIME")) vvstub_timing = *t && *t != '0';

    try {
        emu.load(program, guest_args);
    } catch (const std::exception& err) {
        std::fflush(stdout);
        std::fprintf(stderr, "vvcudaemu: cannot load %s: %s\n", program.c_str(),
                     err.what());
        return 1;
    }

    // The guest's state goes back first.  The device arena is part of it - a
    // region like any other, marked contiguous - so restoring the address space
    // is what brings the arena back, and only then can the shim find out where
    // on this host it landed.
    if (!resume.empty()) {
        if (!emu.load_state(resume)) {
            std::fprintf(stderr, "vvcudaemu: cannot resume from %s\n", resume.c_str());
            return 1;
        }
        if (!vv_restore_shim(emu, resume + ".shim")) {
            std::fprintf(stderr, "vvcudaemu: cannot resume from %s.shim\n",
                         resume.c_str());
            return 1;
        }
    }

    try {
        return emu.run();
    } catch (const x86emu::CpuError& err) {
        std::fflush(stdout);
        std::fprintf(stderr, "\nvvcudaemu: %s\n", err.what());
        std::fprintf(stderr, "  %s\n", emu.cpu().state_line().c_str());
        std::fprintf(stderr, "  after %llu instructions\n",
                     (unsigned long long)emu.cpu().instructions_executed);
        return 1;
    } catch (const x86emu::MemoryFault& err) {
        std::fflush(stdout);
        std::fprintf(stderr, "\nvvcudaemu: %s\n", err.what());
        std::string what = emu.describe_address(err.addr);
        if (!what.empty()) std::fprintf(stderr, "  %s\n", what.c_str());
        std::fprintf(stderr, "  %s\n", emu.cpu().state_line().c_str());
        return 1;
    }
}
