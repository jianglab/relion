#ifndef ASYNC_FILE_WRITER_H_
#define ASYNC_FILE_WRITER_H_

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <sys/stat.h>

#include "src/error.h"

namespace asyncwrite
{

/// "Name:   1234 kB" of a /proc file, in bytes, or -1
inline long long procKb(const char *file, const char *name)
{
    std::ifstream in(file);
    std::string line;
    const size_t n = strlen(name);
    while (std::getline(in, line))
        if (line.compare(0, n, name) == 0) return atoll(line.c_str() + n) * 1024LL;
    return -1;
}

/// Memory this process could still take: MemAvailable, or less where the cgroup (v2) limit or the Slurm job's memory
/// (SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE, less what the process holds) is nearer. Some clusters
/// set no cgroup limit, so only the Slurm variables say what the job asked for. Returns -1 when unknown.
inline long long memoryAvailableBytes()
{
    long long avail = procKb("/proc/meminfo", "MemAvailable:");
    // cgroup v2: smallest memory.max - memory.current from this process's group up to the root
    {
        std::ifstream in("/proc/self/cgroup");
        std::string line, path;
        while (std::getline(in, line)) if (line.compare(0, 3, "0::") == 0) path = line.substr(3);
        if (path.empty()) path = "/";
        for (;;)
        {
            const std::string dir = "/sys/fs/cgroup" + (path == "/" ? std::string("") : path);
            std::ifstream max_f((dir + "/memory.max").c_str()), cur_f((dir + "/memory.current").c_str());
            std::string max_s;
            long long cur = 0;
            if (max_f && cur_f && (max_f >> max_s) && (cur_f >> cur) && max_s != "max")
            {
                const long long left = std::max(0LL, atoll(max_s.c_str()) - cur);
                if (avail < 0 || left < avail) avail = left;
            }
            if (path == "/") break;
            const size_t slash = path.find_last_of('/');
            path = (slash == 0 || slash == std::string::npos) ? "/" : path.substr(0, slash);
        }
    }
    long long mb = -1;
    const char *node = getenv("SLURM_MEM_PER_NODE"), *cpu = getenv("SLURM_MEM_PER_CPU"), *cpus = getenv("SLURM_CPUS_ON_NODE");
    if (node && atoll(node) > 0) mb = atoll(node);
    else if (cpu && cpus && atoll(cpu) > 0 && atoll(cpus) > 0) mb = atoll(cpu) * atoll(cpus);
    if (mb > 0)
    {
        const long long left = std::max(0LL, mb * 1048576LL - std::max(0LL, procKb("/proc/self/status", "VmRSS:")));
        if (avail < 0 || left < avail) avail = left;
    }
    return avail;
}

}

// Runs file writes on one background thread, in submission order, so the
// next iteration does not wait for a slow (network) file system.
// RELION_ASYNC_WRITE=0 makes submit() run the job at once.
// submit(job, bytes) also bounds the memory the queue holds: it blocks while the queued bytes would pass
// budgetBytes() (min(2 GB, 10% of the memory available), or RELION_ASYNC_WRITE_MB), but always takes a job when
// the queue is empty. A job that fails is not lost: the first failure is remembered and waitOrThrow() raises it
// (wait() still only waits, as relion_refine expects).
class AsyncFileWriter
{
public:
    static AsyncFileWriter& instance()
    {
        static AsyncFileWriter w;
        return w;
    }

    static bool enabled()
    {
        static const bool on = []{ const char *e = getenv("RELION_ASYNC_WRITE"); return !(e && (!strcmp(e, "0") || !strcmp(e, "off"))); }();
        return on;
    }

    // RELION_ASYNC_STAR_WRITE: 1 (default) the text is formatted at once (from a snapshot where the tables change
    // later) and the file is opened, written and closed by the background thread, so the next step does not
    // wait for the file system; 0 or "off": the STAR files of an iteration are formatted and written by the
    // calling thread as before; "check" is 1, and the background thread also reads each file back and
    // compares it with the text (data.star: with a text formatted by the calling thread too), and reports
    // any difference. The files are byte for byte the same in all modes.
    static int starMode()
    {
        static const int m = []{ const char *e = getenv("RELION_ASYNC_STAR_WRITE");
            if (e && (!strcmp(e, "0") || !strcmp(e, "off"))) return 0;
            return e && !strcmp(e, "check") ? 2 : 1; }();
        return m;
    }

    // Puts the text that "produce" returns in file "fn" (both done by the background thread when allowed), and reads it back in check mode.
    static void putText(const std::string &fn, std::function<std::shared_ptr<std::string>()> produce, bool allow_async, const std::shared_ptr<std::string> &expect_same = nullptr)
    {
        auto put = [fn, produce, expect_same]
        {
            std::shared_ptr<std::string> text = produce();
            { std::ofstream out(fn); if (!out) throw std::runtime_error("cannot write " + fn); out.write(text->data(), text->size()); }
            if (starMode() == 2)
            {
                std::ifstream in(fn, std::ios::binary);
                std::string back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                const bool same_file = (back == *text);
                const bool same_sync = !expect_same || (*expect_same == *text);
                std::cerr << "ASYNC_STAR_WRITE check " << fn << (same_file && same_sync ? ": identical" : ": DIFFERENT") << std::endl;
                if (!same_file || !same_sync) std::abort();
            }
        };
        if (allow_async) instance().submit(put); else put();
    }

    // Bytes the queue may hold: RELION_ASYNC_WRITE_MB, or min(2 GB, 10% of the memory available) when that is known
    // and 256 MB when not. Chosen at the first use and reported once.
    static size_t budgetBytes()
    {
        if (budget_override() != 0) return budget_override();
        static const size_t b = []
        {
            size_t bytes;
            std::string why;
            const char *e = getenv("RELION_ASYNC_WRITE_MB");
            const long long avail = asyncwrite::memoryAvailableBytes();
            if (e && atoll(e) > 0) { bytes = (size_t) atoll(e) << 20; why = "RELION_ASYNC_WRITE_MB"; }
            else if (avail >= 0) { bytes = (size_t) std::min(2048LL << 20, avail / 10); why = "10% of the " + std::to_string(avail >> 20) + " MB available, at most 2048 MB"; }
            else { bytes = (size_t) 256 << 20; why = "memory availability unknown"; }
            std::cerr << " Background writes of image files may queue up to " << (bytes >> 20) << " MB (" << why << ")" << std::endl;
            return bytes;
        }();
        return b;
    }

    // For tests: a budget of "bytes" instead (0 returns to the chosen one)
    static void setBudgetBytes(size_t bytes) { budget_override() = bytes; }

    void submit(std::function<void()> job, size_t bytes = 0)
    {
        if (!enabled()) { job(); return; }
        const size_t budget = bytes > 0 ? budgetBytes() : 0;
        std::unique_lock<std::mutex> lk(mtx_);
        if (bytes > 0)
            space_cv_.wait(lk, [&] { return queued_bytes_ == 0 || queued_bytes_ + bytes <= budget; });
        if (!thread_.joinable()) thread_ = std::thread([this] { loop(); });
        queue_.push_back(std::make_pair(std::move(job), bytes));
        queued_bytes_ += bytes;
        cv_.notify_all();
    }

    // Returns when every submitted job has finished.
    void wait()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        idle_cv_.wait(lk, [this] { return queue_.empty() && !busy_; });
    }

    // wait(), then raises the first failure of a job since the last call (as REPORT_ERROR, in this thread).
    void waitOrThrow()
    {
        wait();
        std::string err;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            err.swap(first_error_);
        }
        if (!err.empty()) REPORT_ERROR(err);
    }

    // Bytes in the queue now (running job included)
    size_t queuedBytes()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        return queued_bytes_;
    }

    // "dir/name.ext" -> "dir/.tmp_name.ext": the file is written there and renamed, so an aborted job leaves no
    // partial file under the final name (the extension stays, as it selects the format)
    static std::string tmpName(const std::string &fn)
    {
        const size_t slash = fn.find_last_of('/');
        return slash == std::string::npos ? ".tmp_" + fn : fn.substr(0, slash + 1) + ".tmp_" + fn.substr(slash + 1);
    }

    // Has "write" put the whole content in the temporary name, then renames it to "fn". Any failure removes the
    // temporary file and is reported with the file name (waitOrThrow() of the queue, or here when synchronous).
    static void putFile(const std::string &fn, size_t bytes, std::function<void(const std::string &tmp)> write)
    {
        instance().submit([fn, write]
        {
            const std::string tmp = tmpName(fn);
            try
            {
                write(tmp);
                if (rename(tmp.c_str(), fn.c_str()) != 0) REPORT_ERROR("cannot rename " + tmp + " to " + fn);
            }
            catch (const RelionError &e) { remove(tmp.c_str()); REPORT_ERROR("cannot write " + fn + ": " + stripError(e.msg)); }
            catch (const std::exception &e) { remove(tmp.c_str()); REPORT_ERROR("cannot write " + fn + ": " + e.what()); }
        }, bytes);
    }

    ~AsyncFileWriter()
    {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
            cv_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
    }

private:
    AsyncFileWriter() {}

    void loop()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        for (;;)
        {
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;
            std::function<void()> job = std::move(queue_.front().first);
            const size_t bytes = queue_.front().second;
            queue_.pop_front();
            busy_ = true;
            lk.unlock();
            std::string err;
            try { job(); }
            catch (const std::exception &e) { err = e.what(); }
            catch (const RelionError &e) { err = stripError(e.msg); }
            catch (...) { err = "unknown error"; }
            if (!err.empty()) std::cerr << "ERROR in background file write: " << err << std::endl;
            job = nullptr; // frees what the job held
            lk.lock();
            if (!err.empty() && first_error_.empty()) first_error_ = err;
            busy_ = false;
            queued_bytes_ -= bytes;
            space_cv_.notify_all();
            idle_cv_.notify_all();
        }
    }

    // RelionError messages start with "ERROR: \n"; REPORT_ERROR adds it again when the error is raised in the calling thread
    static std::string stripError(const std::string &msg)
    {
        return msg.compare(0, 8, "ERROR: \n") == 0 ? msg.substr(8) : msg;
    }

    static size_t &budget_override() { static size_t b = 0; return b; }

    std::mutex mtx_;
    std::condition_variable cv_, idle_cv_, space_cv_;
    std::deque<std::pair<std::function<void()>, size_t> > queue_;
    size_t queued_bytes_ = 0;
    std::string first_error_;
    std::thread thread_;
    bool stop_ = false, busy_ = false;
};

#endif
