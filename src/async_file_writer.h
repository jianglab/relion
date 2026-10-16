#ifndef ASYNC_FILE_WRITER_H_
#define ASYNC_FILE_WRITER_H_

#include <condition_variable>
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

// Runs file writes on one background thread, in submission order, so the
// next iteration does not wait for a slow (network) file system.
// RELION_ASYNC_WRITE=0 makes submit() run the job at once.
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

    void submit(std::function<void()> job)
    {
        if (!enabled()) { job(); return; }
        std::unique_lock<std::mutex> lk(mtx_);
        if (!thread_.joinable()) thread_ = std::thread([this] { loop(); });
        queue_.push_back(std::move(job));
        cv_.notify_all();
    }

    // Returns when every submitted job has finished.
    void wait()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        idle_cv_.wait(lk, [this] { return queue_.empty() && !busy_; });
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
            std::function<void()> job = std::move(queue_.front());
            queue_.pop_front();
            busy_ = true;
            lk.unlock();
            try { job(); }
            catch (const std::exception &e) { std::cerr << "ERROR in background file write: " << e.what() << std::endl; }
            catch (...) { std::cerr << "ERROR in background file write" << std::endl; }
            lk.lock();
            busy_ = false;
            idle_cv_.notify_all();
        }
    }

    std::mutex mtx_;
    std::condition_variable cv_, idle_cv_;
    std::deque<std::function<void()> > queue_;
    std::thread thread_;
    bool stop_ = false, busy_ = false;
};

#endif
