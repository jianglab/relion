#ifndef ASYNC_FILE_WRITER_H_
#define ASYNC_FILE_WRITER_H_

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
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
