/*
 * tests/unit/test_async_file_writer.cpp
 *
 * The background file writer runs jobs in submission order and wait() returns
 * only after all of them have finished. Failures are raised by waitOrThrow() with the file name, the memory the
 * queue holds is bounded, and AsyncMrcStack writes the same bytes as the per-image WRITE_APPEND it replaces.
 */

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <random>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#include "src/async_file_writer.h"
#include "src/async_mrc_stack.h"

TEST_CASE("AsyncFileWriter runs jobs in order and wait() drains the queue", "[async_write]")
{
    std::vector<int> order;
    std::atomic<int> done(0);
    for (int i = 0; i < 20; i++)
        AsyncFileWriter::instance().submit([&order, &done, i]
        {
            if (i == 0) std::this_thread::sleep_for(std::chrono::milliseconds(30));
            order.push_back(i);
            done++;
        });
    AsyncFileWriter::instance().wait();
    REQUIRE(done == 20);
    REQUIRE(order.size() == 20);
    for (int i = 0; i < 20; i++) REQUIRE(order[i] == i);
}

TEST_CASE("AsyncFileWriter survives a job that throws", "[async_write]")
{
    std::atomic<int> done(0);
    AsyncFileWriter::instance().submit([] { throw std::runtime_error("expected in test"); });
    AsyncFileWriter::instance().submit([&done] { done++; });
    AsyncFileWriter::instance().wait();
    REQUIRE(done == 1);
    // wait() does not raise; waitOrThrow() does, once (the message is the job's own)
    REQUIRE_THROWS_AS(AsyncFileWriter::instance().waitOrThrow(), RelionError);
    REQUIRE_NOTHROW(AsyncFileWriter::instance().waitOrThrow());
}

namespace
{
std::string scratchDir()
{
    char tmpl[] = "/tmp/async_writer_test_XXXXXX";
    const char *d = mkdtemp(tmpl);
    REQUIRE(d != nullptr);
    return d;
}

std::string slurp(const std::string &fn, bool mask_label)
{
    std::ifstream in(fn, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (mask_label && s.size() >= 304) std::fill(s.begin() + 224, s.begin() + 304, '\0'); // time stamp
    return s;
}
}

TEST_CASE("waitOrThrow raises the first failed write with its file name, once", "[async_write]")
{
    AsyncFileWriter &w = AsyncFileWriter::instance();
    w.waitOrThrow();
    const std::string dir = scratchDir();

    // a file in a directory that does not exist
    AsyncFileWriter::putFile(dir + "/missing/out.mrc", 10, [&](const std::string &tmp)
    {
        std::ofstream out(tmp);
        if (!out) REPORT_ERROR("cannot open " + tmp);
    });
    AsyncFileWriter::putFile(dir + "/fine.txt", 10, [](const std::string &tmp) { std::ofstream(tmp) << "ok"; });
    bool raised = false;
    try { w.waitOrThrow(); }
    catch (RelionError &e)
    {
        raised = true;
        REQUIRE(e.msg.find("missing/out.mrc") != std::string::npos);
    }
    REQUIRE(raised);
    REQUIRE(slurp(dir + "/fine.txt", false) == "ok");   // later writes still happen, under the final name
    REQUIRE_NOTHROW(w.waitOrThrow());                   // the error was reported once
}

TEST_CASE("a write into a read-only directory makes waitOrThrow raise, and leaves nothing", "[async_write]")
{
    if (geteuid() == 0) return; // root can write anywhere
    AsyncFileWriter &w = AsyncFileWriter::instance();
    w.waitOrThrow();
    const std::string dir = scratchDir();
    REQUIRE(chmod(dir.c_str(), 0555) == 0);

    Image<RFLOAT> img(8, 8);
    img().initConstant(1.);
    AsyncMrcStack stack(dir + "/s.mrcs", Float);
    stack.add(img);
    stack.add(img);
    stack.add(img);
    stack.finish();
    REQUIRE_THROWS_AS(w.waitOrThrow(), RelionError);
    struct stat sb;
    REQUIRE(stat((dir + "/s.mrcs").c_str(), &sb) != 0);
    chmod(dir.c_str(), 0755);
}

TEST_CASE("an abandoned stack leaves no file", "[async_write]")
{
    AsyncFileWriter::instance().waitOrThrow();
    const std::string dir = scratchDir();
    Image<RFLOAT> img(8, 8);
    img().initConstant(1.);
    {
        AsyncMrcStack stack(dir + "/s.mrcs", Float);
        stack.add(img);
        stack.add(img);
    }
    AsyncFileWriter::instance().waitOrThrow();
    struct stat sb;
    REQUIRE(stat((dir + "/s.mrcs").c_str(), &sb) != 0);
    REQUIRE(stat((dir + "/.tmp_s.mrcs").c_str(), &sb) != 0);
}

TEST_CASE("submit blocks while the queued bytes pass the budget, but takes a job into an empty queue", "[async_write]")
{
    AsyncFileWriter &w = AsyncFileWriter::instance();
    w.wait();
    AsyncFileWriter::setBudgetBytes(1000);

    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();
    std::atomic<bool> a_started(false), b_submitted(false), b_ran(false);
    w.submit([&] { a_started = true; released.wait(); }, 800);
    REQUIRE(w.queuedBytes() == 800);
    std::thread t([&] { w.submit([&] { b_ran = true; }, 800); b_submitted = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    REQUIRE_FALSE(b_submitted);                 // 800 + 800 > 1000: waits for the first to finish
    REQUIRE(w.queuedBytes() == 800);
    release.set_value();
    t.join();
    w.wait();
    REQUIRE(b_submitted);
    REQUIRE(b_ran);
    REQUIRE(w.queuedBytes() == 0);

    // a job bigger than the whole budget still goes in when nothing is queued
    std::atomic<bool> big_ran(false);
    w.submit([&] { big_ran = true; }, 5000);
    w.wait();
    REQUIRE(big_ran);
    REQUIRE(w.queuedBytes() == 0);
    AsyncFileWriter::setBudgetBytes(0);
}

TEST_CASE("AsyncMrcStack writes the bytes of the per-image overwrite and append", "[async_write]")
{
    AsyncFileWriter &w = AsyncFileWriter::instance();
    w.waitOrThrow();
    const std::string dir = scratchDir();
    std::mt19937 rng(5);
    std::normal_distribution<double> noise(0., 1.);

    for (DataType dt : {Float, Float16})
        for (int n : {1, 2, 3, 9})
            for (size_t chunk : {(size_t) 1, (size_t) 700, (size_t) 1 << 20})
            {
                std::vector<Image<RFLOAT> > imgs;
                RFLOAT mn = 1e30, mx = -1e30;
                for (int i = 0; i < n; i++)
                {
                    imgs.emplace_back(12, 10);
                    for (long int k = 0; k < MULTIDIM_SIZE(imgs[i]()); k++)
                        DIRECT_MULTIDIM_ELEM(imgs[i](), k) = noise(rng) * (i + 1);
                    mn = std::min(mn, imgs[i]().computeMin());
                    mx = std::max(mx, imgs[i]().computeMax());
                }
                // as Extract does: the statistics of the whole stack and the pixel size go in the last image's main header
                imgs.back().MDMainHeader.setValue(EMDL_IMAGE_STATS_MIN, mn);
                imgs.back().MDMainHeader.setValue(EMDL_IMAGE_STATS_MAX, mx);
                imgs.back().MDMainHeader.setValue(EMDL_IMAGE_STATS_AVG, (RFLOAT) 0.25);
                imgs.back().MDMainHeader.setValue(EMDL_IMAGE_STATS_STDDEV, (RFLOAT) 1.5);
                imgs.back().setSamplingRateInHeader(1.75);

                const std::string old_fn = dir + "/old.mrcs", new_fn = dir + "/new.mrcs";
                remove(old_fn.c_str());
                for (int i = 0; i < n; i++)
                    imgs[i].write(old_fn, -1, i == 0 && n > 1, i == 0 ? WRITE_OVERWRITE : WRITE_APPEND, dt);

                AsyncMrcStack stack(new_fn, dt, chunk);
                for (int i = 0; i < n; i++) stack.add(imgs[i]);
                stack.finish();
                w.waitOrThrow();

                INFO("datatype " << dt << " n " << n << " chunk " << chunk);
                REQUIRE(slurp(new_fn, true) == slurp(old_fn, true));
                REQUIRE(slurp(new_fn, false).size() == 1024 + (size_t) n * 120 * gettypesize(dt));
            }
}
