/*
 * tests/unit/test_async_file_writer.cpp
 *
 * The background file writer runs jobs in submission order and wait() returns
 * only after all of them have finished.
 */

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "src/async_file_writer.h"

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
}
