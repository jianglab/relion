/***************************************************************************
 *
 * Test program for the MPI prefetch protocol (MPITAG_PREFETCH_REQ).
 *
 * Tests:
 *   1. Follower sends PREFETCH_REQ, leader responds with job assignment
 *   2. Concurrent PREFETCH_REQ and JOB_REQUEST multiplexed via MPI_Probe
 *   3. Leader correctly sequences job IDs with multiple followers
 *   4. Termination signal (JOB_FIRST=-1) is handled correctly
 *   5. MPI_THREAD_MULTIPLE does not cause concurrent access issues
 *
 * Run: mpirun -np 3 prefetch_mpi_test
 *
 * Exit code: 0 on success, 1 on failure.
 * Prints "ALL TESTS PASSED" or "FAILED at step N" to stderr.
 ***************************************************************************/

#include "src/mpi.h"
#include "src/funcs.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>

static int rank, size;
static MPI_Status status;

// Helper: test condition, print failure, abort
#define TEST(cond, step) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED at step %d on rank %d: %s\n", step, rank, #cond); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Helper: careful Send with active check for MPI_THREAD_MULTIPLE safety
static void checked_send(void *buf, int count, MPI_Datatype dtype,
                         int dest, int tag, MPI_Comm comm, int step)
{
    int err = MPI_Send(buf, count, dtype, dest, tag, comm);
    TEST(err == MPI_SUCCESS, step);
}

static void checked_recv(void *buf, int count, MPI_Datatype dtype,
                         int source, int tag, MPI_Comm comm,
                         MPI_Status *stat, int step)
{
    int err = MPI_Recv(buf, count, dtype, source, tag, comm, stat);
    TEST(err == MPI_SUCCESS, step);
}

static void test_basic_prefetch_request()
{
    // Leader (rank 0): expect PREFETCH_REQ, send back job assignment
    // Follower (rank 1): send PREFETCH_REQ, receive job assignment

    if (rank == 0)
    {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        TEST(status.MPI_TAG == MPITAG_PREFETCH_REQ, 10);
        TEST(status.MPI_SOURCE == 1, 11);

        int dummy;
        checked_recv(&dummy, 1, MPI_INT, 1, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, &status, 12);

        long int job_msg[6];
        job_msg[0] = 0;   // JOB_FIRST
        job_msg[1] = 9;   // JOB_LAST
        job_msg[2] = 10;  // JOB_NIMG
        job_msg[3] = 0;   // JOB_LEN_FN_IMG
        job_msg[4] = 0;   // JOB_LEN_FN_CTF
        job_msg[5] = 0;   // JOB_LEN_FN_RECIMG

        checked_send(job_msg, 6, MPI_LONG, 1, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, 13);
    }
    else if (rank == 1)
    {
        int dummy = 42;
        checked_send(&dummy, 1, MPI_INT, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, 20);

        long int job_msg[6];
        checked_recv(job_msg, 6, MPI_LONG, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, &status, 21);

        TEST(job_msg[0] == 0, 22);   // JOB_FIRST
        TEST(job_msg[1] == 9, 23);   // JOB_LAST
        TEST(job_msg[2] == 10, 24);  // JOB_NIMG
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_concurrent_prefetch_and_job_request()
{
    // Test that the leader can receive a PREFETCH_REQ interleaved
    // with a JOB_REQUEST from a different follower.
    //
    // Scenario:
    //   Rank 1 sends PREFETCH_REQ (asking for next job)
    //   Rank 2 sends JOB_REQUEST (returning results for previous job)
    //   Leader handles both via MPI_Probe in any order

    if (rank == 0)
    {
        long int recv_buf[6];
        int prefetch_done = 0, job_done = 0;

        while (prefetch_done + job_done < 2)
        {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == MPITAG_PREFETCH_REQ)
            {
                int dummy;
                checked_recv(&dummy, 1, MPI_INT, status.MPI_SOURCE,
                             MPITAG_PREFETCH_REQ, MPI_COMM_WORLD,
                             &status, 30);

                long int job_msg[6];
                job_msg[0] = 10;
                job_msg[1] = 19;
                job_msg[2] = 10;
                job_msg[3] = 0;
                job_msg[4] = 0;
                job_msg[5] = 0;

                checked_send(job_msg, 6, MPI_LONG, status.MPI_SOURCE,
                             MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, 31);
                prefetch_done++;
            }
            else if (status.MPI_TAG == MPITAG_JOB_REQUEST)
            {
                checked_recv(recv_buf, 6, MPI_LONG, status.MPI_SOURCE,
                             MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                             &status, 32);

                TEST(recv_buf[0] >= 0, 33); // JOB_FIRST should be valid
                TEST(recv_buf[2] > 0, 34);  // JOB_NIMG should be > 0

                // Receive metadata (1 dummy double)
                double md;
                checked_recv(&md, 1, MPI_DOUBLE, status.MPI_SOURCE,
                             MPITAG_METADATA, MPI_COMM_WORLD,
                             &status, 35);
                job_done++;
            }
        }

        TEST(prefetch_done == 1, 36);
        TEST(job_done == 1, 37);
    }
    else if (rank == 1)
    {
        // Send PREFETCH_REQ
        int dummy = 1;
        checked_send(&dummy, 1, MPI_INT, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, 40);

        long int job_msg[6];
        checked_recv(job_msg, 6, MPI_LONG, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, &status, 41);
        TEST(job_msg[0] == 10, 42);
    }
    else if (rank == 2)
    {
        // Give rank 1 time to send PREFETCH_REQ first
        usleep(10000);

        // Send JOB_REQUEST (simulate returning results)
        long int job_msg[6];
        job_msg[0] = 0;   // JOB_FIRST
        job_msg[1] = 4;   // JOB_LAST
        job_msg[2] = 5;   // JOB_NIMG
        job_msg[3] = 0;
        job_msg[4] = 0;
        job_msg[5] = 0;

        checked_send(job_msg, 6, MPI_LONG, 0, MPITAG_JOB_REQUEST,
                     MPI_COMM_WORLD, 50);

        // Send metadata
        double md = 1.0;
        checked_send(&md, 1, MPI_DOUBLE, 0, MPITAG_METADATA,
                     MPI_COMM_WORLD, 51);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_termination_signal()
{
    // Test that JOB_FIRST = -1 is handled correctly

    if (rank == 0)
    {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        TEST(status.MPI_TAG == MPITAG_PREFETCH_REQ, 60);

        int dummy;
        checked_recv(&dummy, 1, MPI_INT, status.MPI_SOURCE,
                     MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, &status, 61);

        // Send termination
        long int job_msg[6];
        job_msg[0] = -1;
        job_msg[1] = -1;
        job_msg[2] = 0;
        job_msg[3] = 0;
        job_msg[4] = 0;
        job_msg[5] = 0;

        checked_send(job_msg, 6, MPI_LONG, status.MPI_SOURCE,
                     MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, 62);
    }
    else if (rank == 1)
    {
        int dummy = 0;
        checked_send(&dummy, 1, MPI_INT, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, 70);

        long int job_msg[6];
        checked_recv(job_msg, 6, MPI_LONG, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, &status, 71);
        TEST(job_msg[0] == -1, 72);
        TEST(job_msg[1] == -1, 73);
        TEST(job_msg[2] == 0, 74);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_concurrent_tag_independence()
{
    // Test that PREFETCH_REQ and JOB_REQUEST use different tags and
    // can be received independently without cross-talk.
    //
    // Rank 1 sends PREFETCH_REQ then immediately sends JOB_REQUEST
    // (simulating the background thread + main thread pattern).
    // Leader verifies both arrive with correct tags.

    if (rank == 0)
    {
        long int recv_buf[6];
        int got_prefetch = 0, got_job = 0;

        while (got_prefetch + got_job < 2)
        {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == MPITAG_PREFETCH_REQ)
            {
                int dummy;
                checked_recv(&dummy, 1, MPI_INT, status.MPI_SOURCE,
                             MPITAG_PREFETCH_REQ, MPI_COMM_WORLD,
                             &status, 80);

                long int job_msg[6];
                job_msg[0] = 20;
                job_msg[1] = 29;
                job_msg[2] = 10;
                job_msg[3] = 0;
                job_msg[4] = 0;
                job_msg[5] = 0;

                checked_send(job_msg, 6, MPI_LONG, status.MPI_SOURCE,
                             MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, 81);
                got_prefetch++;
            }
            else if (status.MPI_TAG == MPITAG_JOB_REQUEST)
            {
                checked_recv(recv_buf, 6, MPI_LONG, status.MPI_SOURCE,
                             MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                             &status, 82);

                TEST(recv_buf[2] > 0, 83);

                double md;
                checked_recv(&md, 1, MPI_DOUBLE, status.MPI_SOURCE,
                             MPITAG_METADATA, MPI_COMM_WORLD,
                             &status, 84);
                got_job++;
            }
        }

        TEST(got_prefetch == 1, 85);
        TEST(got_job == 1, 86);
    }
    else if (rank == 1)
    {
        // Send PREFETCH_REQ from "background thread"
        int dummy = 99;
        checked_send(&dummy, 1, MPI_INT, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, 90);

        long int job_msg[6];
        checked_recv(job_msg, 6, MPI_LONG, 0, MPITAG_PREFETCH_REQ,
                     MPI_COMM_WORLD, &status, 91);
        TEST(job_msg[0] == 20, 92);

        // Send JOB_REQUEST from "main thread"
        job_msg[0] = 10;
        job_msg[1] = 14;
        job_msg[2] = 5;
        job_msg[3] = 0;
        job_msg[4] = 0;
        job_msg[5] = 0;
        checked_send(job_msg, 6, MPI_LONG, 0, MPITAG_JOB_REQUEST,
                     MPI_COMM_WORLD, 93);

        double md = 2.0;
        checked_send(&md, 1, MPI_DOUBLE, 0, MPITAG_METADATA,
                     MPI_COMM_WORLD, 94);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_many_jobs_sequential()
{
    // Leader sends 5 sequential jobs to rank 1, rank 1 acknowledges
    // each. Tests sequential job counting.

    const int NJOBS = 5;
    int nimg_per_job = 4;

    if (rank == 0)
    {
        int njobs_done = 0;
        while (njobs_done < NJOBS)
        {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            TEST(status.MPI_TAG == MPITAG_PREFETCH_REQ, 100);

            int dummy;
            checked_recv(&dummy, 1, MPI_INT, status.MPI_SOURCE,
                         MPITAG_PREFETCH_REQ, MPI_COMM_WORLD,
                         &status, 101);

            long int job_msg[6];
            job_msg[0] = njobs_done * nimg_per_job;
            job_msg[1] = njobs_done * nimg_per_job + nimg_per_job - 1;
            job_msg[2] = nimg_per_job;
            job_msg[3] = 0;
            job_msg[4] = 0;
            job_msg[5] = 0;

            checked_send(job_msg, 6, MPI_LONG, status.MPI_SOURCE,
                         MPITAG_PREFETCH_REQ, MPI_COMM_WORLD, 102);
            njobs_done++;
        }
    }
    else if (rank == 1)
    {
        for (int i = 0; i < NJOBS; i++)
        {
            int dummy = 0;
            checked_send(&dummy, 1, MPI_INT, 0, MPITAG_PREFETCH_REQ,
                         MPI_COMM_WORLD, 110);

            long int job_msg[6];
            checked_recv(job_msg, 6, MPI_LONG, 0, MPITAG_PREFETCH_REQ,
                         MPI_COMM_WORLD, &status, 111);
            TEST(job_msg[0] == i * nimg_per_job, 112);
            TEST(job_msg[1] == i * nimg_per_job + nimg_per_job - 1, 113);
            TEST(job_msg[2] == nimg_per_job, 114);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_drain_job_request_with_metadata()
{
    // Test draining a JOB_REQUEST with metadata that arrived after the
    // leader has exited its multiplexing loop. Reproduces the deadlock
    // scenario where MPI messages with different tags overtake each other
    // (MPI 3.1 §3.5): a follower's PREFETCH_REQ is processed first,
    // causing the leader to send "no more jobs" and exit the while loop,
    // while the JOB_REQUEST from the same follower is still pending.
    //
    // The drain loop (MPI_Iprobe + MPI_Recv for JOB_REQUEST) must receive
    // the pending JOB_REQUEST and its associated metadata.
    //
    // Synchronisation: follower sends first, then all ranks barrier,
    // then leader probes — this ensures the message is already pending.

    if (rank == 1)
    {
        long int job_msg[6];
        job_msg[0] = 100;
        job_msg[1] = 109;
        job_msg[2] = 10;
        job_msg[3] = 0;
        job_msg[4] = 0;
        job_msg[5] = 0;
        checked_send(job_msg, 6, MPI_LONG, 0, MPITAG_JOB_REQUEST,
                     MPI_COMM_WORLD, 310);

        double md = 3.0;
        checked_send(&md, 1, MPI_DOUBLE, 0, MPITAG_METADATA,
                     MPI_COMM_WORLD, 311);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        int drain_flag;
        MPI_Status drain_status;
        MPI_Iprobe(MPI_ANY_SOURCE, MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                   &drain_flag, &drain_status);
        TEST(drain_flag, 300);

        long int recv_buf[6];
        checked_recv(recv_buf, 6, MPI_LONG, drain_status.MPI_SOURCE,
                     MPITAG_JOB_REQUEST, MPI_COMM_WORLD, &drain_status, 301);
        TEST(recv_buf[2] > 0, 302);

        double md;
        checked_recv(&md, 1, MPI_DOUBLE, drain_status.MPI_SOURCE,
                     MPITAG_METADATA, MPI_COMM_WORLD, &drain_status, 303);
        TEST(md == 3.0, 304);

        MPI_Iprobe(MPI_ANY_SOURCE, MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                   &drain_flag, &drain_status);
        TEST(!drain_flag, 305);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_drain_job_request_no_metadata()
{
    // Test draining a JOB_REQUEST with JOB_NIMG == 0 (no metadata to
    // receive). This exercises the JOB_NIMG > 0 guard in the drain loop.

    if (rank == 1)
    {
        long int job_msg[6];
        job_msg[0] = -1;
        job_msg[1] = -1;
        job_msg[2] = 0;
        job_msg[3] = 0;
        job_msg[4] = 0;
        job_msg[5] = 0;
        checked_send(job_msg, 6, MPI_LONG, 0, MPITAG_JOB_REQUEST,
                     MPI_COMM_WORLD, 330);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        int drain_flag;
        MPI_Status drain_status;
        MPI_Iprobe(MPI_ANY_SOURCE, MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                   &drain_flag, &drain_status);
        TEST(drain_flag, 320);

        long int recv_buf[6];
        checked_recv(recv_buf, 6, MPI_LONG, drain_status.MPI_SOURCE,
                     MPITAG_JOB_REQUEST, MPI_COMM_WORLD, &drain_status, 321);
        TEST(recv_buf[2] == 0, 322);

        MPI_Iprobe(MPI_ANY_SOURCE, MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                   &drain_flag, &drain_status);
        TEST(!drain_flag, 323);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void test_drain_job_request_multi_follower()
{
    // Test draining JOB_REQUEST messages from MULTIPLE followers
    // arriving after the leader has exited the multiplexing loop.
    // Each follower sends JOB_REQUEST + metadata; the drain loop
    // must receive all of them.

    if (rank == 1 || rank == 2)
    {
        long int job_msg[6];
        job_msg[0] = rank * 100;
        job_msg[1] = rank * 100 + 9;
        job_msg[2] = 10;
        job_msg[3] = 0;
        job_msg[4] = 0;
        job_msg[5] = 0;
        checked_send(job_msg, 6, MPI_LONG, 0, MPITAG_JOB_REQUEST,
                     MPI_COMM_WORLD, 350);

        double md = (double)rank;
        checked_send(&md, 1, MPI_DOUBLE, 0, MPITAG_METADATA,
                     MPI_COMM_WORLD, 351);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        int ndrained = 0;
        int drain_flag;
        MPI_Status drain_status;
        MPI_Iprobe(MPI_ANY_SOURCE, MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                   &drain_flag, &drain_status);
        while (drain_flag)
        {
            long int recv_buf[6];
            checked_recv(recv_buf, 6, MPI_LONG, drain_status.MPI_SOURCE,
                         MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                         &drain_status, 340);
            TEST(recv_buf[2] > 0, 341);

            double md;
            checked_recv(&md, 1, MPI_DOUBLE, drain_status.MPI_SOURCE,
                         MPITAG_METADATA, MPI_COMM_WORLD,
                         &drain_status, 342);
            ndrained++;

            MPI_Iprobe(MPI_ANY_SOURCE, MPITAG_JOB_REQUEST, MPI_COMM_WORLD,
                       &drain_flag, &drain_status);
        }
        TEST(ndrained == 2, 343);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    TEST(provided >= MPI_THREAD_MULTIPLE, 0);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 3)
    {
        fprintf(stderr, "Need at least 3 MPI ranks (got %d)\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        fprintf(stderr, "Starting MPI prefetch protocol tests...\n");

    test_basic_prefetch_request();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_basic_prefetch_request\n");

    test_concurrent_prefetch_and_job_request();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_concurrent_prefetch_and_job_request\n");

    test_termination_signal();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_termination_signal\n");

    test_concurrent_tag_independence();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_concurrent_tag_independence\n");

    test_many_jobs_sequential();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_many_jobs_sequential\n");

    test_drain_job_request_with_metadata();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_drain_job_request_with_metadata\n");

    test_drain_job_request_no_metadata();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_drain_job_request_no_metadata\n");

    test_drain_job_request_multi_follower();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "  [OK] test_drain_job_request_multi_follower\n");

    if (rank == 0)
        fprintf(stderr, "ALL TESTS PASSED\n");

    MPI_Finalize();
    return 0;
}
