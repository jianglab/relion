/*
 * tests/unit/main.cpp
 *
 * Catch2 test runner entry point for the RELION unit test suite.
 * Each test file includes its own TEST_CASE blocks; this file only
 * provides the main() function.
 */

// glibc 2.34+ makes SIGSTKSZ a runtime value; provide a constant for catch.hpp
#include <signal.h>
#ifndef SIGSTKSZ
#  define SIGSTKSZ 65536
#elif defined(__GLIBC__) && __GLIBC__ >= 2 && defined(__GLIBC_MINOR__) && __GLIBC_MINOR__ >= 34
// SIGSTKSZ is no longer constexpr in glibc 2.34+; redefine to a literal
#  undef  SIGSTKSZ
#  define SIGSTKSZ 65536
#endif

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
