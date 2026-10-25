#ifndef ASYNC_MRC_STACK_H_
#define ASYNC_MRC_STACK_H_

#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "src/async_file_writer.h"
#include "src/image.h"

// Writes the 2D images given to add(), in order, as one MRC stack, on the background writer: the same bytes
// (header time stamp aside) as Image::write(WRITE_OVERWRITE) of the first image followed by WRITE_APPEND of each
// other one, which opens, rewrites the header of and closes the file for every image.
//
// The first image is written at once and the last one by Image::write(WRITE_APPEND) in finish(), after the
// header's slice count was set to N-1, so Image::write makes the final header itself (the last image's
// statistics and pixel size, as before). The images in between are cast and appended in chunks of about
// "chunk_bytes". The file is written under AsyncFileWriter::tmpName and renamed once complete and checked
// (size), so no partial stack ever has the final name. Nothing here waits: use AsyncFileWriter::waitOrThrow().
class AsyncMrcStack
{
public:
    // chunk_bytes 0: RELION_ASYNC_STACK_CHUNK_KB, or 32 MB
    AsyncMrcStack(const std::string &fn, DataType datatype, size_t chunk_bytes = 0)
        : st_(new State), datatype_(datatype), chunk_bytes_(chunk_bytes)
    {
        if (chunk_bytes_ == 0)
        {
            const char *e = getenv("RELION_ASYNC_STACK_CHUNK_KB");
            chunk_bytes_ = e && atoll(e) > 0 ? (size_t) atoll(e) << 10 : (size_t) 32 << 20;
        }
        st_->fn = fn;
        st_->tmp = AsyncFileWriter::tmpName(fn);
    }

    // An unfinished stack (an error left the loop that fills it) is abandoned: its temporary file goes
    ~AsyncMrcStack()
    {
        if (finished_ || n_ == 0) return;
        std::shared_ptr<State> st = st_;
        AsyncFileWriter::instance().submit([st] { st->failed = true; remove(st->tmp.c_str()); });
    }

    long int size() const { return n_; }

    // Takes a copy of "img" (pixels and main header)
    void add(const Image<RFLOAT> &img)
    {
        std::shared_ptr<State> st = st_;
        std::shared_ptr<Image<RFLOAT> > mine(new Image<RFLOAT>(img));
        const size_t bytes = MULTIDIM_SIZE(img.data) * sizeof(RFLOAT);
        if (n_ == 0)
        {
            DataType dt = datatype_;
            st->x = XSIZE(img.data); st->y = YSIZE(img.data);
            AsyncFileWriter::instance().submit([st, mine, dt]
            {
                if (st->failed) return;
                guard(st, [&] { mine->write(st->tmp, -1, false, WRITE_OVERWRITE, dt); });
            }, bytes);
        }
        else
        {
            if (XSIZE(img.data) != st->x || YSIZE(img.data) != st->y || ZSIZE(img.data) != 1 || NSIZE(img.data) != 1)
                REPORT_ERROR("AsyncMrcStack: image " + integerToString(n_ + 1) + " of " + st->fn + " differs in size from the first");
            if (held_)
            {
                chunk_.push_back(held_);
                chunk_size_ += bytes;
            }
            if (chunk_size_ >= chunk_bytes_) flushChunk();
        }
        held_ = n_ == 0 ? nullptr : mine;
        n_++;
    }

    // Submits the last image, the check and the rename
    void finish()
    {
        if (finished_ || n_ == 0) { finished_ = true; return; }
        finished_ = true;
        flushChunk();
        std::shared_ptr<State> st = st_;
        const long int n = n_;
        if (n >= 2)
        {
            std::shared_ptr<Image<RFLOAT> > last = held_;
            DataType dt = datatype_;
            const size_t bytes = MULTIDIM_SIZE(last->data) * sizeof(RFLOAT);
            AsyncFileWriter::instance().submit([st, last, dt, n]
            {
                if (st->failed) return;
                guard(st, [&]
                {
                    // the header of the stack so far says N-1 images, so that Image::write(WRITE_APPEND) makes it say N
                    FILE *f = fopen(st->tmp.c_str(), "r+b");
                    if (!f) REPORT_ERROR("cannot open " + st->tmp);
                    const int nz = (int) (n - 1);
                    const bool ok = fseek(f, 8, SEEK_SET) == 0 && fwrite(&nz, sizeof(int), 1, f) == 1;
                    if (fclose(f) != 0 || !ok) REPORT_ERROR("cannot update the header of " + st->tmp);
                    last->write(st->tmp, -1, false, WRITE_APPEND, dt);
                });
            }, bytes);
        }
        DataType dt = datatype_;
        AsyncFileWriter::instance().submit([st, n, dt]
        {
            if (st->failed) { remove(st->tmp.c_str()); return; }
            guard(st, [&]
            {
                struct stat sb;
                const long long want = 1024LL + (long long) n * st->x * st->y * (long long) gettypesize(dt);
                if (stat(st->tmp.c_str(), &sb) != 0 || (long long) sb.st_size != want)
                    REPORT_ERROR("short write of " + st->tmp + ": " + std::to_string(stat(st->tmp.c_str(), &sb) == 0 ? (long long) sb.st_size : -1LL)
                                 + " bytes instead of " + std::to_string(want) + " (disk full?)");
                if (rename(st->tmp.c_str(), st->fn.c_str()) != 0) REPORT_ERROR("cannot rename " + st->tmp + " to " + st->fn);
            });
        });
        held_.reset();
    }

private:
    struct State
    {
        std::string fn, tmp;
        long int x = 0, y = 0;
        bool failed = false; // only the writer thread touches it
    };

    // Runs "f"; on failure removes the temporary file, marks the stack failed and rethrows with the file name
    template <typename F>
    static void guard(const std::shared_ptr<State> &st, F f)
    {
        try { f(); }
        catch (const RelionError &e)
        {
            st->failed = true; remove(st->tmp.c_str());
            REPORT_ERROR("cannot write " + st->fn + ": " + e.msg.substr(e.msg.compare(0, 8, "ERROR: \n") == 0 ? 8 : 0));
        }
    }

    void flushChunk()
    {
        if (chunk_.empty()) return;
        std::shared_ptr<State> st = st_;
        std::shared_ptr<std::vector<std::shared_ptr<Image<RFLOAT> > > > chunk(new std::vector<std::shared_ptr<Image<RFLOAT> > >());
        chunk->swap(chunk_);
        const size_t bytes = chunk_size_;
        chunk_size_ = 0;
        DataType dt = datatype_;
        AsyncFileWriter::instance().submit([st, chunk, dt]
        {
            if (st->failed) return;
            guard(st, [&]
            {
                FILE *f = fopen(st->tmp.c_str(), "r+b");
                if (!f) REPORT_ERROR("cannot open " + st->tmp);
                bool ok = fseek(f, 0, SEEK_END) == 0;
                const size_t n = (size_t) st->x * st->y, page = n * gettypesize(dt);
                std::vector<char> buf(page);
                for (size_t i = 0; ok && i < chunk->size(); i++)
                {
                    Image<RFLOAT> &im = *(*chunk)[i];
                    im.castPage2Datatype(MULTIDIM_ARRAY(im.data), buf.data(), dt, n);
                    ok = fwrite(buf.data(), page, 1, f) == 1;
                }
                if (fclose(f) != 0) ok = false;
                if (!ok) REPORT_ERROR("error writing to " + st->tmp + " (disk full?)");
            });
        }, bytes);
    }

    std::shared_ptr<State> st_;
    DataType datatype_;
    size_t chunk_bytes_;
    long int n_ = 0;
    bool finished_ = false;
    std::shared_ptr<Image<RFLOAT> > held_; // the newest image after the first, not yet queued: finish() writes it last
    std::vector<std::shared_ptr<Image<RFLOAT> > > chunk_;
    size_t chunk_size_ = 0;
};

#endif
