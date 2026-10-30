#include "src/tuning_cache.h"

#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <map>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tuning
{

namespace
{
const char *kHeader = "tunable,device,sm_count,driver,runtime,build,key,value,rel_time,date\n";
const size_t kColumns = 10;

// fields are plain text without commas and newlines
std::string clean(const std::string &s)
{
	std::string r = s;
	for (char &c : r)
		if (c == ',' || c == '\n' || c == '\r') c = ';';
	return r;
}

std::vector<std::string> split(const std::string &line)
{
	std::vector<std::string> f;
	size_t from = 0;
	while (true)
	{
		const size_t p = line.find(',', from);
		if (p == std::string::npos) { f.push_back(line.substr(from)); break; }
		f.push_back(line.substr(from, p - from));
		from = p + 1;
	}
	return f;
}

// An exclusive or shared lock on <file>.lock for the life of the object
struct Lock
{
	int fd = -1;
	Lock(const std::string &path, bool exclusive, bool create)
	{
		fd = open((path + ".lock").c_str(), (create ? O_CREAT : 0) | (exclusive ? O_RDWR : O_RDONLY) | O_CLOEXEC, 0600);
		if (fd >= 0 && flock(fd, exclusive ? LOCK_EX : LOCK_SH) != 0) { close(fd); fd = -1; }
	}
	~Lock() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
	bool ok() const { return fd >= 0; }
};

bool readAll(const std::string &path, std::string &out)
{
	FILE *f = fopen(path.c_str(), "rb");
	if (!f) return false;
	char buf[8192];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
	fclose(f);
	return true;
}

// the complete lines of a file's text (a last line without newline may be half written: dropped)
std::vector<std::string> lines(const std::string &text)
{
	std::vector<std::string> v;
	size_t from = 0;
	while (true)
	{
		const size_t p = text.find('\n', from);
		if (p == std::string::npos) break;
		v.push_back(text.substr(from, p - from));
		from = p + 1;
	}
	return v;
}

std::string rowKey(const std::vector<std::string> &f)
{
	return f[0] + "\x1f" + f[1] + "\x1f" + f[2] + "\x1f" + f[6];
}

bool writeAll(int fd, const std::string &s)
{
	size_t done = 0;
	while (done < s.size())
	{
		const ssize_t w = write(fd, s.data() + done, s.size() - done);
		if (w < 0) { if (errno == EINTR) continue; return false; }
		done += (size_t) w;
	}
	return true;
}

void makeParentDir(const std::string &path)
{
	const size_t p = path.rfind('/');
	if (p == std::string::npos || p == 0) return;
	mkdir(path.substr(0, p).c_str(), 0700);   // one level: $HOME/.relion; fails harmlessly if it exists
}
}

std::string defaultPath()
{
	const char *off = getenv("RELION_TUNING_CACHE");
	if (off && (!strcmp(off, "off") || !strcmp(off, "0") || !strcmp(off, "no"))) return "";
	const char *file = getenv("RELION_TUNING_FILE");
	if (file && *file) return file;
	const char *home = getenv("HOME");
	if (!home || !*home) return "";
	return std::string(home) + "/.relion/tuning.csv";
}

Cache::Cache(const Context &ctx, const std::string &path) : ctx_(ctx), path_(path.empty() ? defaultPath() : path) {}

bool Cache::get(const std::string &tunable, const std::string &key, Entry &out) const
{
	if (path_.empty()) return false;
	std::string text;
	{
		Lock lock(path_, false, false);   // no lock file yet: nothing has been written
		if (!lock.ok() || !readAll(path_, text)) return false;
	}
	const std::string dev = clean(ctx_.device), tun = clean(tunable), k = clean(key), sm = std::to_string(ctx_.sm_count);
	bool found = false;
	for (const std::string &l : lines(text))
	{
		const std::vector<std::string> f = split(l);
		if (f.size() != kColumns || f[0] != tun || f[1] != dev || f[2] != sm || f[6] != k) continue;
		out.value = f[7];
		out.rel_time = atof(f[8].c_str());
		out.date = f[9];
		found = true;   // the last matching row is the newest
	}
	return found;
}

bool Cache::put(const std::string &tunable, const std::string &key, const std::string &value, double rel_time, double same_within)
{
	if (path_.empty()) return false;
	makeParentDir(path_);
	Lock lock(path_, true, true);
	if (!lock.ok()) return false;

	// the newest row already says this: nothing to add (equal, or numbers within same_within of each other)
	{
		std::string text;
		const std::string dev = clean(ctx_.device), tun = clean(tunable), k = clean(key), sm = std::to_string(ctx_.sm_count), v = clean(value);
		std::string last;
		bool found = false;
		if (readAll(path_, text))
			for (const std::string &l : lines(text))
			{
				const std::vector<std::string> f = split(l);
				if (f.size() == kColumns && f[0] == tun && f[1] == dev && f[2] == sm && f[6] == k) { last = f[7]; found = true; }
			}
		if (found && last == v) return true;
		if (found && same_within > 0.)
		{
			char *e1 = NULL, *e2 = NULL;
			const double a = strtod(last.c_str(), &e1), b = strtod(v.c_str(), &e2);
			if (*e1 == 0 && *e2 == 0 && a > 0. && std::fabs(a - b) <= same_within * a) return true;
		}
	}
	char date[32] = "";
	const time_t now = time(NULL);
	struct tm tmv;
	if (gmtime_r(&now, &tmv)) strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%SZ", &tmv);
	char rel[32];
	snprintf(rel, sizeof(rel), "%.4f", rel_time);
	const std::string row = clean(tunable) + "," + clean(ctx_.device) + "," + std::to_string(ctx_.sm_count) + "," + clean(ctx_.driver) + "," +
		clean(ctx_.runtime) + "," + clean(ctx_.build) + "," + clean(key) + "," + clean(value) + "," + rel + "," + date + "\n";

	struct stat st;
	const bool fresh = stat(path_.c_str(), &st) != 0 || st.st_size == 0;
	const int fd = open(path_.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
	if (fd < 0) return false;
	const bool ok = writeAll(fd, fresh ? std::string(kHeader) + row : row);   // one write: no interleaving even without the lock
	close(fd);
	if (!ok) return false;

	// keep the file small: rewrite it with the newest row per (tunable, device, sm, key)
	if (!fresh && (size_t) st.st_size > kCompactRows * 120)
	{
		std::string text;
		if (readAll(path_, text))
		{
			const std::vector<std::string> all = lines(text);
			if (all.size() > kCompactRows)
			{
				std::map<std::string, size_t> newest;
				for (size_t i = 0; i < all.size(); i++)
				{
					const std::vector<std::string> f = split(all[i]);
					if (f.size() == kColumns && f[0] != "tunable") newest[rowKey(f)] = i;
				}
				std::string out = kHeader;
				for (const auto &kv : newest) out += all[kv.second] + "\n";
				const std::string tmp = path_ + ".tmp." + std::to_string((long) getpid());
				const int tfd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
				if (tfd >= 0)
				{
					const bool w = writeAll(tfd, out);
					close(tfd);
					if (!w || rename(tmp.c_str(), path_.c_str()) != 0) unlink(tmp.c_str());
				}
			}
		}
	}
	return true;
}

} // namespace tuning
