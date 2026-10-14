#ifndef GPU_SWITCH_H_
#define GPU_SWITCH_H_

#include <cstdlib>
#include <cstring>

/*
 * Environment switches of the GPU speed-ups that are on by default. Unset means on; "0", "off",
 * "no" or "false" turn the switch off. Every switch still backs off to the older path by itself
 * when its conditions are not met (see the code that reads it).
 */
namespace gpuswitch
{

inline bool isOffValue(const char *e)
{
	return e != NULL && (strcmp(e, "0") == 0 || strcmp(e, "off") == 0 || strcmp(e, "no") == 0 || strcmp(e, "false") == 0);
}

/// 0 off, 1 on (unset, "1", or anything else not off), 2 "check" (on, and compared with the older path)
inline int mode(const char *name)
{
	const char *e = getenv(name);
	if (e == NULL) return 1;
	if (isOffValue(e)) return 0;
	return strcmp(e, "check") == 0 ? 2 : 1;
}

inline bool on(const char *name) { return mode(name) != 0; }

}

#endif /* GPU_SWITCH_H_ */
