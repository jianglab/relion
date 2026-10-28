/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/manualpicker.h"
#include "src/displayer.h"
#include "src/virtual_movie_averages.h"
#include "src/fftw.h"
#include "src/transformations.h"
#include <signal.h>
#include <cstdio>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <omp.h>
#include <FL/x.H>
#ifdef __APPLE__
#include <objc/runtime.h>
#include <objc/message.h>
#endif

std::vector<int> imics;
std::vector<FileName> global_fn_mics;
std::vector<FileName> global_fn_picks;
std::vector<FileName> global_fn_ctfs;
std::vector<FileName> global_fn_foms;
std::vector<bool> selected;
std::vector<int> number_picked;
std::vector<Fl_Button*> viewmic_buttons;
std::vector<Fl_Button*> viewctf_buttons;
std::vector<Fl_Text_Display*> text_displays;
std::vector<Fl_Text_Display*> count_displays;
std::vector<Fl_Text_Display*> defocus_displays;
std::vector<Fl_Text_Display*> ctfres_displays;
std::vector<Fl_Check_Button*> check_buttons;
int first_pick_viewed, last_pick_viewed;
int prev_pick_viewed = -1;
int last_ctf_viewed;

#define GUI_HIGHLIGHT_ACTIVE_COLOR (fl_rgb_color(200, 230, 255))

void highlight_active(int imic)
{
	// Reset previous active row to default color (respecting selection state)
	if (prev_pick_viewed >= 0 && prev_pick_viewed < (int)text_displays.size())
	{
		Fl_Color bg = selected[prev_pick_viewed] ? GUI_INPUT_COLOR : GUI_BACKGROUND_COLOR;
		text_displays[prev_pick_viewed]->color(bg, bg);
		text_displays[prev_pick_viewed]->redraw();
		count_displays[prev_pick_viewed]->color(bg, bg);
		count_displays[prev_pick_viewed]->redraw();
		if (prev_pick_viewed < (int)defocus_displays.size())
		{
			defocus_displays[prev_pick_viewed]->color(bg, bg);
			defocus_displays[prev_pick_viewed]->redraw();
		}
		if (prev_pick_viewed < (int)ctfres_displays.size())
		{
			ctfres_displays[prev_pick_viewed]->color(bg, bg);
			ctfres_displays[prev_pick_viewed]->redraw();
		}
	}
	// Highlight the new active row (respecting selection state)
	if (imic >= 0 && imic < (int)text_displays.size())
	{
		Fl_Color bg = selected[imic] ? GUI_HIGHLIGHT_ACTIVE_COLOR : GUI_BACKGROUND_COLOR;
		text_displays[imic]->color(bg, bg);
		text_displays[imic]->redraw();
		count_displays[imic]->color(bg, bg);
		count_displays[imic]->redraw();
		if (imic < (int)defocus_displays.size())
		{
			defocus_displays[imic]->color(bg, bg);
			defocus_displays[imic]->redraw();
		}
		if (imic < (int)ctfres_displays.size())
		{
			ctfres_displays[imic]->color(bg, bg);
			ctfres_displays[imic]->redraw();
		}
	}
	prev_pick_viewed = imic;
}

bool   global_has_ctf;
bool   global_pick_startend;
StartEndStyle global_startend_style;
bool   global_pick_lines;
RFLOAT global_angpix;
RFLOAT global_coord_scale;
RFLOAT global_lowpass;
RFLOAT global_highpass;
bool global_do_topaz_denoise;
RFLOAT global_particle_diameter;
RFLOAT global_sigma_contrast;
RFLOAT global_black_val;
RFLOAT global_white_val;
RFLOAT global_micscale;
RFLOAT global_ctfscale;
RFLOAT global_ctfsigma;
RFLOAT global_minimum_fom;
RFLOAT global_blue_value;
RFLOAT global_red_value;
int    global_total_count;
int global_nr_simultaneous;
FileName global_fn_odir;
FileName global_pickname;
FileName global_fn_color;
FileName global_color_label;
bool global_do_color;
int global_nr_preload;
int global_nr_load_threads;
int global_nr_vmovie_builders;
int global_nr_prebuild;
// Remember the micrograph opened last, to continue from it (when picking, not when only displaying)
static bool record_last_viewed = false;

// ---------------------------------------------------------------------------
// Loading ahead

// A micrograph as a pick window shows it: filtered, contrast applied and scaled (what canvas->fill makes of it)
struct PreparedMic
{
	Image<RFLOAT> I;
	RFLOAT minval, maxval;
};

static std::shared_ptr<PreparedMic> prepareMicrograph(int imic)
{
	std::shared_ptr<PreparedMic> mic(new PreparedMic);
	mic->I.read(global_fn_mics[imic]);
	MultidimArray<RFLOAT> &img = mic->I();
	if (global_lowpass > 0.)
		lowPassFilterMap(img, global_lowpass, global_angpix);
	if (global_highpass > 0.)
		highPassFilterMap(img, global_highpass, global_angpix, 25);
	mic->minval = global_black_val;
	mic->maxval = global_white_val;
	RFLOAT sigma_contrast = global_sigma_contrast;
	getImageContrast(img, mic->minval, mic->maxval, sigma_contrast);
	if (ABS(global_micscale - 1.) > 0.01)
		selfScaleToSize(img, CEIL(XSIZE(img) * global_micscale), CEIL(YSIZE(img) * global_micscale));
	return mic;
}

// Threads for one background load or build: the cores shared by the loaders and the builders
static int backgroundJobThreads()
{
	const int jobs = std::max(1, global_nr_load_threads + global_nr_vmovie_builders);
	return std::max(1, omp_get_max_threads() / jobs);
}

// Loads the micrographs after those viewed on background threads, ready to show, so that the next one opens at once;
// the one before those viewed is kept for Z. One that is not ready when asked for is loaded by the caller (or waited
// for, while a thread is loading it), and one whose background load failed is loaded again, so errors show as before.
class MicrographPreloader
{
public:
	MicrographPreloader() : lo(0), hi(-1), stop(false) {}

	// Waits for the loads in progress (a virtual micrograph that is not cached yet may take a while)
	~MicrographPreloader()
	{
		int loading = 0;
		{
			std::lock_guard<std::mutex> lock(m);
			stop = true;
			queue.clear();
			for (auto &s : slots)
				if (s.second.loading) loading++;
		}
		cv.notify_all();
		if (loading > 0)
			std::cout << " Waiting for " << loading << " micrograph(s) being loaded in the background..." << std::endl;
		for (size_t i = 0; i < workers.size(); i++)
			workers[i].join();
	}

	// Micrographs first..last are about to be viewed: load them and global_nr_preload after them, the nearest first
	void want(int first, int last)
	{
		if (global_nr_preload <= 0)
			return;
		{
			std::lock_guard<std::mutex> lock(m);
			lo = std::max(0, first - 1);
			hi = std::min((int)global_fn_mics.size() - 1, last + global_nr_preload);
			for (auto it = slots.begin(); it != slots.end(); )
			{
				if ((it->first < lo || it->first > hi) && !it->second.loading)
					it = slots.erase(it);
				else
					++it;
			}
			queue.clear();
			for (int i = first; i <= hi; i++)
			{
				Slot &s = slots[i];
				if (!s.mic && !s.loading && !s.failed)
					queue.push_back(i);
			}
			if (workers.empty())
			{
				const int threads = backgroundJobThreads();
				for (int i = 0; i < global_nr_load_threads; i++)
					workers.push_back(std::thread(&MicrographPreloader::run, this, threads));
			}
		}
		cv.notify_all();
	}

	std::shared_ptr<PreparedMic> take(int imic)
	{
		std::unique_lock<std::mutex> lock(m);
		std::map<int, Slot>::iterator it = slots.find(imic);
		if (it != slots.end() && it->second.loading)
		{
			cv.wait(lock, [this, imic] { std::map<int, Slot>::iterator j = slots.find(imic); return j == slots.end() || !j->second.loading; });
			it = slots.find(imic);
		}
		if (it != slots.end() && it->second.mic)
			return it->second.mic;
		if (it != slots.end())
		{
			// Not started yet, or failed: load it here rather than wait for a thread
			queue.erase(std::remove(queue.begin(), queue.end(), imic), queue.end());
			it->second.loading = true;
		}
		const bool tracked = (it != slots.end());
		lock.unlock();

		std::shared_ptr<PreparedMic> mic;
		try
		{
			mic = prepareMicrograph(imic);
		}
		catch (...)
		{
			if (tracked)
			{
				lock.lock();
				slots.erase(imic);
				lock.unlock();
				cv.notify_all();
			}
			throw;
		}

		lock.lock();
		if (tracked || (imic >= lo && imic <= hi))
		{
			Slot &s = slots[imic];
			s.mic = mic;
			s.loading = s.failed = false;
		}
		lock.unlock();
		cv.notify_all();
		return mic;
	}

private:
	struct Slot
	{
		std::shared_ptr<PreparedMic> mic;   // null while queued or loading
		bool loading, failed;
		Slot() : loading(false), failed(false) {}
	};
	std::mutex m;
	std::condition_variable cv;
	std::map<int, Slot> slots;   // micrographs lo..hi: loaded, loading or queued (and ones outside still loading)
	std::deque<int> queue;
	std::vector<std::thread> workers;
	int lo, hi;
	bool stop;

	void run(int threads)
	{
		// The reader's threads, for building a virtual micrograph that is not cached yet
		omp_set_num_threads(threads);
		for (;;)
		{
			int imic;
			{
				std::unique_lock<std::mutex> lock(m);
				cv.wait(lock, [this] { return stop || !queue.empty(); });
				if (stop) return;
				imic = queue.front();
				queue.pop_front();
				slots[imic].loading = true;
			}
			std::shared_ptr<PreparedMic> mic;
			try
			{
				// FFTW's cleanup, were it run, would wait for the transforms of this load
				FftwBackgroundScope fftw_in_use;
				mic = prepareMicrograph(imic);
			}
			catch (RelionError&) {}   // the viewer loads it again and reports the error
			catch (std::exception&) {}
			{
				std::lock_guard<std::mutex> lock(m);
				std::map<int, Slot>::iterator it = slots.find(imic);
				if (it != slots.end())
				{
					if (imic < lo || imic > hi)
						slots.erase(it);
					else
					{
						it->second.mic = mic;
						it->second.loading = false;
						it->second.failed = !mic;
					}
				}
			}
			cv.notify_all();
		}
	}
};

// Made at first use, so that at exit it goes (waiting for its threads) before the globals that those threads use
static MicrographPreloader &preloader()
{
	static MicrographPreloader p;
	return p;
}

// Virtual movie averages further ahead than those preloaded are built into the cache in the background, and so is the
// prebuild batch (the destructor waits for the builds in progress)
static vmovies::ReadAhead &buildAhead()
{
	static vmovies::ReadAhead read_ahead;
	static const bool configured = (read_ahead.limitBuilders(global_nr_vmovie_builders),
	                                read_ahead.setBuildThreads(std::min(4, backgroundJobThreads())), true);
	(void) configured;
	return read_ahead;
}

// Micrographs imic and on will be viewed next
static void buildAheadFrom(int imic)
{
	if (global_nr_vmovie_builders <= 0 || !buildAhead().enabled())
		return;
	std::vector<std::string> next;
	for (int k = imic; k < (int)global_fn_mics.size() && (int)next.size() < vmovies::ReadAhead::WINDOW; k++)
		next.push_back(global_fn_mics[k]);
	buildAhead().upcoming(next);
}

// Where picking was left off: the micrograph opened last, kept in the output directory
static FileName lastViewedFile()
{
	return global_fn_odir + ".last_viewed_micrograph";
}

static void recordLastViewed(int imic)
{
	if (!record_last_viewed || global_fn_odir == "")
		return;
	std::ofstream fh(lastViewedFile().c_str(), std::ios::out | std::ios::trunc);
	if (fh)
		fh << global_fn_mics[imic] << std::endl;
}

static int resumePoint()
{
	const int n = global_fn_mics.size();
	if (!record_last_viewed || global_fn_odir == "" || n == 0)
		return 0;
	std::ifstream fh(lastViewedFile().c_str());
	std::string name;
	if (fh && std::getline(fh, name))
	{
		for (int i = 0; i < n; i++)
			if (global_fn_mics[i] == name)
				return i;
		return 0;
	}
	// No record (picked before it was kept): after the last micrograph with picks, unless all have some (continued autopicking)
	int last = -1;
	bool all = true;
	for (int i = 0; i < n; i++)
	{
		if (number_picked[i] > 0) last = i;
		else all = false;
	}
	return (last < 0 || all) ? 0 : std::min(last + 1, n - 1);
}

// Before anything is opened: load the micrograph to continue from and those after it, and prebuild virtual ones
static void startLoadingAt(int imic)
{
	if (imic < 0 || imic >= (int)global_fn_mics.size())
		return;
	preloader().want(imic, imic);
	const int after_preload = imic + 1 + global_nr_preload;
	buildAheadFrom(after_preload);
	// Only when picking: not when displaying the picks of a finished job
	if (global_nr_prebuild <= 0 || !record_last_viewed)
		return;
	if (global_nr_vmovie_builders <= 0 || vmovies::cacheDirectory() == "")
	{
		std::cout << " Not prebuilding virtual micrographs: "
		          << (global_nr_vmovie_builders <= 0 ? "--vmovie_builders is 0" : "the virtual movie average cache is off") << std::endl;
		return;
	}
	// The preloader loads (and builds) the first ones itself
	std::vector<std::string> list;
	for (int k = after_preload; k < imic + global_nr_prebuild && k < (int)global_fn_mics.size(); k++)
		list.push_back(global_fn_mics[k]);
	if (list.empty())
		return;
	buildAhead().batch(list);
	std::cout << " Building in the background the virtual micrographs among the " << global_nr_prebuild
	          << " from " << global_fn_mics[imic] << " on" << std::endl;
}

#ifdef __APPLE__
// Deferred call to disable macOS native tabbing on a window
static void disable_tabbing_cb(void *data)
{
	Fl_Window *win = (Fl_Window*)data;
	void *xid = fl_xid(win);
	if (xid)
	{
		SEL setTabbingMode = sel_registerName("setTabbingMode:");
		if (setTabbingMode)
		{
			typedef void (*msgSend_fn)(void*, SEL, long);
			((msgSend_fn)objc_msgSend)(xid, setTabbingMode, 2);
		}
	}
}
#endif

class MyPickCanvas : public pickerViewerCanvas
{
public:
	int my_imic;
	MyPickCanvas(int X, int Y, int W, int H) : pickerViewerCanvas(X, Y, W, H), my_imic(-1) {}
};

struct PickWindowEntry {
	Fl_Window *win;
	Fl_Scroll *scroll;
	MyPickCanvas *canvas;
	int imic;
};
std::vector<PickWindowEntry> pick_windows;
std::map<int, std::pair<int,int>> mic_scroll_pos;

// Hide a pick window and delete it once the current event is done: it may be the
// window whose key press or close button is being handled right now
static void destroyPickWindow(Fl_Window *win)
{
#ifdef __APPLE__
	Fl::remove_timeout(disable_tabbing_cb, win);
#endif
	win->hide();
	Fl::delete_widget(win);
}

// Pick buttons and Z/X only ask for a micrograph; it is loaded from a timeout. Loading blocks
// the event loop, so presses made meanwhile queue up: they then only move the target, and the
// micrograph browsed to is loaded once, instead of every one passed on the way.
static int pending_imic = -1;
static int pending_nr_simultaneous = 1;
// Z/X wait this long (s) for the next press, so that browsing quickly or holding a key loads nothing on the way
#define NAV_KEY_DELAY 0.15

static void viewMics(int imic, int nr_simultaneous);

static void cb_load_pending(void*)
{
	const int imic = pending_imic;
	pending_imic = -1;
	if (imic >= 0)
		viewMics(imic, pending_nr_simultaneous);
}

static void requestViewMic(int imic, int nr_simultaneous, double delay)
{
	if (imic < 0 || imic >= (int)viewmic_buttons.size())
		return;
	pending_imic = imic;
	pending_nr_simultaneous = nr_simultaneous;
	highlight_active(imic);
	Fl::remove_timeout(cb_load_pending);
	Fl::add_timeout(delay, cb_load_pending);
}

// Z/X step from the micrograph still waiting to be loaded, else from the one shown
static int navigationBase(int shown)
{
	return (pending_imic >= 0) ? pending_imic : shown;
}

// Z/X are handled by the window, not the canvas: keys reach the window whichever of its
// widgets has the keyboard focus (the canvas, a scrollbar, or none)
class PickWindow : public Fl_Double_Window
{
public:
	PickWindow(int W, int H, const char *L) : Fl_Double_Window(W, H, L)
	{
		callback(cb_close);
	}

	int handle(int ev)
	{
		if (Fl_Double_Window::handle(ev))
			return 1;
		if (ev == FL_KEYBOARD || ev == FL_SHORTCUT)
		{
			const int key = Fl::event_key();
			int step = 0;
			if (key == 'z' || key == 'Z') step = -1;
			else if (key == 'x' || key == 'X') step = 1;
			if (step != 0)
			{
				for (auto &entry : pick_windows)
				{
					if (entry.win != this)
						continue;
					requestViewMic(navigationBase(entry.imic) + step, 1, NAV_KEY_DELAY);
					return 1;
				}
			}
		}
		return 0;
	}

	// Closing a pick window deletes it, so that the next Pick, Z or X opens a new one
	// (a hidden window would be reused without being shown)
	static void cb_close(Fl_Widget *w, void *)
	{
		for (int i = 0; i < (int)pick_windows.size(); i++)
		{
			if (pick_windows[i].win != w)
				continue;
			PickWindowEntry &entry = pick_windows[i];
			mic_scroll_pos[entry.imic] = {entry.scroll->hscrollbar.value(), entry.scroll->scrollbar.value()};
			if (entry.imic >= 0 && entry.imic < (int)viewmic_buttons.size())
			{
				viewmic_buttons[entry.imic]->color(GUI_BUTTON_COLOR, GUI_BUTTON_COLOR);
				viewmic_buttons[entry.imic]->redraw();
			}
			pick_windows.erase(pick_windows.begin() + i);
			break;
		}
		destroyPickWindow((Fl_Window*)w);
	}
};

static void viewMics(int imic, int nr_simultaneous)
{
	// Show that the GUI is busy, not frozen, while the micrographs load
	for (Fl_Window *win = Fl::first_window(); win; win = Fl::next_window(win))
		win->cursor(FL_CURSOR_WAIT);
	Fl::flush();

	for (int mymic = first_pick_viewed; mymic <= last_pick_viewed; mymic++)
	{
		if (mymic >= 0 && mymic < count_displays.size())
		{
			MetaDataTable MDcoord;
			FileName fn_coord = global_fn_picks[mymic];
			int my_nr_picked;
			if (exists(fn_coord))
			{
				MDcoord.read(fn_coord);
				if (fabs(global_minimum_fom + 9999.) > 1e-6)
				{
					if (MDcoord.containsLabel(EMDL_PARTICLE_AUTOPICK_FOM))
					{
						my_nr_picked = 0;
						FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDcoord)
						{
							RFLOAT fom;
							MDcoord.getValue(EMDL_PARTICLE_AUTOPICK_FOM, fom);
							if (fom > global_minimum_fom) my_nr_picked++;
						}
					}
					else
					{
						my_nr_picked = MDcoord.numberOfObjects();
					}
				}
				else
				{
					my_nr_picked = MDcoord.numberOfObjects();
				}
			}
			else
			{
				my_nr_picked = 0;
			}
			Fl_Text_Buffer *textbuff2 = new Fl_Text_Buffer();
			textbuff2->text(floatToString(my_nr_picked).c_str());
			count_displays[mymic]->buffer(textbuff2);
			count_displays[mymic]->redraw();
			viewmic_buttons[mymic]->color(GUI_BUTTON_COLOR, GUI_BUTTON_COLOR);
			viewmic_buttons[mymic]->redraw();
		}
	}

	// Save scroll positions before reusing/creating windows
	for (auto &entry : pick_windows)
		mic_scroll_pos[entry.imic] = {entry.scroll->hscrollbar.value(), entry.scroll->scrollbar.value()};

	first_pick_viewed = imic;
	highlight_active(imic);
	last_pick_viewed = XMIPP_MIN((int)global_fn_mics.size() - 1, imic + nr_simultaneous - 1);

	int required = last_pick_viewed - first_pick_viewed + 1;

	// The ones after these load on other threads while these are shown
	preloader().want(first_pick_viewed, last_pick_viewed);
	buildAheadFrom(last_pick_viewed + 1 + global_nr_preload);
	recordLastViewed(first_pick_viewed);

	// Delete excess windows (range shrunk)
	while ((int)pick_windows.size() > required)
	{
		destroyPickWindow(pick_windows.back().win);
		pick_windows.pop_back();
	}

	// Helper: set up color-related canvas properties (same for all code paths)
	auto setupCanvasColors = [](MyPickCanvas *canvas)
	{
		if (global_fn_foms.size() == 0 && global_color_label != "")
		{
			canvas->color_label = EMDL::str2Label(global_color_label);
			canvas->smallest_color_value = XMIPP_MIN(global_blue_value, global_red_value);
			canvas->biggest_color_value = XMIPP_MAX(global_blue_value, global_red_value);
			canvas->do_blue_to_red = (global_blue_value < global_red_value);
			if (global_fn_color != "")
				canvas->fn_color = global_fn_color;
		}
		canvas->minimum_pick_fom = global_minimum_fom;
		canvas->do_read_whole_stacks = false;
	};

	for (int idx = 0; idx < required; idx++)
	{
		int mymic = first_pick_viewed + idx;

		if (idx < (int)pick_windows.size())
		{
			// ---- Reuse existing window ----
			PickWindowEntry &entry = pick_windows[idx];
			entry.imic = mymic;

			std::shared_ptr<PreparedMic> mic = preloader().take(mymic);
			int xsize_canvas = XSIZE(mic->I());
			int ysize_canvas = YSIZE(mic->I());

			// Delete old DisplayBox children of the scroll and clear canvas
			// pointers before creating new DisplayBoxes for the new micrograph.
			// (Before a canvas is replaced below, so that its image goes with it.)
			for (auto *box : entry.canvas->boxes) delete box;
			entry.canvas->boxes.clear();

			// If canvas dimensions changed, replace the canvas widget
			if (entry.canvas->w() != xsize_canvas || entry.canvas->h() != ysize_canvas)
			{
				entry.scroll->remove(entry.canvas);
				delete entry.canvas;

				entry.scroll->begin();
				MyPickCanvas *new_canvas = new MyPickCanvas(0, 0, xsize_canvas, ysize_canvas);
				entry.scroll->end();
				entry.canvas = new_canvas;
			}

			MyPickCanvas *canvas = entry.canvas;

			FileName fn_coord = global_fn_picks[mymic];
			// The canvas expects the radius in displayed (scaled) pixels
			int rad = ROUND(global_micscale * global_particle_diameter/(2. * global_angpix));
			canvas->my_imic = mymic;
			canvas->particle_radius = rad;
			canvas->do_startend = global_pick_startend;
			canvas->startend_style = global_startend_style;
			canvas->do_lines = global_pick_lines;
			canvas->coord_scale = global_coord_scale;
			canvas->SetScroll(entry.scroll);

			// DisplayBox objects must be children of the window, not the scroll
			// or canvas. FLTK draws window children in reverse index order
			// (highest index first), so DisplayBox (highest) draws the image
			// first, then MyPickCanvas draws coordinates on top, then the
			// scroll draws the frame last. Ensure the window is the current
			// Fl_Group so new DisplayBoxes are correctly parented.
			entry.win->begin();
			canvas->fillPrescaled(mic->I(), mic->minval, mic->maxval, global_micscale);
			entry.win->end();

			canvas->fn_coords = fn_coord;
			canvas->fn_mic = global_fn_mics[mymic];
			setupCanvasColors(canvas);

			// Always clear old coordinate data before loading new.
			// Using loadCoordinates here caused stale data issues; clear
			// explicitly and read directly from the correct file path.
			canvas->MDcoords.clear();
			if (exists(fn_coord))
				canvas->MDcoords.read(fn_coord);

			// Replicate do_lines and color handling that loadCoordinates did
			if (canvas->do_lines)
			{
				int ifil_max = 0;
				FOR_ALL_OBJECTS_IN_METADATA_TABLE(canvas->MDcoords)
				{
					int ifil;
					canvas->MDcoords.getValue(EMDL_PARTICLE_SELECTION_TYPE, ifil);
					if (ifil > ifil_max) ifil_max = ifil;
				}
				current_selection_type = ifil_max + 1;
			}
			if (canvas->fn_color != "")
				canvas->findColorColumnForCoordinates();

			canvas->redraw();
			entry.win->label(global_fn_mics[mymic].c_str());

			// Restore saved scroll position for this micrograph
			auto it = mic_scroll_pos.find(mymic);
			if (it != mic_scroll_pos.end())
				entry.scroll->scroll_to(it->second.first, it->second.second);

			if (mymic == first_pick_viewed)
				Fl::focus(canvas);
		}
		else
		{
			// ---- Create new window ----
			std::shared_ptr<PreparedMic> mic = preloader().take(mymic);
			int xsize_canvas = XSIZE(mic->I());
			int ysize_canvas = YSIZE(mic->I());
			FileName fn_coord = global_fn_picks[mymic];

			PickWindow *win = new PickWindow(xsize_canvas + 20, ysize_canvas + 20,
							 global_fn_mics[mymic].c_str());
			Fl_Scroll *scroll = new Fl_Scroll(0, 0, win->w(), win->h());
			int rad = ROUND(global_micscale * global_particle_diameter/(2. * global_angpix));
			MyPickCanvas *canvas = new MyPickCanvas(0, 0, xsize_canvas, ysize_canvas);
			canvas->my_imic = mymic;
			canvas->particle_radius = rad;
			canvas->do_startend = global_pick_startend;
			canvas->startend_style = global_startend_style;
			canvas->do_lines = global_pick_lines;
			canvas->coord_scale = global_coord_scale;
			canvas->SetScroll(scroll);
			canvas->fillPrescaled(mic->I(), mic->minval, mic->maxval, global_micscale);
			canvas->fn_coords = fn_coord;
			canvas->fn_mic = global_fn_mics[mymic];
			setupCanvasColors(canvas);

			if (exists(fn_coord))
			{
				canvas->loadCoordinates(false);
				canvas->redraw();
			}

			win->resizable(*win);
			win->show();

			// Restore saved scroll position for this micrograph
			auto it = mic_scroll_pos.find(mymic);
			if (it != mic_scroll_pos.end())
				scroll->scroll_to(it->second.first, it->second.second);

			if (mymic == first_pick_viewed)
				Fl::focus(canvas);

#ifdef __APPLE__
			Fl::add_timeout(0.0, disable_tabbing_cb, win);
#endif
			pick_windows.push_back({win, scroll, canvas, mymic});
		}
	}

	for (int i = 0; i < (int)viewmic_buttons.size(); i++)
	{
		if (i >= first_pick_viewed && i <= last_pick_viewed)
			viewmic_buttons[i]->color(GUI_BUTTON_DARK_COLOR, GUI_BUTTON_DARK_COLOR);
		else
			viewmic_buttons[i]->color(GUI_BUTTON_COLOR, GUI_BUTTON_COLOR);
		viewmic_buttons[i]->redraw();
	}

	for (Fl_Window *win = Fl::first_window(); win; win = Fl::next_window(win))
		win->cursor(FL_CURSOR_DEFAULT);
}

void cb_viewmic(Fl_Widget* w, void* data)
{
	int *iptr = (int*)data;
	const bool with_control = (Fl::event_ctrl() != 0);
	requestViewMic(*iptr, (with_control) ? global_nr_simultaneous : 1, 0.);
}

void cb_viewctf(Fl_Widget* w, void* data)
{
	// Get my own number back
	int *iptr = (int*)data;
	int imic = *iptr;

	std::string command;
	command =  "relion_display --i " + global_fn_ctfs[imic];
	command += " --scale " + floatToString(global_ctfscale);
	command += " --sigma_contrast " + floatToString(global_ctfsigma);
    command += " &";
	int res = system(command.c_str());

	last_ctf_viewed = imic;
	for (int i = 0; i < viewctf_buttons.size(); i++)
	{
		if (i == last_ctf_viewed)
		{
			viewctf_buttons[i]->color(GUI_BUTTON_DARK_COLOR, GUI_BUTTON_DARK_COLOR);
		}
		else
		{
			viewctf_buttons[i]->color(GUI_BUTTON_COLOR, GUI_BUTTON_COLOR);
		}
	}
}

void cb_selectmic(Fl_Widget* w, void* data)
{
	// Get my own number back
	int *iptr = (int*)data;
	int imic = *iptr;

	Fl_Text_Buffer *textbuff2 = new Fl_Text_Buffer();
	selected[imic] = !selected[imic];
	if (selected[imic])
	{
		text_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
		text_displays[imic]->activate();
		viewmic_buttons[imic]->activate();
		count_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
		textbuff2->text(floatToString(number_picked[imic]).c_str());
		count_displays[imic]->buffer(textbuff2);
		count_displays[imic]->activate();
		if (global_has_ctf)
		{
			viewctf_buttons[imic]->activate();
			defocus_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
			defocus_displays[imic]->activate();
			if (imic < (int)ctfres_displays.size())
			{
				ctfres_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
				ctfres_displays[imic]->activate();
			}
		}
	}
	else
	{
		text_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
		text_displays[imic]->deactivate();
		viewmic_buttons[imic]->deactivate();
		count_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
		textbuff2->text("");
		count_displays[imic]->buffer(textbuff2);
		count_displays[imic]->deactivate();
		if (global_has_ctf)
		{
			viewctf_buttons[imic]->deactivate();
			defocus_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
			defocus_displays[imic]->deactivate();
			if (imic < (int)ctfres_displays.size())
			{
				ctfres_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
				ctfres_displays[imic]->deactivate();
			}
		}
	}
}

int manualpickerGuiWindow::fill()
{
	color(GUI_BACKGROUND_COLOR);

	this->callback(cb_closing);

	Fl_Menu_Bar *menubar = new Fl_Menu_Bar(0, 0, w(), 25);
	if (do_allow_save)
	{
		menubar->add("File/Save selection",  FL_ALT+'s', cb_menubar_save, this);
		menubar->add("File/Invert selection",  FL_ALT+'i', cb_menubar_invert_selection, this);
        menubar->add("File/Select all",  FL_ALT+'a', cb_menubar_select_all, this);
	}
	menubar->add("File/Recount picked particles",  FL_ALT+'c', cb_menubar_recount, this);
	menubar->add("File/Set FOM threshold",  FL_ALT+'c', cb_menubar_setFOM, this);
	menubar->add("File/Quit", FL_ALT+'q', cb_menubar_quit, this);
	int current_y = 25;

	// Scroll bars
	scroll_widget = new Fl_Scroll(0, current_y, w(), h()-current_y);
	scroll_widget->type(Fl_Scroll::VERTICAL);

	selected.clear();
	number_picked.clear();

	global_has_ctf = MDin.containsLabel(EMDL_CTF_IMAGE);

	FileName fn_mic, fn_pick, fn_ctf, fn_fom;
	int ystep = 35;

	imics.clear();
	for (int ii =0; ii < MDin.numberOfObjects(); ii++)
	{
		imics.push_back(ii);
	}

	int imic =0;
	global_fn_mics.clear();
	global_fn_picks.clear();
	global_fn_ctfs.clear();
    global_fn_foms.clear();
	text_displays.clear();
	ctfres_displays.clear();
	viewmic_buttons.clear();
	viewctf_buttons.clear();
	number_picked.clear();
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDin)
	{
		MDin.getValue(EMDL_MICROGRAPH_NAME, fn_mic);
		// Display the name of the micrograph
		global_fn_mics.push_back(fn_mic);

		if (MDin.containsLabel(EMDL_MICROGRAPH_COORDINATES))
		{
			// relion 3.2+
			MDin.getValue(EMDL_MICROGRAPH_COORDINATES, fn_pick);

		}
		else
		{
			//relion 3.1-
			FileName fn_pre, fn_jobnr, fn_post;
			decomposePipelineSymlinkName(fn_mic, fn_pre, fn_jobnr, fn_post);
			fn_pick = global_fn_odir + fn_post.withoutExtension() + "_" + global_pickname + ".star";
		}
		global_fn_picks.push_back(fn_pick);

        if (MDin.containsLabel(EMDL_MICROGRAPH_AUTOPICK_FOM))
        {
            MDin.getValue(EMDL_MICROGRAPH_AUTOPICK_FOM, fn_fom);
            global_fn_foms.push_back(fn_fom);
        }

		Fl_Check_Button *mycheck = new Fl_Check_Button(4, current_y, ystep-8, ystep-8, "");
		mycheck->callback(cb_selectmic, &(imics[imic]));
		mycheck->value(1);
		if (!do_allow_save)
			mycheck->deactivate();
		selected.push_back(true);
		check_buttons.push_back(mycheck);

		Fl_Text_Buffer *textbuff = new Fl_Text_Buffer();
		textbuff->text(fn_mic.c_str());
		int ystep2 = (fn_mic.length() > MWCOL1/12) ? ystep - 5 : ystep - 10;
		Fl_Text_Display* mydisp = new Fl_Text_Display(MXCOL0, current_y, MWCOL1, ystep2);
		mydisp->scrollbar_width(5);
		mydisp->buffer(textbuff);
		mydisp->scroll(0, 9999);
		mydisp->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
		text_displays.push_back(mydisp);

		// Button to display the micrographimage
		Fl_Button *myviewmic = new Fl_Button(MXCOL1, current_y, MWCOL2, ystep-5, "pick");
		myviewmic->color(GUI_BUTTON_COLOR);
		myviewmic->callback(cb_viewmic, &(imics[imic]));
		viewmic_buttons.push_back(myviewmic);

		// Count how many particles have been picked
		Fl_Text_Buffer *textbuff2 = new Fl_Text_Buffer();
		textbuff2->text("");
		Fl_Text_Display* mycount = new Fl_Text_Display(MXCOL2, current_y, MWCOL3, ystep-5);
		mycount->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
		mycount->buffer(textbuff2);
		count_displays.push_back(mycount);
		number_picked.push_back(10);

		// Button to display the CTF image
		if (global_has_ctf)
		{
			MDin.getValue(EMDL_CTF_IMAGE, fn_ctf);
			global_fn_ctfs.push_back(fn_ctf);
			// Button to display the CTF image
			Fl_Button *myviewctf = new Fl_Button(MXCOL3, current_y, MWCOL4, ystep-5, "CTF");
			myviewctf->color(GUI_BUTTON_COLOR);
			myviewctf->callback(cb_viewctf, &(imics[imic]));
			viewctf_buttons.push_back(myviewctf);

			Fl_Text_Buffer *textbuffDF = new Fl_Text_Buffer();
			RFLOAT defocus;
			MDin.getValue(EMDL_CTF_DEFOCUSU, defocus);

			std::ostringstream os;
			os << defocus;
			std::string str = os.str();
			textbuffDF->text(str.c_str());

			Fl_Text_Display* myDF = new Fl_Text_Display(MXCOL4, current_y, MWCOL4, ystep-5);
			myDF->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
			myDF->buffer(textbuffDF);
			defocus_displays.push_back(myDF);

			// CTF resolution
			Fl_Text_Buffer *textbuffCTFR = new Fl_Text_Buffer();
			if (MDin.containsLabel(EMDL_CTF_MAXRES))
			{
				RFLOAT ctf_res;
				MDin.getValue(EMDL_CTF_MAXRES, ctf_res);
				textbuffCTFR->text(floatToString(ctf_res).c_str());
			}
			Fl_Text_Display* myCTFR = new Fl_Text_Display(MXCOL5, current_y, MWCOL5, ystep-5);
			myCTFR->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
			myCTFR->buffer(textbuffCTFR);
			ctfres_displays.push_back(myCTFR);
		}

		imic++;
		current_y += ystep;
	}


	// See if the output STAR file already exists, if so apply that selection
	if (do_allow_save)
		readOutputStarfile();

	if (do_fast_save)
		cb_menubar_save_i();

	// Also count the number of particles that were already picked
	cb_menubar_recount_i();

	// Continue where picking was left off: highlight that micrograph, scroll to it, let X open it,
	// and start loading it and the ones after it
	record_last_viewed = do_allow_save;
	if (text_displays.size() > 0)
	{
		const int resume = resumePoint();
		highlight_active(resume);
		first_pick_viewed = last_pick_viewed = resume - 1;
		if (resume > 0)
		{
			const int max_scroll = XMIPP_MAX(0, (int)text_displays.size() * ystep - scroll_widget->h());
			scroll_widget->scroll_to(0, XMIPP_MIN(max_scroll, XMIPP_MAX(0, (resume - 2) * ystep)));
			std::cout << " Continuing from micrograph " << resume + 1 << ": " << global_fn_mics[resume] << std::endl;
		}
		startLoadingAt(resume);
	}

	// Global keyboard handler for Z/X passthrough from pick windows
#ifdef __APPLE__
	// Disable macOS native tabbing on the main window too
	Fl::add_timeout(0.0, disable_tabbing_cb, this);
#endif

	resizable(*this);
	show();
	return Fl::run();
}

int manualpickerGuiWindow::handle(int event)
{
	if (event == FL_SHORTCUT || event == FL_KEYDOWN)
	{
		const int key = Fl::event_key();
		const int n = (int)viewmic_buttons.size();

		const int base = navigationBase(first_pick_viewed);

		if ((key == 'z' || key == 'Z') && base > 0)
		{
			int imic = base - 1;
			if (imic >= 0 && imic < n)
			{
				requestViewMic(imic, 1, NAV_KEY_DELAY);
				scroll_widget->scroll_to(0, 25 + imic * 35);
			}
			return 1;
		}
		else if ((key == 'x' || key == 'X') && base < n - 1)
		{
			int imic = base + 1;
			if (imic >= 0 && imic < n)
			{
				requestViewMic(imic, 1, NAV_KEY_DELAY);
				scroll_widget->scroll_to(0, 25 + (imic - 2) * 35);
			}
			return 1;
		}
	}
	return Fl_Window::handle(event);
}

void manualpickerGuiWindow::readOutputStarfile()
{
	FileName fn_select = global_fn_odir + "local_selection.star";
    if (exists(fn_select))
	{
        for (int imic = 0; imic < selected.size(); imic++)
			selected[imic] = true;

        MetaDataTable MDselect;
        MDselect.read(fn_select);

        // Set all unselected micrographs to false
        FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDselect)
        {

            FileName fn_mic;
            int is_selected;
            MDselect.getValue(EMDL_MICROGRAPH_NAME, fn_mic);
            MDselect.getValue(EMDL_SELECTED, is_selected);
            if (is_selected == 0)
            {
                // find fn_mic
                for (int imic = 0; imic < selected.size(); imic++)
                {
                    if (global_fn_mics[imic] == fn_mic)
                    {
                        selected[imic] = false;
                        break;
                    }
                }
            }

        }

        // Apply the selection to the buttons
		for (int imic = 0; imic < selected.size(); imic++)
		{
			if (selected[imic])
			{
				check_buttons[imic]->value(1);
				text_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
				text_displays[imic]->activate();
				viewmic_buttons[imic]->activate();
				count_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
				count_displays[imic]->activate();
				if (global_has_ctf)
					viewctf_buttons[imic]->activate();
			}
			else
			{
				check_buttons[imic]->value(0);
				text_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
				text_displays[imic]->deactivate();
				viewmic_buttons[imic]->deactivate();
				count_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
				count_displays[imic]->deactivate();
				if (global_has_ctf)
					viewctf_buttons[imic]->deactivate();
			}
		}
	}
}

void manualpickerGuiWindow::writeOutputStarfiles(bool verb)
{
	if (!do_allow_save) return;

	MDcoords.clear();
    MDcoords.setName("coordinate_files");
    MetaDataTable MDmics, MDselect;
	int c = 0;
	for (int imic = 0; imic < selected.size(); imic++)
	{
		if (selected[imic])
		{
			MDmics.addObject(MDin.getObject(imic));
			FileName fn_coord = global_fn_picks[imic];
			if (exists(fn_coord))
			{
				MDcoords.addObject();
				MDcoords.setValue(EMDL_MICROGRAPH_NAME, global_fn_mics[imic]);
				MDcoords.setValue(EMDL_MICROGRAPH_COORDINATES, fn_coord);
				c++;
			}
        }

        MDselect.addObject();
        MDselect.setValue(EMDL_MICROGRAPH_NAME, global_fn_mics[imic]);
        int sel = (selected[imic]) ? 1 : 0;
        MDselect.setValue(EMDL_SELECTED, sel);

    }

	if (obsModel.opticsMdt.numberOfObjects() > 0)
	{
		obsModel.save(MDmics, fn_sel, "micrographs");
	}
	else
	{
		MDmics.write(fn_sel);
	}
	if (verb) std::cout << " Saved list of selected micrographs in: " << fn_sel << std::endl;

    // Save selection star file for manualpicker continue jobs only
    FileName fn_select = global_fn_odir + "local_selection.star";
    MDselect.write(fn_select);


    std::string picktype("particles");
    if (global_pick_lines) picktype = "lines";
    else if (global_pick_startend) picktype = "startend";
    MetaDataTable MDhead;
    MDhead.setName("general");
    MDhead.setIsList(true);
    MDhead.addObject();
    MDhead.setValue(EMDL_MICROGRAPH_PICKTYPE, picktype);

    std::vector<MetaDataTable> MDins;
    MDins.push_back(MDhead);
    MDins.push_back(MDcoords);
    FileName fn_coords = global_fn_odir + global_pickname + ".star";
    writeMultipleTablesToStar(MDins, fn_coords);

    if (verb) std::cout << " Saved list with " << c << " coordinate files in: " << fn_coords << std::endl;

}
void manualpickerGuiWindow::cb_menubar_save(Fl_Widget* w, void* v)
{
	manualpickerGuiWindow* T=(manualpickerGuiWindow*)v;
	T->cb_menubar_save_i();
}

void manualpickerGuiWindow::cb_menubar_save_i()
{
	writeOutputStarfiles();
	RELION_EXIT_SUCCESS;
}

void manualpickerGuiWindow::cb_menubar_select_all(Fl_Widget* w, void* v)
{
    manualpickerGuiWindow* T=(manualpickerGuiWindow*)v;
    T->cb_menubar_select_all_i();
}

void manualpickerGuiWindow::cb_menubar_select_all_i()
{
	for (int imic = 0; imic < selected.size(); imic++)
    {
        selected[imic] = true;
        check_buttons[imic]->value(1);
        text_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
        text_displays[imic]->activate();
        viewmic_buttons[imic]->activate();
        count_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
        count_displays[imic]->activate();
        if (global_has_ctf)
            viewctf_buttons[imic]->activate();
    }

}

void manualpickerGuiWindow::cb_menubar_invert_selection(Fl_Widget* w, void* v)
{
	manualpickerGuiWindow* T=(manualpickerGuiWindow*)v;
	T->cb_menubar_invert_selection_i();
}

void manualpickerGuiWindow::cb_menubar_invert_selection_i()
{
	for (int imic = 0; imic < selected.size(); imic++)
	{
		selected[imic] = !selected[imic];
		if (selected[imic])
		{
			check_buttons[imic]->value(1);
			text_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
			text_displays[imic]->activate();
			viewmic_buttons[imic]->activate();
			count_displays[imic]->color(GUI_INPUT_COLOR, GUI_INPUT_COLOR);
			count_displays[imic]->activate();
			if (global_has_ctf)
				viewctf_buttons[imic]->activate();
		}
		else
		{
			check_buttons[imic]->value(0);
			text_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
			text_displays[imic]->deactivate();
			viewmic_buttons[imic]->deactivate();
			count_displays[imic]->color(GUI_BACKGROUND_COLOR, GUI_BACKGROUND_COLOR);
			count_displays[imic]->deactivate();
			if (global_has_ctf)
				viewctf_buttons[imic]->deactivate();
		}
	}
}

void manualpickerGuiWindow::cb_menubar_quit(Fl_Widget* w, void* v)
{
	manualpickerGuiWindow* T=(manualpickerGuiWindow*)v;
	T->cb_menubar_quit_i();
}

void manualpickerGuiWindow::cb_menubar_quit_i()
{
	cb_menubar_recount_i();
	for (auto &pw_entry : pick_windows) delete pw_entry.win;
	pick_windows.clear();
	exit(0);
}

void manualpickerGuiWindow::cb_closing(Fl_Widget* w, void* v)
{
	manualpickerGuiWindow* T=(manualpickerGuiWindow*)w;
	T->cb_menubar_quit_i();
}

void manualpickerGuiWindow::cb_menubar_recount(Fl_Widget* w, void* v)
{
	manualpickerGuiWindow* T=(manualpickerGuiWindow*)v;
	T->cb_menubar_recount_i();
}

void manualpickerGuiWindow::cb_menubar_recount_i()
{

	global_total_count = 0;
	int nr_sel_mic = 0;
	for (int imic = 0; imic < global_fn_mics.size(); imic++)
	{
		FileName fn_coord = global_fn_picks[imic];
		MetaDataTable MDcoord;

		int my_nr_picked;
		if (exists(fn_coord))
		{
			MDcoord.read(fn_coord);

			if (fabs(global_minimum_fom + 9999.) > 1e-6)
			{
				if (MDcoord.containsLabel(EMDL_PARTICLE_AUTOPICK_FOM))
				{
					my_nr_picked = 0;
					FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDcoord)
					{
						RFLOAT fom;
						MDcoord.getValue(EMDL_PARTICLE_AUTOPICK_FOM, fom);
						if (fom > global_minimum_fom) my_nr_picked++;
					}
				}
				else
				{
					my_nr_picked = MDcoord.numberOfObjects();
				}

			}
			else
			{
				my_nr_picked = MDcoord.numberOfObjects();
			}
		}
		else
		{
			my_nr_picked = 0;
		}

		Fl_Text_Buffer *textbuff2 = new Fl_Text_Buffer();
		if (selected[imic])
		{
			global_total_count += my_nr_picked;
			textbuff2->text(floatToString(my_nr_picked).c_str());
			count_displays[imic]->buffer(textbuff2);
			count_displays[imic]->redraw();
			nr_sel_mic++;
		}
		else
		{
			textbuff2->text("");
			count_displays[imic]->buffer(textbuff2);
		}
		number_picked[imic] = my_nr_picked;
	}
	std::cout << " Total number of picked particles: " << global_total_count << " from " << nr_sel_mic << " selected micrographs." << std::endl;
	writeOutputStarfiles();
}

void manualpickerGuiWindow::cb_menubar_setFOM(Fl_Widget* w, void* v)
{
	manualpickerGuiWindow* T=(manualpickerGuiWindow*)v;
	T->cb_menubar_setFOM_i();
	T->cb_menubar_recount_i();
}

void manualpickerGuiWindow::cb_menubar_setFOM_i()
{
	const char *pfom;
	std::string currentval = floatToString(global_minimum_fom);
	pfom =  fl_input("Minimum rlnAutopickFigureOfMerit to display: ", currentval.c_str());
	if (pfom == NULL)
		return;
	std::string newval(pfom);
	global_minimum_fom = textToFloat(newval);

}


void ManualPicker::read(int argc, char **argv)
{
	parser.setCommandLine(argc, argv);

	int gen_section = parser.addSection("General options");
	fn_in = parser.getOption("--i", "Micrograph STAR file OR filenames from which to pick particles, e.g. \"Micrographs/*.mrc\"");
	global_fn_odir = parser.getOption("--odir", "Output directory for coordinate files (default is to store next to micrographs)", "ManualPick/");
	fn_sel = parser.getOption("--selection", "STAR file with selected micrographs", "micrographs_selected.star");
	global_pickname = parser.getOption("--pickname", "Rootname for the picked coordinate files", "manualpick");
	global_angpix = textToFloat(parser.getOption("--angpix", "Pixel size in Angstroms", "-1."));
	global_coord_scale = textToFloat(parser.getOption("--coord_scale", "Scale coordinates before display", "1.0"));
	global_particle_diameter = textToFloat(parser.getOption("--particle_diameter", "Diameter of the circles that will be drawn around each picked particle (in Angstroms)"));
	global_pick_startend = parser.checkOption("--pick_start_end", "Pick start-end coordinates of helices");
	global_startend_style = stringToStartEndStyle(parser.getOption("--startend_style", "How to draw start-end picks: box, H-shape or classic (circles joined by a line)", "box"));
    global_pick_lines = parser.checkOption("--pick_lines", "Pick lines for curvy helices");
	do_allow_save = parser.checkOption("--allow_save", "Allow saving of the selected micrographs");
	do_fast_save = parser.checkOption("--fast_save", "Save a default selection of all micrographs immediately");
	sort_micrographs_by = parser.getOption("--sort_micrographs_by", "Sort micrographs: none, ctf_resolution, defocus_high_low, defocus_low_high", "none");
	if (parser.checkOption("--sort_by_ctf_res", ""))
		sort_micrographs_by = "CTF resolution";
	if (parser.checkOption("--sort_by_defocus_high_low", ""))
		sort_micrographs_by = "defocus (high->low)";
	if (parser.checkOption("--sort_by_defocus_low_high", ""))
		sort_micrographs_by = "defocus (low->high)";
	global_nr_simultaneous = textToInteger(parser.getOption("--open_simultaneous", "Open this many of the next micrographs simultaneously when pressing CTRL and a Pick button", "10"));

	int mic_section = parser.addSection("Displaying options");
	global_micscale = textToFloat(parser.getOption("--scale", "Relative scale for the micrograph display", "1"));
	global_black_val = textToFloat(parser.getOption("--black", "Pixel value for black (default is auto-contrast)", "0"));
	global_white_val = textToFloat(parser.getOption("--white", "Pixel value for white (default is auto-contrast)", "0"));
	global_sigma_contrast  = textToFloat(parser.getOption("--sigma_contrast", "Set white and black pixel values this many times the image stddev from the mean (default is auto-contrast)", "0"));
	global_lowpass = textToFloat(parser.getOption("--lowpass", "Lowpass filter in Angstroms for the micrograph (0 for no filtering)","0"));
	global_highpass = textToFloat(parser.getOption("--highpass", "Highpass filter in Angstroms for the micrograph (0 for no filtering)","0"));
	global_do_topaz_denoise = parser.checkOption("--topaz_denoise", "Or instead of filtering, use Topaz denoising before picking (on GPU 0)");
	global_ctfscale = textToFloat(parser.getOption("--ctf_scale", "Relative scale for the CTF-image display", "1"));
	global_ctfsigma = textToFloat(parser.getOption("--ctf_sigma_contrast", "Sigma-contrast for the CTF-image display", "3"));
	global_minimum_fom = textToFloat(parser.getOption("--minimum_pick_fom", "Minimum value for rlnAutopickFigureOfMerit to display picks", "-9999."));
	// coloring
	global_fn_color = parser.getOption("--color_star", "STAR file with a column for red-blue coloring (a subset of) the particles", "");
	global_color_label = parser.getOption("--color_label", "MetaDataLabel to color particles on (e.g. rlnParticleSelectZScore)", "");
	global_blue_value = textToFloat(parser.getOption("--blue", "Value of the blue color", "1."));
	global_red_value = textToFloat(parser.getOption("--red", "Value of the red color", "0."));

	int load_section = parser.addSection("Loading options");
	global_nr_preload = textToInteger(parser.getOption("--preload", "Load this many micrographs after the one viewed in the background, ready to show (0: load each one when it is opened)", "3"));
	global_nr_load_threads = textToInteger(parser.getOption("--load_threads", "Background threads loading those micrographs at once (each builds a virtual micrograph that is not cached yet)", "2"));
	global_nr_vmovie_builders = textToInteger(parser.getOption("--vmovie_builders", "Virtual micrographs built into the cache at once in the background, further ahead than those preloaded (0: none)", "2"));
	global_nr_prebuild = textToInteger(parser.getOption("--prebuild", "At the start, build this many virtual micrographs into the cache in the background, from where picking was left off (0: none)", "0"));
	global_nr_load_threads = XMIPP_MAX(0, global_nr_load_threads);
	global_nr_preload = (global_nr_load_threads > 0) ? XMIPP_MAX(0, global_nr_preload) : 0;
	global_nr_vmovie_builders = XMIPP_MAX(0, global_nr_vmovie_builders);
	global_nr_prebuild = XMIPP_MAX(0, global_nr_prebuild);

	// Check for errors in the command-line option
	if (parser.checkForErrors())
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
}

void ManualPicker::usage()
{
	parser.writeUsage(std::cout);
}

void ManualPicker::initialise()
{
	if (fn_in.isStarFile())
	{
		MetaDataTable MDhead;
        if (MDhead.read(fn_in, "general"))
        {
            std::string picktype;
            MDhead.getValue(EMDL_MICROGRAPH_PICKTYPE, picktype);
            if (picktype == "particles")
            {
                if (global_pick_startend || global_pick_lines)
                    std::cerr << "WARNING: coordinate file states these are particles, ignoring --from_startend or --from_lines "<< std::endl;
                global_pick_startend = global_pick_lines = false;
            }
            else if (picktype == "startend")
            {
                if (global_pick_lines)
                    std::cerr << "WARNING: coordinate file states these are startend, ignoring --from_lines "<< std::endl;
                global_pick_startend = true;
                global_pick_lines = false;
            }
            else if (picktype == "lines")
            {
                if (global_pick_startend)
                    std::cerr << "WARNING: coordinate file states these are lines, ignoring --from_startend "<< std::endl;
                global_pick_startend = false;
                global_pick_lines = true;
            }
            else
            {
                REPORT_ERROR("ERROR: unrecognised micrograph picktype from the general table of the coordinate file");
            }

        }


        // First try 2-column list of coordinate files as in relion-3.2+
		MDin.read(fn_in, "coordinate_files");
		if (MDin.numberOfObjects() > 0)
		{
			if (global_angpix < 0.)
			{
				std::cerr << " WARNING: no --angpix provided and no information about pixel size in input STAR file. Setting angpix to 1..." << std::endl;
				global_angpix = 1.;
			}

		}
		else
		{

			// Normal micrographs.star file (with optics table etc)

			ObservationModel::loadSafely(fn_in, obsModel, MDin, "micrographs");
			if (obsModel.opticsMdt.containsLabel(EMDL_MICROGRAPH_PIXEL_SIZE))
			{
				obsModel.opticsMdt.getValue(EMDL_MICROGRAPH_PIXEL_SIZE, global_angpix, 0);
				std::cout << " Setting angpix to " << global_angpix << " based on the input STAR file... " << std::endl;
			}
			else
			{
				if (global_angpix < 0.)
				{
					REPORT_ERROR("ERROR: the input STAR file does not contain the micrograph pixel size, and it is not given through --angpix.");
				}
				std::cout << " Setting angpix to " << global_angpix << " based on command-line input... " << std::endl;
				FOR_ALL_OBJECTS_IN_METADATA_TABLE(obsModel.opticsMdt)
				{
					obsModel.opticsMdt.setValue(EMDL_MICROGRAPH_PIXEL_SIZE, global_angpix);
				}
			}
		}
	}
	else
	{
		std::vector<FileName> glob_fn_mics;
		fn_in.globFiles(glob_fn_mics);
		for (int imic = 0; imic < glob_fn_mics.size(); imic++)
		{
			MDin.addObject();
			MDin.setValue(EMDL_MICROGRAPH_NAME, glob_fn_mics[imic]);
		}

		if (global_angpix < 0.)
		{
			std::cerr << " WARNING: no --angpix provided and no information about pixel size in input STAR file. Setting angpix to 1..." << std::endl;
			global_angpix = 1.;
		}
	}

	// If we down-scale the micrograph: always low-pass filter to get better displays
	if (global_micscale < 1.)
	{
		RFLOAT new_nyquist = global_angpix * 2. / global_micscale;
		if (new_nyquist > global_lowpass)
			global_lowpass = new_nyquist;
		std::cout << " Set low-pass filter to " << global_lowpass << " due to downscaling of " << global_micscale << std::endl;
	}
}

void ManualPicker::run()
{
	Fl::scheme("gtk+");

	manualpickerGuiWindow win(TOTALWIDTH, TOTALHEIGHT, "RELION manual-picking  [Z:prev  X:next]");

	// Sort micrographs if requested
	if (sort_micrographs_by == "CTF resolution" && MDin.containsLabel(EMDL_CTF_MAXRES))
		MDin.sort(EMDL_CTF_MAXRES);
	else if (sort_micrographs_by == "defocus (high->low)" && MDin.containsLabel(EMDL_CTF_DEFOCUSU))
		MDin.sort(EMDL_CTF_DEFOCUSU, true);
	else if (sort_micrographs_by == "defocus (low->high)" && MDin.containsLabel(EMDL_CTF_DEFOCUSU))
		MDin.sort(EMDL_CTF_DEFOCUSU, false);

	// Pick in green, as relion_display --pick does
	current_selection_type = 2;

	// Transfer all parameters to the gui
	win.MDin = MDin;
	win.MDcoords = MDcoords;
	win.obsModel = obsModel;
	win.fn_sel = fn_sel;
	win.do_allow_save = do_allow_save;
	win.do_fast_save = do_fast_save;
	win.fill();
}
