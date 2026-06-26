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
#include "src/fftw.h"
#include <signal.h>
#include <cstdio>
#include <FL/x.H>
#ifdef __APPLE__
#include <objc/runtime.h>
#include <objc/message.h>
#endif

struct PickWindowEntry {
	Fl_Window *win;
	Fl_Scroll *scroll;
	int imic;
};
std::vector<PickWindowEntry> pick_windows;
std::map<int, std::pair<int,int>> mic_scroll_pos;

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
	int handle(int ev)
	{
		if (ev == FL_KEYDOWN || ev == FL_SHORTCUT)
		{
			int key = Fl::event_key();
			if (my_imic >= 0 && (key == 'z' || key == 'Z'))
			{
				int imic = my_imic - 1;
				if (imic >= 0 && imic < (int)viewmic_buttons.size())
				{
					viewmic_buttons[imic]->do_callback();
					return 1;
				}
			}
			if (my_imic >= 0 && (key == 'x' || key == 'X'))
			{
				int imic = my_imic + 1;
				if (imic >= 0 && imic < (int)viewmic_buttons.size())
				{
					viewmic_buttons[imic]->do_callback();
					return 1;
				}
			}
		}
		return pickerViewerCanvas::handle(ev);
	}
};

void cb_viewmic(Fl_Widget* w, void* data)
{
	int *iptr = (int*)data;
	int imic = *iptr;

	const bool with_control = (Fl::event_ctrl() != 0);
	int nr_simultaneous = (with_control) ? global_nr_simultaneous : 1;

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
	// Save scroll positions and window position before closing
	static int prev_win_x = -1, prev_win_y = -1;
	for (auto &entry : pick_windows)
	{
		mic_scroll_pos[entry.imic] = {entry.scroll->hscrollbar.value(), entry.scroll->scrollbar.value()};
		prev_win_x = entry.win->x();
		prev_win_y = entry.win->y();
	}

	for (auto &entry : pick_windows) delete entry.win;
	pick_windows.clear();

	first_pick_viewed = imic;
	highlight_active(imic);
	last_pick_viewed = XMIPP_MIN(global_fn_mics.size() - 1, imic + nr_simultaneous - 1);
	for (int mymic = first_pick_viewed; mymic <= last_pick_viewed; mymic++)
	{
		FileName fn_coord = global_fn_picks[mymic];

		Image<RFLOAT> img;
		img.read(global_fn_mics[mymic]);

		if (global_lowpass > 0.)
			lowPassFilterMap(img(), global_lowpass, global_angpix);
		if (global_highpass > 0.)
			highPassFilterMap(img(), global_highpass, global_angpix, 25);

		int xsize_canvas = CEIL(XSIZE(img()) * global_micscale);
		int ysize_canvas = CEIL(YSIZE(img()) * global_micscale);

		Fl_Double_Window *win = new Fl_Double_Window(xsize_canvas + 20, ysize_canvas + 20, global_fn_mics[mymic].c_str());
		Fl_Scroll *scroll = new Fl_Scroll(0, 0, win->w(), win->h());
		int rad = ROUND(global_particle_diameter/(2. * global_angpix));
		MyPickCanvas *canvas = new MyPickCanvas(0, 0, xsize_canvas, ysize_canvas);
		canvas->my_imic = mymic;
		canvas->particle_radius = rad;
		canvas->do_startend = global_pick_startend;
		canvas->do_lines = global_pick_lines;
		canvas->coord_scale = global_coord_scale;
		canvas->SetScroll(scroll);
		canvas->fill(img(), global_black_val, global_white_val, global_sigma_contrast, global_micscale);
		canvas->fn_coords = fn_coord;
		canvas->fn_mic = global_fn_mics[mymic];
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
		if (exists(fn_coord))
		{
			canvas->loadCoordinates(false);
			canvas->redraw();
		}

		win->resizable(*win);
		win->show();

		// Restore saved window position (after show on macOS)
		if (prev_win_x >= 0)
			win->position(prev_win_x, prev_win_y);

		// Restore saved scroll position for this micrograph
		auto it = mic_scroll_pos.find(mymic);
		if (it != mic_scroll_pos.end())
			scroll->scroll_to(it->second.first, it->second.second);

		// Give focus to the canvas so keyboard Z/X works from the pick window
		if (mymic == first_pick_viewed)
			Fl::focus(canvas);

#ifdef __APPLE__
		Fl::add_timeout(0.0, disable_tabbing_cb, win);
#endif
		pick_windows.push_back({win, scroll, mymic});
	}

	for (int i = 0; i < viewmic_buttons.size(); i++)
	{
		if (i >= first_pick_viewed && i <= last_pick_viewed)
			viewmic_buttons[i]->color(GUI_BUTTON_DARK_COLOR, GUI_BUTTON_DARK_COLOR);
		else
			viewmic_buttons[i]->color(GUI_BUTTON_COLOR, GUI_BUTTON_COLOR);
		viewmic_buttons[i]->redraw();
	}
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

	// Highlight the first micrograph as initially active
	if (text_displays.size() > 0)
		highlight_active(0);

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

		if ((key == 'z' || key == 'Z') && first_pick_viewed > 0)
		{
			int imic = first_pick_viewed - 1;
			if (imic >= 0 && imic < n)
			{
				viewmic_buttons[imic]->do_callback();
				scroll_widget->scroll_to(0, 25 + imic * 35);
			}
			return 1;
		}
		else if ((key == 'x' || key == 'X') && first_pick_viewed < n - 1)
		{
			int imic = first_pick_viewed + 1;
			if (imic >= 0 && imic < n)
			{
				viewmic_buttons[imic]->do_callback();
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

	// Transfer all parameters to the gui
	win.MDin = MDin;
	win.MDcoords = MDcoords;
	win.obsModel = obsModel;
	win.fn_sel = fn_sel;
	win.do_allow_save = do_allow_save;
	win.do_fast_save = do_fast_save;
	win.fill();
}
