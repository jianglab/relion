/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "src/gui_reference_table.h"
#include "src/gui_entries.h"

#include <FL/Fl.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Scrollbar.H>
#include <FL/fl_draw.H>
#include <algorithm>
#include <cmath>

namespace {

const char *const MAP = "rlnReferenceImage";
const char *const MASK = "rlnSolventMaskName";

const int CLASS_WIDTH = 132;   // one class
const int HEADING_WIDTH = 84;  // the names of the settings
const int BUTTON_WIDTH = 22;   // Browse, inside the map and mask cells

std::string tooltipOf(const std::string &label)
{
	if (label == MAP) return "The reference map that this class starts from";
	if (label == MASK) return "None (a sphere of the mask diameter), auto (made from the class) or a mask file";
	if (label == "rlnSymmetryGroup") return "Point group of this class (C1 for none)";
	if (label == "rlnHelicalTwist") return "Initial helical twist of this class (deg)";
	if (label == "rlnHelicalRise") return "Initial helical rise of this class (A)";
	if (label == "rlnHelicalTwistMin") return "Smallest helical twist searched for this class (deg)";
	if (label == "rlnHelicalTwistMax") return "Largest helical twist searched for this class (deg)";
	if (label == "rlnHelicalTwistInitialStep") return "Initial step of the helical twist search of this class (deg); 0: chosen by RELION";
	if (label == "rlnHelicalRiseMin") return "Smallest helical rise searched for this class (A)";
	if (label == "rlnHelicalRiseMax") return "Largest helical rise searched for this class (A)";
	if (label == "rlnHelicalRiseInitialStep") return "Initial step of the helical rise search of this class (A); 0: chosen by RELION";
	if (label == "rlnHelicalMaskTubeInnerDiameter") return "Inner tube diameter of this class (A); negative if not hollow";
	if (label == "rlnHelicalMaskTubeOuterDiameter") return "Outer tube diameter of this class (A)";
	return label;
}

}

std::string ReferenceTable::headingOf(const std::string &label)
{
	if (label == MAP) return "Map";
	for (const reflist::TableColumn &c : reflist::tableColumns())
		if (label == c.label) return c.heading;
	return label;
}

ReferenceTable::ReferenceTable(int X, int Y, int W, int H) : Fl_Table(X, Y, W, H)
{
	col_header(1);
	col_header_height(STEPY + 2);
	row_header(1);
	row_header_width(HEADING_WIDTH);
	col_resize(0);
	row_resize(0);
	color(GUI_BACKGROUND_COLOR);
	end();  // Fl_Table leaves its cell group open
}

std::vector<std::string> ReferenceTable::shownLabels() const
{
	std::vector<std::string> labels;
	for (const Row &r : rows_)
		labels.push_back(r.label);
	return labels;
}

int ReferenceTable::fullHeight()
{
	// the headings, every setting, and room for the horizontal scroll bar
	return col_header_height() + (int)rows_.size() * (STEPY + 2) + Fl::scrollbar_size() + Fl::box_dh(box());
}

void ReferenceTable::setTable(const reflist::ClassTable &table, const std::vector<std::string> &shown)
{
	std::vector<Row> rows;
	for (const std::string &l : shown)
		rows.push_back({l, l == MAP || l == MASK});
	const bool same = table.maps.size() == table_.maps.size() && rows.size() == rows_.size() &&
		std::equal(rows.begin(), rows.end(), rows_.begin(), [](const Row &a, const Row &b) { return a.label == b.label; });
	table_ = table;
	rows_ = rows;
	if (!same)
	{
		rebuild();
		return;
	}
	for (int k = 0; k < classes(); k++)
		for (const Row &r : rows_)
		{
			Fl_Input *in = cell(k, r.label);
			const int col = table_.column(r.label);
			if (!in || (r.label != MAP && col < 0)) continue;
			const std::string &v = r.label == MAP ? table_.maps[k] : table_.cells[k][col];
			if (v != in->value())
				in->value(v.c_str());
		}
}

void ReferenceTable::rebuild()
{
	Fl_Group *current = Fl_Group::current();
	// One of the old cells may be running its callback: delete them once it has returned
	std::vector<Fl_Widget *> old(inputs_.begin(), inputs_.end());
	old.insert(old.end(), buttons_.begin(), buttons_.end());
	for (Fl_Widget *w : old)
		if (w)
		{
			remove(*w);
			Fl::delete_widget(w);
		}
	inputs_.clear();
	buttons_.clear();
	rows((int)rows_.size());
	cols(classes());
	row_height_all(STEPY + 2);
	col_width_all(CLASS_WIDTH);
	begin();
	for (const Row &r : rows_)
		for (int k = 0; k < classes(); k++)
		{
			Fl_Input *in = new Fl_Input(0, 0, 10, 10);
			const int tc = table_.column(r.label);
			in->value(r.label == MAP ? table_.maps[k].c_str() : (tc >= 0 ? table_.cells[k][tc].c_str() : ""));
			in->textsize(ENTRY_FONTSIZE);
			in->color(GUI_INPUT_COLOR);
			in->copy_tooltip(tooltipOf(r.label).c_str());
			in->when(FL_WHEN_CHANGED);
			in->callback(cb_cell, this);
			inputs_.push_back(in);
			Fl_Button *b = nullptr;
			if (r.button)
			{
				b = new Fl_Button(0, 0, 10, 10, "...");
				b->color(GUI_BUTTON_COLOR);
				b->labelsize(ENTRY_FONTSIZE);
				b->tooltip(r.label == MAP ? "Browse for the map of this class (several maps fill this class and the next ones)"
				                          : "Browse for the mask of this class");
				b->callback(cb_button, this);
			}
			buttons_.push_back(b);
		}
	end();
	Fl_Group::current(current);
	placeWidgets();
	redraw();
}

void ReferenceTable::placeWidgets()
{
	const int n = classes();
	if (n == 0)
		return;
	for (size_t i = 0; i < inputs_.size(); i++)
	{
		int X, Y, W, H;
		find_cell(CONTEXT_TABLE, (int)(i / n), (int)(i % n), X, Y, W, H);
		if (buttons_[i])
		{
			inputs_[i]->resize(X, Y, W - BUTTON_WIDTH, H);
			buttons_[i]->resize(X + W - BUTTON_WIDTH, Y, BUTTON_WIDTH, H);
		}
		else
			inputs_[i]->resize(X, Y, W, H);
	}
	init_sizes();
}

int ReferenceTable::indexOf(const Fl_Widget *w, bool button) const
{
	if (button)
	{
		const auto it = std::find(buttons_.begin(), buttons_.end(), w);
		return it == buttons_.end() ? -1 : (int)(it - buttons_.begin());
	}
	const auto it = std::find(inputs_.begin(), inputs_.end(), w);
	return it == inputs_.end() ? -1 : (int)(it - inputs_.begin());
}

Fl_Input *ReferenceTable::cell(int cls, const std::string &label) const
{
	if (cls < 0 || cls >= classes())
		return nullptr;
	for (size_t r = 0; r < rows_.size(); r++)
		if (rows_[r].label == label)
			return inputs_[r * classes() + cls];
	return nullptr;
}

Fl_Widget *ReferenceTable::browseButton(int cls, bool mask) const
{
	if (cls < 0 || cls >= classes())
		return nullptr;
	for (size_t r = 0; r < rows_.size(); r++)
		if (rows_[r].label == (mask ? MASK : MAP))
			return buttons_[r * classes() + cls];
	return nullptr;
}

void ReferenceTable::cb_cell(Fl_Widget *w, void *data)
{
	ReferenceTable *t = static_cast<ReferenceTable *>(data);
	const int i = t->indexOf(w, false);
	if (i < 0) return;
	const int k = i % t->classes();
	const Row &r = t->rows_[i / t->classes()];
	const std::string v = static_cast<Fl_Input *>(w)->value();
	const int col = t->table_.column(r.label);
	if (r.label == MAP)
		t->table_.maps[k] = v;
	else if (col >= 0)
		t->table_.cells[k][col] = v;
	if (t->on_edit)
		t->on_edit();
}

void ReferenceTable::cb_button(Fl_Widget *w, void *data)
{
	ReferenceTable *t = static_cast<ReferenceTable *>(data);
	const int i = t->indexOf(w, true);
	if (i < 0 || !t->on_browse) return;
	t->on_browse(i % t->classes(), t->rows_[i / t->classes()].label == MASK);
}

int ReferenceTable::handle(int event)
{
	// The mouse wheel scrolls the settings (up and down) while some are hidden, else the classes (sideways)
	if (event == FL_MOUSEWHEEL && Fl::event_inside(this) && active_r())
	{
		Fl_Scrollbar *bar = nullptr;
		int ticks = 0, pixels = 0;
		if (Fl::event_dy() != 0 && vscrollbar->visible())
			bar = vscrollbar, ticks = Fl::event_dy(), pixels = 2 * (STEPY + 2);
		else if (hscrollbar->visible())
			bar = hscrollbar, ticks = Fl::event_dx() != 0 ? Fl::event_dx() : Fl::event_dy(), pixels = CLASS_WIDTH;
		if (bar && ticks != 0)
		{
			const double v = bar->clamp(bar->value() + ticks * pixels);
			if ((int)std::lround(v) != bar->value())
			{
				bar->value((int)std::lround(v));
				bar->do_callback();
			}
			return 1;
		}
	}
	return Fl_Table::handle(event);
}

void ReferenceTable::draw_cell(TableContext context, int R, int C, int X, int Y, int W, int H)
{
	switch (context)
	{
	case CONTEXT_STARTPAGE:
		fl_font(FL_HELVETICA, ENTRY_FONTSIZE);
		return;
	case CONTEXT_RC_RESIZE:  // (also when scrolled)
		placeWidgets();
		return;
	case CONTEXT_COL_HEADER:
	case CONTEXT_ROW_HEADER:
	{
		std::string text;
		if (context == CONTEXT_COL_HEADER)
			text = "Class " + std::to_string(C + 1);
		else if (R < (int)rows_.size())
			text = headingOf(rows_[R].label);
		fl_push_clip(X, Y, W, H);
		fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, row_header_color());
		fl_color(active_r() ? FL_BLACK : fl_inactive(FL_BLACK));
		fl_font(FL_HELVETICA, ENTRY_FONTSIZE);
		fl_draw(text.c_str(), X + 4, Y, W - 8, H, context == CONTEXT_COL_HEADER ? FL_ALIGN_CENTER : FL_ALIGN_LEFT);
		fl_pop_clip();
		return;
	}
	default:
		return;
	}
}
