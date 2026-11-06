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
#include <FL/fl_draw.H>
#include <algorithm>

namespace {

const char *const MAP = "rlnReferenceImage";
const char *const MASK = "rlnSolventMaskName";

int widthOf(const std::string &label)
{
	if (label == MAP) return 170;
	if (label == MASK) return 90;
	if (label == "rlnSymmetryGroup") return 72;
	if (label.find("TubeOuter") != std::string::npos || label.find("TubeInner") != std::string::npos) return 78;
	if (label.find("Min") != std::string::npos || label.find("Max") != std::string::npos) return 74;
	return 56;
}

std::string tooltipOf(const std::string &label)
{
	if (label == MAP) return "The reference map that this class starts from";
	if (label == MASK) return "None (a sphere of the mask diameter), auto (made from the class) or a mask file";
	if (label == "rlnSymmetryGroup") return "Point group of this class (C1 for none)";
	if (label == "rlnHelicalTwist") return "Initial helical twist of this class (deg)";
	if (label == "rlnHelicalRise") return "Initial helical rise of this class (A)";
	if (label == "rlnHelicalTwistMin") return "Smallest helical twist searched for this class (deg)";
	if (label == "rlnHelicalTwistMax") return "Largest helical twist searched for this class (deg)";
	if (label == "rlnHelicalRiseMin") return "Smallest helical rise searched for this class (A)";
	if (label == "rlnHelicalRiseMax") return "Largest helical rise searched for this class (A)";
	if (label == "rlnHelicalMaskTubeInnerDiameter") return "Inner tube diameter of this class (A); negative if not hollow";
	if (label == "rlnHelicalMaskTubeOuterDiameter") return "Outer tube diameter of this class (A)";
	return label;
}

}

std::string ReferenceTable::headingOf(const std::string &label)
{
	if (label == MAP) return "Reference map";
	for (const reflist::TableColumn &c : reflist::tableColumns())
		if (label == c.label) return c.heading;
	return label;
}

ReferenceTable::ReferenceTable(int X, int Y, int W, int H) : Fl_Table(X, Y, W, H)
{
	col_header(1);
	col_header_height(STEPY + 2);
	row_header(1);
	row_header_width(28);
	row_height_all(STEPY + 2);
	col_resize(0);
	row_resize(0);
	color(GUI_BACKGROUND_COLOR);
	end();  // Fl_Table leaves its cell group open
}

void ReferenceTable::setTable(const reflist::ClassTable &table, const std::vector<std::string> &shown)
{
	std::vector<Column> columns;
	for (const std::string &l : shown)
	{
		columns.push_back({l, false, widthOf(l)});
		if (l == MAP || l == MASK)
			columns.push_back({l, true, 24});  // its Browse button
	}
	const bool same = table.maps.size() == table_.maps.size() && columns.size() == columns_.size() &&
		std::equal(columns.begin(), columns.end(), columns_.begin(),
		           [](const Column &a, const Column &b) { return a.label == b.label && a.button == b.button; });
	table_ = table;
	columns_ = columns;
	if (!same)
	{
		rebuild();
		return;
	}
	for (size_t r = 0; r < table_.maps.size(); r++)
		for (const Column &c : columns_)
		{
			Fl_Input *in = c.button ? nullptr : cell((int)r, c.label);
			if (!in) continue;
			const int col = table_.column(c.label);
			if (c.label != MAP && col < 0) continue;
			const std::string &v = c.label == MAP ? table_.maps[r] : table_.cells[r][col];
			if (v != in->value())
				in->value(v.c_str());
		}
}

void ReferenceTable::rebuild()
{
	Fl_Group *current = Fl_Group::current();
	// One of the old cells may be running its callback: delete them once it has returned
	for (Fl_Widget *w : widgets_)
	{
		remove(*w);
		Fl::delete_widget(w);
	}
	widgets_.clear();
	rows((int)table_.maps.size());
	cols((int)columns_.size());
	for (size_t c = 0; c < columns_.size(); c++)
		col_width((int)c, columns_[c].width);
	begin();
	for (size_t r = 0; r < table_.maps.size(); r++)
		for (const Column &c : columns_)
		{
			const int col = (int)widgets_.size() % (int)columns_.size();
			int X, Y, W, H;
			find_cell(CONTEXT_TABLE, (int)r, col, X, Y, W, H);
			if (c.button)
			{
				Fl_Button *b = new Fl_Button(X, Y, W, H, "...");
				b->color(GUI_BUTTON_COLOR);
				b->labelsize(ENTRY_FONTSIZE);
				b->tooltip(c.label == MAP ? "Browse for the map of this class (several maps fill the rows from here on)"
				                          : "Browse for the mask of this class");
				b->callback(cb_button, this);
				widgets_.push_back(b);
			}
			else
			{
				Fl_Input *in = new Fl_Input(X, Y, W, H);
				const int tc = table_.column(c.label);
				in->value(c.label == MAP ? table_.maps[r].c_str() : (tc >= 0 ? table_.cells[r][tc].c_str() : ""));
				in->textsize(ENTRY_FONTSIZE);
				in->color(GUI_INPUT_COLOR);
				in->copy_tooltip(tooltipOf(c.label).c_str());
				in->when(FL_WHEN_CHANGED);
				in->callback(cb_cell, this);
				widgets_.push_back(in);
			}
		}
	end();
	Fl_Group::current(current);
	redraw();
}

void ReferenceTable::placeWidgets()
{
	if (columns_.empty())
		return;
	for (size_t i = 0; i < widgets_.size(); i++)
	{
		int X, Y, W, H;
		find_cell(CONTEXT_TABLE, (int)(i / columns_.size()), (int)(i % columns_.size()), X, Y, W, H);
		widgets_[i]->resize(X, Y, W, H);
	}
	init_sizes();
}

int ReferenceTable::indexOf(const Fl_Widget *w) const
{
	const auto it = std::find(widgets_.begin(), widgets_.end(), w);
	return it == widgets_.end() ? -1 : (int)(it - widgets_.begin());
}

Fl_Input *ReferenceTable::cell(int row, const std::string &label) const
{
	for (size_t c = 0; c < columns_.size(); c++)
		if (columns_[c].label == label && !columns_[c].button && row >= 0 && row < (int)table_.maps.size())
			return static_cast<Fl_Input *>(widgets_[row * columns_.size() + c]);
	return nullptr;
}

Fl_Widget *ReferenceTable::browseButton(int row, bool mask) const
{
	for (size_t c = 0; c < columns_.size(); c++)
		if (columns_[c].button && columns_[c].label == (mask ? MASK : MAP) && row >= 0 && row < (int)table_.maps.size())
			return widgets_[row * columns_.size() + c];
	return nullptr;
}

void ReferenceTable::cb_cell(Fl_Widget *w, void *data)
{
	ReferenceTable *t = static_cast<ReferenceTable *>(data);
	const int i = t->indexOf(w);
	if (i < 0) return;
	const int r = i / (int)t->columns_.size();
	const Column &c = t->columns_[i % t->columns_.size()];
	const std::string v = static_cast<Fl_Input *>(w)->value();
	const int col = t->table_.column(c.label);
	if (c.label == MAP)
		t->table_.maps[r] = v;
	else if (col >= 0)
		t->table_.cells[r][col] = v;
	if (t->on_edit)
		t->on_edit();
}

void ReferenceTable::cb_button(Fl_Widget *w, void *data)
{
	ReferenceTable *t = static_cast<ReferenceTable *>(data);
	const int i = t->indexOf(w);
	if (i < 0 || !t->on_browse) return;
	t->on_browse(i / (int)t->columns_.size(), t->columns_[i % t->columns_.size()].label == MASK);
}

void ReferenceTable::draw_cell(TableContext context, int R, int C, int X, int Y, int W, int H)
{
	switch (context)
	{
	case CONTEXT_STARTPAGE:
		fl_font(FL_HELVETICA, ENTRY_FONTSIZE);
		return;
	case CONTEXT_RC_RESIZE:
		placeWidgets();
		return;
	case CONTEXT_COL_HEADER:
	case CONTEXT_ROW_HEADER:
	{
		std::string text;
		if (context == CONTEXT_ROW_HEADER)
			text = std::to_string(R + 1);
		else if (C < (int)columns_.size() && !columns_[C].button)
			text = headingOf(columns_[C].label);
		fl_push_clip(X, Y, W, H);
		fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, row_header_color());
		fl_color(active_r() ? FL_BLACK : fl_inactive(FL_BLACK));
		fl_font(FL_HELVETICA, ENTRY_FONTSIZE);
		fl_draw(text.c_str(), X + 3, Y, W - 6, H, context == CONTEXT_ROW_HEADER ? FL_ALIGN_CENTER : FL_ALIGN_LEFT);
		fl_pop_clip();
		return;
	}
	default:
		return;
	}
}
