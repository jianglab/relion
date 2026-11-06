/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef GUI_REFERENCE_TABLE_H_
#define GUI_REFERENCE_TABLE_H_

#include <FL/Fl_Table.H>
#include <functional>
#include <string>
#include <vector>
#include "src/reference_list.h"

class Fl_Input;

/* The table of the "One reference per class" mode of Class3D: one row per class with its reference map
 * and settings, typed into the cells. Fixed headings, scroll bars when the rows or columns do not fit.
 * The table holds every column of reflist::tableColumns(); only those given to setTable() are shown.
 */
class ReferenceTable : public Fl_Table
{
public:
	ReferenceTable(int X, int Y, int W, int H);

	/// Show `table`, with the columns `shown` (rlnReferenceImage for the maps). The cells are made anew
	/// only when the rows or the columns change; otherwise their texts are updated.
	void setTable(const reflist::ClassTable &table, const std::vector<std::string> &shown);
	/// The table, with what has been typed
	const reflist::ClassTable &table() const { return table_; }

	/// The input of a cell, or nullptr if its column is not shown
	Fl_Input *cell(int row, const std::string &label) const;
	/// The Browse button of a map or mask, or nullptr
	Fl_Widget *browseButton(int row, bool mask) const;

	std::function<void()> on_edit;                      ///< a cell was typed in
	std::function<void(int row, bool mask)> on_browse;  ///< the Browse button of a map (mask) was pressed

	/// The heading of a column
	static std::string headingOf(const std::string &label);

protected:
	void draw_cell(TableContext context, int R, int C, int X, int Y, int W, int H) override;

private:
	struct Column { std::string label; bool button; int width; };
	reflist::ClassTable table_;
	std::vector<Column> columns_;
	std::vector<Fl_Widget *> widgets_;  ///< row by row, one per column

	void rebuild();
	void placeWidgets();
	int indexOf(const Fl_Widget *w) const;
	static void cb_cell(Fl_Widget *, void *);
	static void cb_button(Fl_Widget *, void *);
};

#endif
