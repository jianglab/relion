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

/* The table of the "One reference per class" mode of Class3D: one column per class, one row per setting
 * (its reference map, symmetry, mask, ...), typed into the cells. The map and mask cells have a Browse
 * button. Fixed headings, scroll bars (and the mouse wheel) when the classes or settings do not fit.
 * The table holds every column of reflist::tableColumns(); only those given to setTable() are shown.
 */
class ReferenceTable : public Fl_Table
{
public:
	ReferenceTable(int X, int Y, int W, int H);

	/// Show `table`, with the settings `shown` (rlnReferenceImage for the maps), one row each. The cells are
	/// made anew only when the classes or the settings change; otherwise their texts are updated.
	void setTable(const reflist::ClassTable &table, const std::vector<std::string> &shown);
	/// The table, with what has been typed
	const reflist::ClassTable &table() const { return table_; }
	/// The number of classes (columns)
	int classes() const { return (int)table_.maps.size(); }
	/// The settings shown (rows), as STAR labels
	std::vector<std::string> shownLabels() const;

	/// The input of class `cls` for setting `label`, or nullptr if that setting is not shown
	Fl_Input *cell(int cls, const std::string &label) const;
	/// The Browse button of the map (mask) of class `cls`, or nullptr
	Fl_Widget *browseButton(int cls, bool mask) const;

	/// The height that shows every setting without a vertical scroll bar
	int fullHeight();

	std::function<void()> on_edit;                      ///< a cell was typed in
	std::function<void(int cls, bool mask)> on_browse;  ///< the Browse button of a map (mask) was pressed

	/// The heading of a setting
	static std::string headingOf(const std::string &label);

	int handle(int event) override;

protected:
	void draw_cell(TableContext context, int R, int C, int X, int Y, int W, int H) override;

private:
	struct Row { std::string label; bool button; };
	reflist::ClassTable table_;
	std::vector<Row> rows_;
	std::vector<Fl_Input *> inputs_;    ///< setting by setting, one per class
	std::vector<Fl_Widget *> buttons_;  ///< as inputs_; nullptr for settings without a Browse button

	void rebuild();
	void placeWidgets();
	int indexOf(const Fl_Widget *w, bool button) const;
	static void cb_cell(Fl_Widget *, void *);
	static void cb_button(Fl_Widget *, void *);
};

#endif
