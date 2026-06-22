/***************************************************************************
 *
 * Author: "Wen Jiang"
 * Pennsylvania State University
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

#ifndef GUI_TABLE_H_
#define GUI_TABLE_H_

#include <string>
#include <FL/Fl_Table_Row.H>

// ---------------------------------------------------------------------------
// RelionTable — reusable base class for Fl_Table_Row tables
//
// Provides common visual patterns used by both the project management table
// and cache management table:
//   - Zebra-stripe row backgrounds
//   - Selection highlighting
//   - Grid lines
//   - Column-header drawing with sort indicators (▲/▼)
//   - Double-click detection
//   - Column-header-click sort toggle
//
// Subclasses override virtual methods to provide data.
// The owning window stores the actual data and handles row-level actions.
// ---------------------------------------------------------------------------

class RelionTable : public Fl_Table_Row
{
public:
    RelionTable(int X, int Y, int W, int H);

    // --- Data access (must override) ---
    virtual int numberOfRows() const = 0;
    virtual int numberOfCols() const = 0;
    virtual const char* headerText(int col) const = 0;
    virtual std::string cellText(int row, int col) const = 0;

    // --- Optional customization ---
    virtual Fl_Color cellBackgroundColor(int row, int col);
    virtual Fl_Color cellTextColor(int row, int col);
    virtual bool isCurrentRow(int row) const;

    // --- Events (override for row-level actions) ---
    virtual void onDoubleClick(int row);
    virtual void onSelectionChanged();
    virtual void onSortChanged(int col, bool asc);

    // --- Sorting ---
    int sortCol() const { return sort_col_; }
    bool sortAsc() const { return sort_asc_; }
    void setSort(int col, bool asc);
    void toggleSort(int col);

    // Call after data changes to sync row count and redraw
    void syncRowCount();

protected:
    void draw_cell(TableContext context, int R, int C,
                   int X, int Y, int W, int H) override;
    int handle(int event) override;

    int sort_col_ = -1;
    bool sort_asc_ = true;
};

#endif
