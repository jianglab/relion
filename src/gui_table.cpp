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

#include "gui_table.h"
#include <FL/Fl.H>
#include <FL/fl_draw.H>

RelionTable::RelionTable(int X, int Y, int W, int H)
    : Fl_Table_Row(X, Y, W, H)
{
}

void RelionTable::draw_cell(TableContext context, int R, int C,
                            int X, int Y, int W, int H)
{
    switch (context)
    {
    case CONTEXT_COL_HEADER:
    {
        fl_push_clip(X, Y, W, H);
        fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H,
                    fl_color_average(FL_BACKGROUND_COLOR, FL_WHITE, 0.3f));
        fl_color(FL_BLACK);
        fl_font(FL_HELVETICA_BOLD, 12);

        int ncols = numberOfCols();
        if (C >= 0 && C < ncols)
        {
            std::string label = headerText(C);
            fl_draw(label.c_str(), X + 5, Y, W - 10, H,
                    FL_ALIGN_LEFT | FL_ALIGN_CENTER);

            if (C == sort_col_)
            {
                int cx = X + 5 + (int)fl_width(label.c_str()) + 6;
                int cy = Y + H / 2;
                int s = 4;
                fl_color(FL_BLACK);
                fl_begin_polygon();
                if (sort_asc_)
                {
                    fl_vertex(cx, cy - s);
                    fl_vertex(cx - s, cy + s);
                    fl_vertex(cx + s, cy + s);
                }
                else
                {
                    fl_vertex(cx, cy + s);
                    fl_vertex(cx - s, cy - s);
                    fl_vertex(cx + s, cy - s);
                }
                fl_end_polygon();
            }
        }
        fl_pop_clip();
        return;
    }

    case CONTEXT_CELL:
    {
        int nrows = numberOfRows();
        int ncols = numberOfCols();
        if (R < 0 || R >= nrows || C < 0 || C >= ncols)
            return;

        fl_push_clip(X, Y, W, H);

        // Background
        Fl_Color bg = cellBackgroundColor(R, C);
        fl_rectf(X, Y, W, H, bg);

        // Grid lines
        fl_color(fl_darker(FL_BACKGROUND_COLOR));
        fl_xyline(X, Y + H - 1, X + W - 1);
        fl_xyline(X + W - 1, Y, X + W - 1, Y + H - 1);

        // Text
        fl_color(cellTextColor(R, C));
        fl_font(FL_HELVETICA, 12);
        std::string text = cellText(R, C);
        fl_draw(text.c_str(), X + 5, Y, W - 10, H,
                FL_ALIGN_LEFT | FL_ALIGN_CENTER);

        fl_pop_clip();
        return;
    }

    default:
        return;
    }
}

int RelionTable::handle(int event)
{
    int ret = Fl_Table_Row::handle(event);

    if (event == FL_PUSH && Fl::event_clicks() > 0)
    {
        int r = callback_row();
        if (r >= 0 && r < numberOfRows())
            onDoubleClick(r);
    }
    else if (event == FL_RELEASE)
    {
        // Check if click was on a column header
        TableContext ctx = callback_context();
        if (ctx == CONTEXT_COL_HEADER)
        {
            int c = callback_col();
            if (c >= 0 && c < numberOfCols())
                toggleSort(c);
        }
        onSelectionChanged();
    }
    return ret;
}

void RelionTable::setSort(int col, bool asc)
{
    sort_col_ = col;
    sort_asc_ = asc;
}

void RelionTable::toggleSort(int col)
{
    if (col < 0 || col >= numberOfCols())
        return;
    if (sort_col_ == col)
        sort_asc_ = !sort_asc_;
    else
    {
        sort_col_ = col;
        sort_asc_ = true;
    }
    onSortChanged(sort_col_, sort_asc_);
}

void RelionTable::syncRowCount()
{
    rows(numberOfRows());
    redraw();
}

// ---------------------------------------------------------------------------
// Default color helpers (zebra stripes + selection)
// ---------------------------------------------------------------------------

Fl_Color RelionTable::cellBackgroundColor(int row, int col)
{
    if (row_selected(row))
        return FL_SELECTION_COLOR;
    if (row % 2)
        return fl_color_average(FL_WHITE, FL_BACKGROUND_COLOR, 0.85f);
    return FL_WHITE;
}

Fl_Color RelionTable::cellTextColor(int row, int col)
{
    return row_selected(row) ? FL_WHITE : FL_BLACK;
}

bool RelionTable::isCurrentRow(int row) const
{
    return false;
}

void RelionTable::onDoubleClick(int row)
{
}

void RelionTable::onSelectionChanged()
{
}

void RelionTable::onSortChanged(int col, bool asc)
{
    redraw();
}
