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

#ifndef GUI_CACHE_H_
#define GUI_CACHE_H_

#include <string>
#include <vector>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include "src/cache_manager.h"
#include "gui_table.h"

class CacheManagementWindow;

class CacheTable : public RelionTable
{
    CacheManagementWindow *win_;
protected:
    int numberOfRows() const override;
    int numberOfCols() const override;
    const char* headerText(int col) const override;
    std::string cellText(int row, int col) const override;
    void onDoubleClick(int row) override;
    void onSelectionChanged() override;
    void onSortChanged(int col, bool asc) override;
public:
    CacheTable(int X, int Y, int W, int H, CacheManagementWindow *w);
};

class CacheManagementWindow : public Fl_Window
{
public:
    CacheManagementWindow(int w, int h, const char *title = "Cache Management");

    const std::vector<CacheManager::RegistryEntry>& getDisplayEntries() const { return display_entries_; }
    void updateButtonStates();
    void evictSelected();
    void evictAll();
    void refresh();
    void sortByColumn(int col, bool asc);

private:
    CacheManager cm;
    CacheTable *table;
    Fl_Box *totalLabel;
    Fl_Button *evictBtn;
    Fl_Button *evictAllBtn;
    std::vector<CacheManager::RegistryEntry> display_entries_;

    static void cb_evict_selected(Fl_Widget *, void *v);
    static void cb_evict_all(Fl_Widget *, void *v);
    static void cb_refresh(Fl_Widget *, void *v);
    static void cb_close(Fl_Widget *, void *v);

    void applySort();
};

#endif
