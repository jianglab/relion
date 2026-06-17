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
#include <FL/Fl_Browser.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include "src/cache_manager.h"

class CacheManagementWindow : public Fl_Window
{
public:
    CacheManagementWindow(int w, int h, const char *title = "Cache Management");

private:
    CacheManager cm;
    Fl_Browser *browser;
    Fl_Box *totalLabel;
    std::vector<CacheManager::RegistryEntry> displayEntries;

    void refresh();
    void evictSelected();
    void evictAll();

    static void cb_refresh(Fl_Widget *, void *v);
    static void cb_evict_selected(Fl_Widget *, void *v);
    static void cb_evict_all(Fl_Widget *, void *v);
    static void cb_close(Fl_Widget *, void *v);
};

#endif
