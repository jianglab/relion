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

#include "gui_cache.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <FL/fl_ask.H>
#include <FL/Fl.H>
#include <FL/fl_draw.H>

namespace {

static std::string formatSize(long bytes)
{
    char buf[32];
    if (bytes < 1024)
        std::snprintf(buf, sizeof(buf), "%ld B", bytes);
    else if (bytes < 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024LL * 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024.0));
    else
        std::snprintf(buf, sizeof(buf), "%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    return std::string(buf);
}

static std::string timestampStr(time_t t)
{
    if (t == 0) return "-";
    char buf[32];
    struct tm *tm = std::localtime(&t);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", tm);
    return std::string(buf);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// CacheTable
// ---------------------------------------------------------------------------

CacheTable::CacheTable(int X, int Y, int W, int H, CacheManagementWindow *w)
    : RelionTable(X, Y, W, H), win_(w)
{
}

int CacheTable::numberOfRows() const
{
    if (!win_) return 0;
    return (int)win_->getDisplayEntries().size();
}

int CacheTable::numberOfCols() const
{
    return 6;
}

const char* CacheTable::headerText(int col) const
{
    static const char *headers[] = {
        "Source Job Dir", "Size", "Node", "Path", "Created", "Last Access"
    };
    if (col >= 0 && col < 6)
        return headers[col];
    return "";
}

std::string CacheTable::cellText(int row, int col) const
{
    if (!win_) return "";
    const auto &entries = win_->getDisplayEntries();
    if (row < 0 || row >= (int)entries.size()) return "";

    const auto &e = entries[row];
    switch (col)
    {
    case 0:
    {
        std::string src = e.sourceJobDir;
        if (src.empty())
            src = e.key.substr(0, 16);
        return src;
    }
    case 1: return formatSize(e.sizeBytes);
    case 2:
    {
        std::string node = e.nodeName;
        return node.empty() ? "(unknown)" : node;
    }
    case 3: return e.cachePath;
    case 4: return timestampStr(e.created);
    case 5: return timestampStr(e.lastAccess);
    default: return "";
    }
}

void CacheTable::onDoubleClick(int row)
{
    // no-op for now; could open the cache directory in the future
}

void CacheTable::onSelectionChanged()
{
    if (win_)
        win_->updateButtonStates();
}

void CacheTable::onSortChanged(int col, bool asc)
{
    if (win_)
        win_->sortByColumn(col, asc);
}

// ---------------------------------------------------------------------------
// CacheManagementWindow
// ---------------------------------------------------------------------------

CacheManagementWindow::CacheManagementWindow(int w, int h, const char *title)
    : Fl_Window(w, h, title)
{
    int margin = 10;
    int labelH = 30;
    int btnH = 30;
    int btnW = 120;

    int table_x = margin;
    int table_y = margin + labelH + 5;
    int table_w = w - 2 * margin;
    int table_h = h - margin * 3 - labelH - btnH - 15;

    totalLabel = new Fl_Box(margin, margin, w - margin * 2, labelH, "Loading...");
    totalLabel->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    totalLabel->labelsize(14);
    totalLabel->labelfont(FL_BOLD);

    table = new CacheTable(table_x, table_y, table_w, table_h, this);
    table->type(Fl_Table_Row::SELECT_MULTI);
    table->col_header(true);
    table->col_header_height(22);
    table->row_height_all(24);
    table->col_resize(true);
    table->col_resize_min(60);
    table->cols(6);

    // Fl_Table constructor leaves current group set to its internal Fl_Scroll
    begin();

    int avail_w = table_w - 16;
    table->col_width(0, avail_w * 20 / 100);
    table->col_width(1, avail_w * 10 / 100);
    table->col_width(2, avail_w * 15 / 100);
    table->col_width(3, avail_w * 25 / 100);
    table->col_width(4, avail_w * 15 / 100);
    table->col_width(5, avail_w * 15 / 100);

    int btnY = h - margin - btnH;

    evictBtn = new Fl_Button(margin, btnY, btnW, btnH, "Evict selected");
    evictBtn->callback(cb_evict_selected, this);
    evictBtn->deactivate();

    evictAllBtn = new Fl_Button(margin + btnW + 10, btnY, btnW, btnH, "Evict all");
    evictAllBtn->callback(cb_evict_all, this);

    Fl_Button *refreshBtn = new Fl_Button(margin + (btnW + 10) * 2, btnY, btnW, btnH, "Refresh");
    refreshBtn->callback(cb_refresh, this);

    Fl_Button *closeBtn = new Fl_Button(w - margin - btnW, btnY, btnW, btnH, "Close");
    closeBtn->callback(cb_close, this);

    end();
    resizable(table);

    cm.setRegistryDir("~");
    refresh();
}

void CacheManagementWindow::refresh()
{
    display_entries_.clear();

    std::vector<CacheManager::RegistryEntry> rawEntries;
    cm.readRegistry(rawEntries);

    display_entries_ = rawEntries;

    if (display_entries_.empty())
    {
        totalLabel->copy_label("No cache entries found.");
        table->syncRowCount();
        updateButtonStates();
        return;
    }

    long grandTotal = 0;
    for (size_t i = 0; i < display_entries_.size(); i++)
        grandTotal += display_entries_[i].sizeBytes;

    char totalBuf[128];
    if (grandTotal > 0)
    {
        std::string sizeStr = formatSize(grandTotal);
        std::snprintf(totalBuf, sizeof(totalBuf),
                      "Total cache: %s across %zu entries",
                      sizeStr.c_str(), display_entries_.size());
    }
    else
    {
        std::snprintf(totalBuf, sizeof(totalBuf),
                      "Total cache: unknown across %zu entries",
                      display_entries_.size());
    }
    totalLabel->copy_label(totalBuf);

    applySort();
    table->syncRowCount();
    updateButtonStates();
}

void CacheManagementWindow::updateButtonStates()
{
    int selected_count = 0;
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r))
            selected_count++;
    }

    if (selected_count > 0)
        evictBtn->activate();
    else
        evictBtn->deactivate();
}

void CacheManagementWindow::evictSelected()
{
    int nitems = table->rows();
    std::vector<int> selectedRows;
    for (int i = 0; i < nitems; i++)
        if (table->row_selected(i))
            selectedRows.push_back(i);

    if (selectedRows.empty()) return;

    int total = (int)selectedRows.size();
    int evicted = 0;
    int failed = 0;
    std::string log;

    for (int done = 0; done < total; done++)
    {
        int idx = selectedRows[done];
        if (idx < 0 || idx >= (int)display_entries_.size())
            continue;

        const CacheManager::RegistryEntry &entry = display_entries_[idx];
        std::string label = entry.nodeName + ":" + entry.cachePath;

        char status[512];
        std::snprintf(status, sizeof(status),
                      "Evicting %d/%d: %s ... removing files",
                      done + 1, total, label.c_str());
        totalLabel->copy_label(status);
        Fl::check();

        bool evictOk = cm.evict(entry.key, entry.nodeName);

        if (evictOk)
        {
            std::snprintf(status, sizeof(status),
                          "Evicting %d/%d: %s ... verifying removal",
                          done + 1, total, label.c_str());
            totalLabel->copy_label(status);
            Fl::check();

            bool stillThere = !entry.cachePath.empty() && exists(FileName(entry.cachePath));
            if (stillThere)
            {
                std::snprintf(status, sizeof(status),
                              "Evicting %d/%d: %s ... FAILED (directory still exists)",
                              done + 1, total, label.c_str());
                totalLabel->copy_label(status);
                Fl::check();

                failed++;
                log += "  " + label + " ... rm reported success but directory still present\n";
            }
            else
            {
                std::snprintf(status, sizeof(status),
                              "Evicting %d/%d: %s ... verified removed",
                              done + 1, total, label.c_str());
                totalLabel->copy_label(status);
                Fl::check();

                evicted++;
                log += "  " + label + " ... done\n";
            }
        }
        else
        {
            std::snprintf(status, sizeof(status),
                          "Evicting %d/%d: %s ... FAILED",
                          done + 1, total, label.c_str());
            totalLabel->copy_label(status);
            Fl::check();

            failed++;
            log += "  " + label + " ... failed (eviction returned error)\n";
        }
    }

    char summary[256];
    std::snprintf(summary, sizeof(summary),
                  "Evicted %d, failed %d (out of %d selected)",
                  evicted, failed, total);
    log += "\n";
    log += summary;

    totalLabel->copy_label(summary);
    Fl::check();
    fl_message("%s", log.c_str());

    refresh();
}

void CacheManagementWindow::evictAll()
{
    int total = (int)display_entries_.size();
    int evicted = 0;
    int failed = 0;
    std::string log;

    for (size_t i = 0; i < display_entries_.size(); i++)
    {
        const CacheManager::RegistryEntry &entry = display_entries_[i];
        std::string label = entry.nodeName + ":" + entry.cachePath;

        char status[512];
        std::snprintf(status, sizeof(status),
                      "Evicting %zu/%d: %s ... removing files",
                      i + 1, total, label.c_str());
        totalLabel->copy_label(status);
        Fl::check();

        bool evictOk = cm.evict(entry.key, entry.nodeName);

        if (evictOk)
        {
            std::snprintf(status, sizeof(status),
                          "Evicting %zu/%d: %s ... verifying removal",
                          i + 1, total, label.c_str());
            totalLabel->copy_label(status);
            Fl::check();

            bool stillThere = !entry.cachePath.empty() && exists(FileName(entry.cachePath));
            if (stillThere)
            {
                std::snprintf(status, sizeof(status),
                              "Evicting %zu/%d: %s ... FAILED (directory still exists)",
                              i + 1, total, label.c_str());
                totalLabel->copy_label(status);
                Fl::check();

                failed++;
                log += "  " + label + " ... rm reported success but directory still present\n";
            }
            else
            {
                std::snprintf(status, sizeof(status),
                              "Evicting %zu/%d: %s ... verified removed",
                              i + 1, total, label.c_str());
                totalLabel->copy_label(status);
                Fl::check();

                evicted++;
                log += "  " + label + " ... done\n";
            }
        }
        else
        {
            std::snprintf(status, sizeof(status),
                          "Evicting %zu/%d: %s ... FAILED",
                          i + 1, total, label.c_str());
            totalLabel->copy_label(status);
            Fl::check();

            failed++;
            log += "  " + label + " ... failed (eviction returned error)\n";
        }
    }

    char summary[256];
    std::snprintf(summary, sizeof(summary),
                  "Evicted %d, failed %d (out of %d total)",
                  evicted, failed, total);
    log += "\n";
    log += summary;

    totalLabel->copy_label(summary);
    Fl::check();
    fl_message("%s", log.c_str());

    refresh();
}

void CacheManagementWindow::sortByColumn(int col, bool asc)
{
    if (col < 0 || col >= 6) return;
    table->setSort(col, asc);
    applySort();
    table->syncRowCount();
}

void CacheManagementWindow::applySort()
{
    int sc = table->sortCol();
    bool sa = table->sortAsc();
    if (sc < 0) return;
    std::sort(display_entries_.begin(), display_entries_.end(),
              [sc, sa](const CacheManager::RegistryEntry &a,
                       const CacheManager::RegistryEntry &b)
              {
                  int cmp = 0;
                  switch (sc)
                  {
                  case 0: cmp = a.sourceJobDir.compare(b.sourceJobDir); break;
                  case 1:
                      cmp = (a.sizeBytes < b.sizeBytes) ? -1 :
                            (a.sizeBytes > b.sizeBytes) ? 1 : 0;
                      break;
                  case 2: cmp = a.nodeName.compare(b.nodeName); break;
                  case 3: cmp = a.cachePath.compare(b.cachePath); break;
                  case 4:
                      cmp = (a.created < b.created) ? -1 :
                            (a.created > b.created) ? 1 : 0;
                      break;
                  case 5:
                      cmp = (a.lastAccess < b.lastAccess) ? -1 :
                            (a.lastAccess > b.lastAccess) ? 1 : 0;
                      break;
                  }
                  return sa ? (cmp < 0) : (cmp > 0);
              });
}

void CacheManagementWindow::cb_evict_selected(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->evictSelected();
}

void CacheManagementWindow::cb_evict_all(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->evictAll();
}

void CacheManagementWindow::cb_refresh(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->refresh();
}

void CacheManagementWindow::cb_close(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->hide();
}
