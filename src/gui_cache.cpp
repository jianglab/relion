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
#include <FL/fl_ask.H>

static void formatSize(long bytes, char *buf, size_t bufsize)
{
    if (bytes < 1024)
        std::snprintf(buf, bufsize, "%ld B", bytes);
    else if (bytes < 1024 * 1024)
        std::snprintf(buf, bufsize, "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024LL * 1024 * 1024)
        std::snprintf(buf, bufsize, "%.1f MB", bytes / (1024.0 * 1024.0));
    else
        std::snprintf(buf, bufsize, "%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
}

static void truncatePath(const std::string &path, char *buf, size_t bufsize)
{
    if (path.size() + 1 <= bufsize)
    {
        strncpy(buf, path.c_str(), bufsize - 1);
        buf[bufsize - 1] = '\0';
        return;
    }
    size_t keep = bufsize - 4;
    strncpy(buf, "...", bufsize);
    strncpy(buf + 3, path.c_str() + (path.size() - keep), keep);
    buf[bufsize - 1] = '\0';
}

CacheManagementWindow::CacheManagementWindow(int w, int h, const char *title)
    : Fl_Window(w, h, title)
{
    int margin = 10;
    int labelH = 30;
    int btnH = 30;
    int btnW = 120;
    int browserH = h - margin * 3 - labelH - btnH - 10;

    totalLabel = new Fl_Box(margin, margin, w - margin * 2, labelH, "Loading...");
    totalLabel->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    totalLabel->labelsize(14);
    totalLabel->labelfont(FL_BOLD);

    browser = new Fl_Browser(margin, margin + labelH + 5,
                             w - margin * 2, browserH);
    browser->type(FL_MULTI_BROWSER);
    browser->textsize(13);
    browser->selection_color(FL_BLUE);
    browser->color(FL_BACKGROUND2_COLOR);

    int btnY = margin + labelH + 5 + browserH + 10;

    Fl_Button *evictBtn = new Fl_Button(margin, btnY, btnW, btnH, "Evict selected");
    evictBtn->callback(cb_evict_selected, this);

    Fl_Button *allBtn = new Fl_Button(margin + btnW + 10, btnY, btnW, btnH, "Evict all");
    allBtn->callback(cb_evict_all, this);

    Fl_Button *refreshBtn = new Fl_Button(margin + (btnW + 10) * 2, btnY, btnW, btnH, "Refresh");
    refreshBtn->callback(cb_refresh, this);

    Fl_Button *closeBtn = new Fl_Button(w - margin - btnW, btnY, btnW, btnH, "Close");
    closeBtn->callback(cb_close, this);

    end();
    resizable(browser);

    cm.setRegistryDir("~");
    refresh();
}

void CacheManagementWindow::refresh()
{
    browser->clear();
    displayEntries.clear();

    std::vector<CacheManager::RegistryEntry> rawEntries;
    cm.readRegistry(rawEntries);

    if (rawEntries.empty())
    {
        totalLabel->copy_label("No cache entries found.");
        return;
    }

    long grandTotal = 0;
    for (size_t i = 0; i < rawEntries.size(); i++)
    {
        grandTotal += rawEntries[i].sizeBytes;
    }

    char totalBuf[128];
    if (grandTotal > 0)
    {
        char sizeStr[32];
        formatSize(grandTotal, sizeStr, sizeof(sizeStr));
        std::snprintf(totalBuf, sizeof(totalBuf),
                      "Total cache: %s across %zu entries", sizeStr, rawEntries.size());
    }
    else
    {
        std::snprintf(totalBuf, sizeof(totalBuf),
                      "Total cache: unknown across %zu entries", rawEntries.size());
    }
    totalLabel->copy_label(totalBuf);

    for (size_t i = 0; i < rawEntries.size(); i++)
    {
        char sizeStr[32];
        formatSize(rawEntries[i].sizeBytes, sizeStr, sizeof(sizeStr));

        char line[1024];
        std::string src = rawEntries[i].sourceJobDir;
        if (src.empty())
            src = rawEntries[i].key.substr(0, 16);

        std::string node = rawEntries[i].nodeName;
        if (node.empty())
            node = "(unknown)";

        char pathBuf[256];
        truncatePath(rawEntries[i].cachePath, pathBuf, sizeof(pathBuf));

        std::snprintf(line, sizeof(line), "%s  [%s]  node: %s  path: %s",
                      src.c_str(), sizeStr, node.c_str(), pathBuf);

        displayEntries.push_back(rawEntries[i]);
        browser->add(line);
    }
}

void CacheManagementWindow::evictSelected()
{
    int nitems = browser->size();
    int evicted = 0;
    int failed = 0;
    int total = 0;
    for (int i = 1; i <= nitems; i++)
        if (browser->selected(i)) total++;

    std::string log;
    int done = 0;

    for (int i = nitems; i >= 1; i--)
    {
        if (!browser->selected(i))
            continue;

        size_t idx = (size_t)(i - 1);
        if (idx >= displayEntries.size())
            continue;

        const CacheManager::RegistryEntry &entry = displayEntries[idx];
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
                browser->remove(i);
                displayEntries.erase(displayEntries.begin() + (long)idx);
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
        done++;
    }

    char summary[256];
    std::snprintf(summary, sizeof(summary),
                  "Evicted %d, failed %d (out of %d selected)", evicted, failed, total);

    if (failed > 0)
    {
        log += "\n" + std::string(summary);
    }
    else
    {
        log += "\n" + std::string(summary);
    }
    totalLabel->copy_label(summary);
    Fl::check();
    fl_message("%s", log.c_str());

    refresh();
}

void CacheManagementWindow::evictAll()
{
    int evicted = 0;
    int failed = 0;
    int total = (int)displayEntries.size();
    std::string log;

    for (size_t i = 0; i < displayEntries.size(); i++)
    {
        const CacheManager::RegistryEntry &entry = displayEntries[i];
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
                  "Evicted %d, failed %d (out of %d total)", evicted, failed, total);

    if (failed > 0)
    {
        log += "\n" + std::string(summary);
    }
    else
    {
        log += "\n" + std::string(summary);
    }
    totalLabel->copy_label(summary);
    Fl::check();
    fl_message("%s", log.c_str());

    refresh();
}

void CacheManagementWindow::cb_refresh(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->refresh();
}

void CacheManagementWindow::cb_evict_selected(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->evictSelected();
}

void CacheManagementWindow::cb_evict_all(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->evictAll();
}

void CacheManagementWindow::cb_close(Fl_Widget *, void *v)
{
    ((CacheManagementWindow *)v)->hide();
}
