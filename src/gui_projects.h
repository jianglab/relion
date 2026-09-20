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

#ifndef GUI_PROJECTS_H_
#define GUI_PROJECTS_H_

#include <memory>
#include <string>
#include <vector>
#include <FL/Fl_Window.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Button.H>
#include "gui_table.h"
#include "src/filename.h"

class ManageProjectsWindow;

/* Offer to delete a project's intermediate per-iteration files.
 *
 * Scans the project, shows what would go and how much space it would free, and
 * deletes only if the user agrees. Returns true if anything was deleted, so a
 * caller that displays project sizes knows to refresh them.
 */
bool runIntermediateCleanupDialog(const std::vector<std::string> &project_paths);

class ProjectTable : public RelionTable
{
    ManageProjectsWindow *win_;
protected:
    int numberOfRows() const override;
    int numberOfCols() const override;
    const char* headerText(int col) const override;
    std::string cellText(int row, int col) const override;
    Fl_Color cellBackgroundColor(int row, int col) override;
    bool isCurrentRow(int row) const override;
    void onDoubleClick(int row) override;
    void onSelectionChanged() override;
    void onSortChanged(int col, bool asc) override;
public:
    ProjectTable(int X, int Y, int W, int H, ManageProjectsWindow *w);
};

class ProjectManager
{
public:
    struct Project
    {
        std::string path;
        std::string name;
        std::string last_opened;
        int jobs = 0;
        long long size_bytes = 0;
        bool exists() const;
    };

    ProjectManager();
    bool load();
    bool save();
    void add(const std::string &path, const std::string &name);
    void remove(const std::string &path);
    void rename(const std::string &path, const std::string &new_name);
    std::vector<Project> getAll() const;
    std::vector<Project> getRecent(int max_count = 5) const;
    bool isRegistered(const std::string &path) const;
    Project *findByPath(const std::string &path);
    const Project *findByPath(const std::string &path) const;
    void touchLastOpened(const std::string &path);
    static FileName getRegistryDir();
    static FileName getRegistryPath();

private:
    std::vector<Project> projects_;
    static std::string now();
};

class ManageProjectsWindow : public Fl_Window
{
public:
    ManageProjectsWindow(int w, int h, const char *title = "Manage Projects");
    ~ManageProjectsWindow();
    void refresh();

    const std::vector<ProjectManager::Project>& getDisplayProjects() const { return display_projects_; }
    ProjectManager& getProjectManager() { return pm; }
    const std::string& getOpenPath() const { return selected_open_path_; }
    const std::string& currentProjectPath() const { return current_project_path_; }
    void updateButtonStates();
    /// Fill the name/path boxes from the selection, and update the buttons.
    void onRowSelectionChanged();
    void openSelected();
    void sortByColumn(int col, bool asc);

private:
    ProjectManager pm;
    std::vector<ProjectManager::Project> display_projects_;
    std::string selected_open_path_;
    std::string current_project_path_;
    ProjectTable *table;
    Fl_Input *name_input;
    Fl_Output *path_input;
    Fl_Button *open_btn;
    Fl_Button *remove_btn;
    Fl_Button *rename_btn;
    Fl_Button *refresh_btn;
    Fl_Button *cleanup_btn;

    static void cb_open(Fl_Widget *, void *v);
    static void cb_remove(Fl_Widget *, void *v);
    static void cb_rename(Fl_Widget *, void *v);
    static void cb_refresh(Fl_Widget *, void *v);
    static void cb_cleanup(Fl_Widget *, void *v);
    static void cb_close(Fl_Widget *, void *v);
    static void cb_name_input(Fl_Widget *, void *v);

    void removeSelected();
    void renameSelected();
    void refreshSelected();
    void cleanupSelected();

    /* Measuring a project means du over the whole tree, which on a project
     * with many jobs takes seconds. Doing that for every row before the window
     * appears made opening the dialog feel broken, so the rows are shown at
     * once with their sizes pending and filled in from worker threads as the
     * measurements come back. */
public:
    struct SizeScan;   ///< public only so that the worker function can name it
private:
    std::shared_ptr<SizeScan> scan_;
    void startSizeScan(const std::vector<std::string> &paths);
    void pollSizeScan();
    void cancelSizeScan();
    static void cb_size_poll(void *v);
    void applySort();
};

#endif
