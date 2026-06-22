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

#include "gui_projects.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <FL/fl_ask.H>
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Return_Button.H>
#include <FL/Fl_Round_Button.H>
#include <FL/fl_draw.H>

// ---------------------------------------------------------------------------
// ProjectManager
// ---------------------------------------------------------------------------

FileName ProjectManager::getRegistryDir()
{
    const char *home = std::getenv("HOME");
    if (home)
        return FileName(std::string(home) + "/.relion/");
    return FileName("~/.relion/");
}

FileName ProjectManager::getRegistryPath()
{
    return getRegistryDir() + "projects.star";
}

ProjectManager::ProjectManager()
{
    // Nothing to init; call load() explicitly
}

bool ProjectManager::load()
{
    projects_.clear();
    FileName fn = getRegistryPath();
    if (!exists(fn))
        return false;

    std::ifstream fh(fn.c_str());
    if (!fh.is_open())
        return false;

    // Look for data_projects loop with _rlnProjectPath, _rlnProjectName, _rlnProjectLastOpened
    std::string line;
    bool in_data = false;
    int col_path = -1, col_name = -1, col_time = -1;
    int col = 0;

    while (std::getline(fh, line))
    {
        if (line.empty())
            continue;

        if (line.find("data_projects") == 0)
        {
            in_data = true;
            continue;
        }

        if (!in_data)
            continue;

        if (line.find("loop_") == 0)
        {
            col_path = col_name = col_time = -1;
            col = 0;
            continue;
        }

        if (line[0] == '_')
        {
            if (line.find("_rlnProjectPath") != std::string::npos)
                col_path = col;
            else if (line.find("_rlnProjectName") != std::string::npos)
                col_name = col;
            else if (line.find("_rlnProjectLastOpened") != std::string::npos)
                col_time = col;
            col++;
            continue;
        }

        if (line[0] == '#')
            continue;

        if (col_path < 0 || col_name < 0 || col_time < 0)
            continue;

        // Data line: parse columns (space-separated)
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token)
            tokens.push_back(token);

        if (tokens.size() <= (size_t)std::max(col_path, std::max(col_name, col_time)))
            continue;

        Project p;
        p.path = tokens[col_path];
        p.name = tokens[col_name];
        p.last_opened = tokens[col_time];
        projects_.push_back(p);
    }
    fh.close();
    return true;
}

bool ProjectManager::save()
{
    FileName dir = getRegistryDir();
    if (!exists(dir))
    {
        int res = mktree(dir, 0777);
        if (res != 0)
            return false;
    }

    FileName fn = getRegistryPath();
    std::ofstream fh(fn.c_str());
    if (!fh.is_open())
        return false;

    fh << "# version 50001\n\n";
    fh << "data_projects\n\n";
    fh << "loop_\n";
    fh << "_rlnProjectPath #1\n";
    fh << "_rlnProjectName #2\n";
    fh << "_rlnProjectLastOpened #3\n";
    for (size_t i = 0; i < projects_.size(); i++)
    {
        fh << projects_[i].path << "  "
           << projects_[i].name << "  "
           << projects_[i].last_opened << "\n";
    }
    fh << "\n";
    fh.close();
    return true;
}

void ProjectManager::add(const std::string &path, const std::string &name)
{
    // Remove existing entry for this path
    remove(path);
    Project p;
    p.path = path;
    p.name = name;
    p.last_opened = now();
    projects_.push_back(p);
}

void ProjectManager::remove(const std::string &path)
{
    for (auto it = projects_.begin(); it != projects_.end(); ++it)
    {
        if (it->path == path)
        {
            projects_.erase(it);
            return;
        }
    }
}

void ProjectManager::rename(const std::string &path, const std::string &new_name)
{
    for (auto &p : projects_)
    {
        if (p.path == path)
        {
            p.name = new_name;
            return;
        }
    }
}

std::vector<ProjectManager::Project> ProjectManager::getAll() const
{
    return projects_;
}

std::vector<ProjectManager::Project> ProjectManager::getRecent(int max_count) const
{
    std::vector<Project> sorted = projects_;
    std::sort(sorted.begin(), sorted.end(),
              [](const Project &a, const Project &b)
              {
                  return a.last_opened > b.last_opened;
              });
    if (sorted.size() > (size_t)max_count)
        sorted.resize(max_count);
    return sorted;
}

bool ProjectManager::isRegistered(const std::string &path) const
{
    return findByPath(path) != nullptr;
}

ProjectManager::Project *ProjectManager::findByPath(const std::string &path)
{
    for (auto &p : projects_)
    {
        if (p.path == path)
            return &p;
    }
    return nullptr;
}

const ProjectManager::Project *ProjectManager::findByPath(const std::string &path) const
{
    for (const auto &p : projects_)
    {
        if (p.path == path)
            return &p;
    }
    return nullptr;
}

std::string ProjectManager::now()
{
    time_t t = std::time(nullptr);
    struct tm *tm = std::localtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", tm);
    return std::string(buf);
}

bool ProjectManager::Project::exists() const
{
    struct stat st;
    return (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

void ProjectManager::touchLastOpened(const std::string &path)
{
    Project *p = findByPath(path);
    if (p)
        p->last_opened = now();
}

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

namespace {

static bool is_jobdir(const std::string &name)
{
    if (name.size() <= 3 || name.substr(0, 3) != "job") return false;
    for (size_t i = 3; i < name.size(); i++)
        if (!std::isdigit((unsigned char)name[i])) return false;
    return true;
}

static int count_jobdirs_in(const std::string &parent)
{
    DIR *dir = opendir(parent.c_str());
    if (!dir) return 0;
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string name(entry->d_name);
        if (is_jobdir(name))
        {
            struct stat st;
            std::string full = parent + "/" + name;
            if (::stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                count++;
        }
    }
    closedir(dir);
    return count;
}

static int count_jobs(const std::string &path)
{
    DIR *dir = opendir(path.c_str());
    if (!dir) return 0;
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string name(entry->d_name);
        if (name.empty() || name[0] == '.') continue;
        struct stat st;
        std::string full = path + "/" + name;
        if (::stat(full.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;
        // Count jobNNN directly at top level (RELION 4.0 flat style)
        if (is_jobdir(name))
        {
            count++;
        }
        else
        {
            // Scan one level deep for jobNNN subdirs (RELION 5.0 nested style)
            count += count_jobdirs_in(full);
        }
    }
    closedir(dir);
    return count;
}

long long compute_size_kb(const std::string &path)
{
    std::string cmd = "du -sk \"" + path + "\" 2>/dev/null";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) return 0;
    long long kb = 0;
    if (fscanf(fp, "%lld", &kb) != 1) kb = 0;
    pclose(fp);
    return kb;
}

std::string format_size(long long bytes)
{
    double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GB", gb);
    return std::string(buf);
}

}

// ---------------------------------------------------------------------------
// ProjectTable
// ---------------------------------------------------------------------------

ProjectTable::ProjectTable(int X, int Y, int W, int H, ManageProjectsWindow *w)
    : RelionTable(X, Y, W, H), win_(w)
{
}

int ProjectTable::numberOfRows() const
{
    if (!win_) return 0;
    return (int)win_->getDisplayProjects().size();
}

int ProjectTable::numberOfCols() const
{
    return 5;
}

const char* ProjectTable::headerText(int col) const
{
    static const char *headers[] = { "Name", "Path", "Jobs", "Size", "Last Opened" };
    if (col >= 0 && col < 5)
        return headers[col];
    return "";
}

std::string ProjectTable::cellText(int row, int col) const
{
    if (!win_) return "";
    const auto &projects = win_->getDisplayProjects();
    if (row < 0 || row >= (int)projects.size()) return "";

    const auto &p = projects[row];
    switch (col)
    {
    case 0:
    {
        std::string text;
        if (p.path == win_->currentProjectPath())
            text = "\xe2\x97\x8f ";  // black circle marker
        text += p.name;
        return text;
    }
    case 1: return p.path;
    case 2: return std::to_string(p.jobs);
    case 3: return format_size(p.size_bytes);
    case 4: return p.last_opened;
    default: return "";
    }
}

Fl_Color ProjectTable::cellBackgroundColor(int row, int col)
{
    if (!win_) return RelionTable::cellBackgroundColor(row, col);
    const auto &projects = win_->getDisplayProjects();
    if (row < 0 || row >= (int)projects.size())
        return RelionTable::cellBackgroundColor(row, col);

    if (row_selected(row))
        return FL_SELECTION_COLOR;

    if (projects[row].path == win_->currentProjectPath())
        return fl_color_average(FL_GREEN, FL_WHITE, 0.2f);

    if (row % 2)
        return fl_color_average(FL_WHITE, FL_BACKGROUND_COLOR, 0.85f);
    return FL_WHITE;
}

bool ProjectTable::isCurrentRow(int row) const
{
    if (!win_) return false;
    const auto &projects = win_->getDisplayProjects();
    return (row >= 0 && row < (int)projects.size() &&
            projects[row].path == win_->currentProjectPath());
}

void ProjectTable::onDoubleClick(int row)
{
    if (win_)
        win_->openSelected();
}

void ProjectTable::onSelectionChanged()
{
    if (win_)
        win_->updateButtonStates();
}

void ProjectTable::onSortChanged(int col, bool asc)
{
    if (win_)
        win_->sortByColumn(col, asc);
}

// ---------------------------------------------------------------------------
// ManageProjectsWindow
// ---------------------------------------------------------------------------

ManageProjectsWindow::ManageProjectsWindow(int w, int h, const char *title)
    : Fl_Window(w, h, title)
{
    pm.load();

    // Capture the current project directory (CWD at window creation time)
    {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)))
            current_project_path_ = cwd;
    }

    int pad = 10;
    int bh = 30;
    int btn_y = h - bh - pad;
    int label_w = 50;

    // Table
    int table_x = pad;
    int table_y = pad;
    int table_w = w - 2 * pad;
    int table_h = btn_y - 5 * pad - 2 * bh - 8;

    table = new ProjectTable(table_x, table_y, table_w, table_h, this);
    table->type(Fl_Table_Row::SELECT_MULTI);
    table->col_header(true);
    table->col_header_height(22);
    table->row_height_all(24);
    table->col_resize(true);
    table->col_resize_min(40);
    table->row_resize(false);
    table->cols(5);

    // Fl_Table constructor leaves current group set to its internal Fl_Scroll,
    // so subsequent widgets would be added there and not appear in the window.
    begin();

    int avail_w = table_w - 16;
    int path_w = avail_w - 160 - 60 - 85 - 160;
    if (path_w < 100) path_w = 100;
    table->col_width(0, 160);
    table->col_width(1, path_w);
    table->col_width(2, 60);
    table->col_width(3, 85);
    table->col_width(4, 160);

    // Path input
    int path_y = table_y + table_h + pad;
    Fl_Box *pl = new Fl_Box(pad, path_y, label_w, bh, "Path:");
    pl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    path_input = new Fl_Output(pad + label_w + 5, path_y, w - 2 * pad - label_w - 5, bh);
    path_input->tooltip("Full path to the project directory");

    // Name input
    int name_y = path_y + bh + pad;
    Fl_Box *nl = new Fl_Box(pad, name_y, label_w, bh, "Name:");
    nl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    name_input = new Fl_Input(pad + label_w + 5, name_y, w - 2 * pad - label_w - 5, bh);
    name_input->tooltip("Display name (editable, press Enter to rename)");
    name_input->when(FL_WHEN_ENTER_KEY);
    name_input->callback(cb_name_input, this);

    // Buttons
    int bx = pad;
    open_btn = new Fl_Button(bx, btn_y, 90, bh, " Open ");
    open_btn->callback(cb_open, this);
    bx += 90 + pad;

    remove_btn = new Fl_Button(bx, btn_y, 90, bh, " Remove ");
    remove_btn->callback(cb_remove, this);
    remove_btn->deactivate();
    bx += 90 + pad;

    rename_btn = new Fl_Button(bx, btn_y, 90, bh, " Rename ");
    rename_btn->callback(cb_rename, this);
    rename_btn->deactivate();
    bx += 90 + pad;

    refresh_btn = new Fl_Button(bx, btn_y, 90, bh, " Refresh ");
    refresh_btn->callback(cb_refresh, this);
    refresh_btn->deactivate();
    bx += 90 + pad;

    Fl_Button *close_btn = new Fl_Button(w - pad - 90, btn_y, 90, bh, " Close ");
    close_btn->callback(cb_close, this);

    refresh();
    end();
    resizable(table);
}

void ManageProjectsWindow::refresh()
{
    display_projects_ = pm.getAll();
    for (auto &p : display_projects_)
    {
        if (p.exists())
        {
            p.jobs = count_jobs(p.path);
            p.size_bytes = compute_size_kb(p.path) * 1024;
        }
    }
    applySort();
    table->syncRowCount();

    name_input->value("");
    path_input->value("");
    updateButtonStates();
}

void ManageProjectsWindow::cb_table(Fl_Widget *, void *v)
{
    ManageProjectsWindow *w = (ManageProjectsWindow *)v;
    Fl_Table::TableContext ctx = w->table->callback_context();

    if (ctx != Fl_Table::CONTEXT_COL_HEADER)
        w->onRowSelectionChanged();
}

void ManageProjectsWindow::cb_open(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->openSelected();
}

void ManageProjectsWindow::cb_remove(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->removeSelected();
}

void ManageProjectsWindow::cb_rename(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->renameSelected();
}

void ManageProjectsWindow::cb_refresh(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->refreshSelected();
}

void ManageProjectsWindow::cb_name_input(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->renameSelected();
}

void ManageProjectsWindow::cb_close(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->hide();
}

void ManageProjectsWindow::onRowSelectionChanged()
{
    int selected_count = 0;
    int row = -1;
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r)) { selected_count++; row = r; }
    }
    if (selected_count == 1 && row >= 0 && row < (int)display_projects_.size())
    {
        name_input->value(display_projects_[row].name.c_str());
        path_input->value(display_projects_[row].path.c_str());
    }
    else
    {
        name_input->value("");
        path_input->value("");
    }
    updateButtonStates();
}

void ManageProjectsWindow::updateButtonStates()
{
    int selected_count = 0;
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r)) { selected_count++; }
    }

    if (selected_count == 1)
        open_btn->activate();
    else
        open_btn->deactivate();

    if (selected_count > 0)
        remove_btn->activate();
    else
        remove_btn->deactivate();

    if (selected_count == 1)
        rename_btn->activate();
    else
        rename_btn->deactivate();

    if (selected_count > 0)
        refresh_btn->activate();
    else
        refresh_btn->deactivate();
}

void ManageProjectsWindow::sortByColumn(int col, bool asc)
{
    if (col < 0 || col >= 5) return;
    table->setSort(col, asc);
    applySort();
    table->syncRowCount();
}

void ManageProjectsWindow::applySort()
{
    int sc = table->sortCol();
    bool sa = table->sortAsc();
    if (sc < 0) return;
    std::sort(display_projects_.begin(), display_projects_.end(),
              [sc, sa](const ProjectManager::Project &a, const ProjectManager::Project &b)
              {
                  int cmp = 0;
                  switch (sc)
                  {
                  case 0: cmp = a.name.compare(b.name); break;
                  case 1: cmp = a.path.compare(b.path); break;
                  case 2:
                      cmp = (a.jobs < b.jobs) ? -1 : (a.jobs > b.jobs) ? 1 : 0;
                      break;
                  case 3:
                      cmp = (a.size_bytes < b.size_bytes) ? -1 : (a.size_bytes > b.size_bytes) ? 1 : 0;
                      break;
                  case 4: cmp = a.last_opened.compare(b.last_opened); break;
                  }
                  return sa ? (cmp < 0) : (cmp > 0);
              });
}

void ManageProjectsWindow::openSelected()
{
    if (display_projects_.empty()) return;

    int row = -1;
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r)) { row = r; break; }
    }
    if (row < 0 || row >= (int)display_projects_.size()) return;

    selected_open_path_ = display_projects_[row].path;
    hide();
}

void ManageProjectsWindow::removeSelected()
{
    if (display_projects_.empty()) return;

    // Collect selected rows
    std::vector<int> rows;
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r)) rows.push_back(r);
    }
    if (rows.empty()) return;

    if (rows.size() == 1)
    {
        int row = rows[0];
        if (row < 0 || row >= (int)display_projects_.size()) return;

        const auto &proj = display_projects_[row];

        // Custom dialog with radio buttons
        struct Dialog {
            Fl_Window *win;
            Fl_Round_Button *entry_only;
            Fl_Round_Button *entry_and_dir;
            int result; // 0=cancel, 1=entry only, 2=entry+dir
            static void cb_ok(Fl_Widget *, void *v) {
                Dialog *d = (Dialog *)v;
                d->result = d->entry_only->value() ? 1 : 2;
                d->win->hide();
            }
            static void cb_cancel(Fl_Widget *, void *v) {
                Dialog *d = (Dialog *)v;
                d->result = 0;
                d->win->hide();
            }
        };

        Dialog d;
        d.result = 0;
        d.win = new Fl_Window(420, 160, "Remove Project");

        char msg[512];
        snprintf(msg, sizeof(msg), "Remove project \"%s\"?", proj.name.c_str());
        Fl_Box *label = new Fl_Box(20, 10, 380, 25, msg);
        label->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        d.entry_only = new Fl_Round_Button(30, 40, 360, 25, "Remove only the project entry");
        d.entry_only->type(FL_RADIO_BUTTON);
        d.entry_only->value(1);

        d.entry_and_dir = new Fl_Round_Button(30, 70, 360, 25, "Remove the project entry and the folder");
        d.entry_and_dir->type(FL_RADIO_BUTTON);

        Fl_Return_Button *cancel = new Fl_Return_Button(220, 115, 80, 25, "Cancel");
        cancel->callback(Dialog::cb_cancel, &d);

        Fl_Button *ok = new Fl_Button(310, 115, 80, 25, "Remove");
        ok->callback(Dialog::cb_ok, &d);

        d.win->end();
        d.win->set_modal();
        d.win->show();

        while (d.win->shown()) Fl::wait();
        delete d.win;

        if (d.result == 0) return;

        if (d.result == 2)
        {
            std::string cmd = "rm -rf \"" + proj.path + "\"";
            if (system(cmd.c_str()) != 0) { }
        }

        pm.remove(proj.path);
        pm.save();
        refresh();
    }
    else
    {
        // Multi-selection: remove all from registry only
        char msg[512];
        snprintf(msg, sizeof(msg), "Remove %zu projects from the registry?\n(The project directories will NOT be deleted.)",
                 rows.size());
        int ret = fl_choice("%s", "Remove", 0, "Cancel", msg);
        if (ret == 2) return;

        for (int row : rows)
        {
            if (row >= 0 && row < (int)display_projects_.size())
                pm.remove(display_projects_[row].path);
        }
        pm.save();
        refresh();
    }
}

void ManageProjectsWindow::renameSelected()
{
    if (display_projects_.empty()) return;

    int row = -1;
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r)) { row = r; break; }
    }
    if (row < 0 || row >= (int)display_projects_.size()) return;

    const char *new_name = name_input->value();
    if (!new_name || strlen(new_name) == 0)
    {
        fl_alert("Please enter a name.");
        return;
    }
    pm.rename(display_projects_[row].path, std::string(new_name));
    pm.save();
    refresh();
}

void ManageProjectsWindow::refreshSelected()
{
    for (int r = 0; r < table->rows(); r++)
    {
        if (table->row_selected(r))
        {
            if (r >= 0 && r < (int)display_projects_.size())
            {
                ProjectManager::Project &p = display_projects_[r];
                if (p.exists())
                {
                    p.jobs = count_jobs(p.path);
                    p.size_bytes = compute_size_kb(p.path) * 1024;
                }
                else
                {
                    p.jobs = 0;
                    p.size_bytes = 0;
                }
            }
        }
    }
    applySort();
    table->syncRowCount();
}
