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
#include "src/remove_intermediates.h"
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
#include <FL/Fl_Progress.H>
#include <mutex>
#include <thread>

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
    // Negative means "still being measured": see ManageProjectsWindow::SizeScan
    if (bytes < 0) return "...";
    double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GB", gb);
    return std::string(buf);
}

}

// ---------------------------------------------------------------------------
// Removing intermediate files
// ---------------------------------------------------------------------------

namespace {

/* The window shown while files are being deleted.
 *
 * Deleting tens of thousands of files takes long enough that a dialog saying
 * what was done would be claiming a result that has not happened yet, so this
 * says what is happening, refuses to close while it happens, and only then
 * turns into the report of what was removed.
 */
struct CleanupProgressWindow {
	Fl_Window   *win;
	Fl_Box      *message;
	Fl_Progress *bar;
	Fl_Button   *close_btn;
	std::string  text;          ///< owned, since Fl_Box does not copy its label
	bool         busy;
	size_t       done_before;   ///< files finished in earlier projects
	size_t       grand_total;

	CleanupProgressWindow(size_t total_files, const std::string &what)
		: busy(true), done_before(0), grand_total(total_files)
	{
		win = new Fl_Window(480, 150, "Removing intermediate files");

		text = "Deleting " + std::to_string(total_files) + " file(s) from\n" + what + " ...";
		message = new Fl_Box(20, 15, 440, 55, text.c_str());
		message->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP);

		bar = new Fl_Progress(20, 75, 440, 22);
		bar->minimum(0);
		bar->maximum((float)(total_files > 0 ? total_files : 1));
		bar->value(0);
		bar->selection_color(FL_BLUE);

		close_btn = new Fl_Button(390, 110, 70, 26, "Close");
		close_btn->callback(cb_close, this);
		close_btn->deactivate();

		// The window manager's close button must not get around the above
		win->callback(cb_close, this);
		win->end();
		win->set_modal();
		win->show();
		Fl::check();
	}

	~CleanupProgressWindow() { delete win; }

	static void cb_close(Fl_Widget *, void *v)
	{
		CleanupProgressWindow *w = (CleanupProgressWindow *)v;
		if (!w->busy) w->win->hide();
	}

	/// relion_cleanup::ProgressFn
	static void onProgress(size_t done, size_t total, void *user_data)
	{
		CleanupProgressWindow *w = (CleanupProgressWindow *)user_data;

		// `done` counts within one project; the bar counts across all of them
		const size_t overall = w->done_before + done;
		w->bar->value((float)overall);

		char label[64];
		snprintf(label, sizeof(label), "%zu / %zu", overall, w->grand_total);
		w->bar->label(label);

		Fl::check();   // repaint, and keep the GUI answering the window manager
	}

	/// Called after each project, so the next one's counts continue from here.
	void projectDone(size_t files) { done_before += files; }

	/// Turn into the report of what was done, and let the user dismiss it.
	void finish(const std::string &report)
	{
		busy = false;
		text = report;
		message->label(text.c_str());
		bar->hide();
		close_btn->activate();
		win->redraw();
	}

	void waitUntilClosed()
	{
		while (win->shown()) Fl::wait();
	}
};

} // namespace

bool runIntermediateCleanupDialog(const std::vector<std::string> &project_paths)
{
    if (project_paths.empty()) return false;

    // Scanning walks every job directory, so say what is happening first
    std::vector<relion_cleanup::Plan> plans;
    long long total_bytes = 0;
    size_t total_files = 0;
    for (size_t i = 0; i < project_paths.size(); i++)
    {
        relion_cleanup::Plan plan = relion_cleanup::planIntermediateRemoval(project_paths[i]);
        total_bytes += plan.total_bytes;
        total_files += plan.remove.size();
        plans.push_back(plan);
    }

    if (total_files == 0)
    {
        fl_message("No intermediate files to remove.\n\n"
                   "Only the first and last iteration of each refinement are kept,\n"
                   "and nothing else was found.");
        return false;
    }

    std::string where = (project_paths.size() == 1)
                      ? project_paths[0]
                      : (std::to_string(project_paths.size()) + " projects");

    char msg[1024];
    snprintf(msg, sizeof(msg),
             "Remove %zu intermediate file(s) from\n%s,\nfreeing %s?\n\n"
             "The first and last iteration of every refinement are kept;\n"
             "the rounds in between are deleted and cannot be recovered.",
             total_files, where.c_str(),
             relion_cleanup::humanSize(total_bytes).c_str());

    int ret = fl_choice("%s", "Cancel", "Remove", NULL, msg);
    if (ret != 1) return false;

    CleanupProgressWindow progress(total_files, where);

    long long freed = 0;
    long removed = 0;
    std::vector<std::string> errors;
    for (size_t i = 0; i < plans.size(); i++)
    {
        long long f = 0;
        removed += relion_cleanup::applyRemoval(plans[i], f, errors,
                                                CleanupProgressWindow::onProgress, &progress);
        progress.projectDone(plans[i].remove.size());
        freed += f;
    }

    if (errors.empty())
    {
        snprintf(msg, sizeof(msg), "Deleted %ld file(s), freeing %s.",
                 removed, relion_cleanup::humanSize(freed).c_str());
    }
    else
    {
        snprintf(msg, sizeof(msg),
                 "Deleted %ld file(s), freeing %s.\n\n"
                 "%zu file(s) could not be removed, the first being:\n%s",
                 removed, relion_cleanup::humanSize(freed).c_str(),
                 errors.size(), errors[0].c_str());
    }

    progress.finish(msg);
    progress.waitUntilClosed();

    return removed > 0;
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
            text = "* ";
        text += p.name;
        return text;
    }
    case 1: return p.path;
    case 2: return (p.jobs < 0) ? "..." : std::to_string(p.jobs);
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
    // Not just the buttons: the name and path boxes below the table show the
    // selected project, and are what Rename edits
    if (win_)
        win_->onRowSelectionChanged();
}

void ProjectTable::onSortChanged(int col, bool asc)
{
    if (win_)
        win_->sortByColumn(col, asc);
}

// ---------------------------------------------------------------------------
// ManageProjectsWindow
// ---------------------------------------------------------------------------

/* Background measurement of project sizes.
 *
 * Workers take paths off a queue, run the du and the job count, and leave the
 * answers in `ready` for the GUI thread to pick up on a timer. Nothing here
 * touches FLTK: a worker only locks the mutex.
 *
 * The struct is held by shared_ptr and the workers are detached, so a window
 * that closes mid-scan sets `cancelled` and walks away; the last worker to
 * finish drops the struct. That matters because a du on a slow filesystem can
 * outlive the dialog by a long way, and joining it would hang the GUI.
 */
struct ManageProjectsWindow::SizeScan {
    struct Result {
        std::string path;
        long long   size_bytes;
        int         jobs;
    };

    std::mutex mutex;
    std::vector<std::string> queue;
    size_t next;
    std::vector<Result> ready;
    int outstanding;
    bool cancelled;

    SizeScan() : next(0), outstanding(0), cancelled(false) {}
};

namespace {

void sizeScanWorker(std::shared_ptr<ManageProjectsWindow::SizeScan> scan)
{
    for (;;)
    {
        std::string path;
        {
            std::lock_guard<std::mutex> guard(scan->mutex);
            if (scan->cancelled || scan->next >= scan->queue.size())
            {
                scan->outstanding--;
                return;
            }
            path = scan->queue[scan->next++];
        }

        ManageProjectsWindow::SizeScan::Result r;
        r.path = path;
        r.jobs = count_jobs(path);
        r.size_bytes = compute_size_kb(path) * 1024;

        {
            std::lock_guard<std::mutex> guard(scan->mutex);
            if (scan->cancelled)
            {
                scan->outstanding--;
                return;
            }
            scan->ready.push_back(r);
        }
    }
}

} // namespace

void ManageProjectsWindow::startSizeScan(const std::vector<std::string> &paths)
{
    cancelSizeScan();
    if (paths.empty()) return;

    scan_.reset(new SizeScan());
    scan_->queue = paths;

    // A handful of parallel du's hides the latency of any one slow project
    // without hammering the filesystem
    size_t n_workers = paths.size() < 4 ? paths.size() : 4;
    {
        std::lock_guard<std::mutex> guard(scan_->mutex);
        scan_->outstanding = (int)n_workers;
    }
    for (size_t i = 0; i < n_workers; i++)
    {
        std::shared_ptr<SizeScan> s = scan_;
        std::thread(sizeScanWorker, s).detach();
    }

    Fl::add_timeout(0.1, cb_size_poll, this);
}

void ManageProjectsWindow::cancelSizeScan()
{
    Fl::remove_timeout(cb_size_poll, this);
    if (scan_)
    {
        std::lock_guard<std::mutex> guard(scan_->mutex);
        scan_->cancelled = true;
    }
    scan_.reset();
}

void ManageProjectsWindow::cb_size_poll(void *v)
{
    ((ManageProjectsWindow *)v)->pollSizeScan();
}

void ManageProjectsWindow::pollSizeScan()
{
    if (!scan_) return;

    std::vector<SizeScan::Result> results;
    bool done;
    {
        std::lock_guard<std::mutex> guard(scan_->mutex);
        results.swap(scan_->ready);
        done = (scan_->outstanding <= 0);
    }

    for (size_t i = 0; i < results.size(); i++)
    {
        for (size_t p = 0; p < display_projects_.size(); p++)
        {
            if (display_projects_[p].path != results[i].path) continue;
            display_projects_[p].jobs = results[i].jobs;
            display_projects_[p].size_bytes = results[i].size_bytes;
            break;
        }
    }

    if (!results.empty())
    {
        // Rows are only re-sorted once everything is in: re-sorting on each
        // arrival would shuffle the list under the user's cursor
        table->redraw();
    }

    if (done)
    {
        scan_.reset();

        // Jobs and Size are the columns whose order depended on what just
        // arrived, so the list is sorted again - but not while the user holds a
        // selection, since selection is by row and re-sorting would move it
        const int sc = table->sortCol();
        bool anything_selected = false;
        for (int r = 0; r < table->rows() && !anything_selected; r++)
            if (table->row_selected(r)) anything_selected = true;

        if ((sc == 2 || sc == 3) && !anything_selected)
        {
            applySort();
            table->syncRowCount();
        }
        table->redraw();
    }
    else
    {
        Fl::add_timeout(0.1, cb_size_poll, this);
    }
}

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

    cleanup_btn = new Fl_Button(bx, btn_y, 110, bh, " Cleanup ");
    cleanup_btn->callback(cb_cleanup, this);
    cleanup_btn->tooltip("Remove intermediate iteration files, keeping the first and "
                         "last iteration of each refinement");
    cleanup_btn->deactivate();
    bx += 110 + pad;

    Fl_Button *close_btn = new Fl_Button(w - pad - 90, btn_y, 90, bh, " Close ");
    close_btn->callback(cb_close, this);

    refresh();
    end();
    resizable(table);
}

void ManageProjectsWindow::refresh()
{
    display_projects_ = pm.getAll();

    std::vector<std::string> to_measure;
    for (auto &p : display_projects_)
    {
        if (p.exists())
        {
            // Shown as "..." until a worker reports the real numbers
            p.jobs = -1;
            p.size_bytes = -1;
            to_measure.push_back(p.path);
        }
        else
        {
            p.jobs = 0;
            p.size_bytes = 0;
        }
    }
    applySort();
    table->syncRowCount();

    name_input->value("");
    path_input->value("");
    updateButtonStates();

    startSizeScan(to_measure);
}

ManageProjectsWindow::~ManageProjectsWindow()
{
    // The timer would otherwise fire into a deleted window, and any worker
    // still running would write into a vector that no longer exists
    cancelSizeScan();
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

void ManageProjectsWindow::cb_cleanup(Fl_Widget *, void *v)
{
    ((ManageProjectsWindow *)v)->cleanupSelected();
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

    if (selected_count > 0)
        cleanup_btn->activate();
    else
        cleanup_btn->deactivate();
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

void ManageProjectsWindow::cleanupSelected()
{
    std::vector<std::string> paths;
    for (int r = 0; r < table->rows(); r++)
    {
        if (!table->row_selected(r)) continue;
        if (r < 0 || r >= (int)display_projects_.size()) continue;
        if (display_projects_[r].exists()) paths.push_back(display_projects_[r].path);
    }
    if (paths.empty()) return;

    // The sizes in the table are now wrong, so re-read them either way
    runIntermediateCleanupDialog(paths);
    refresh();
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
    std::vector<std::string> to_measure;
    for (int r = 0; r < table->rows(); r++)
    {
        if (!table->row_selected(r)) continue;
        if (r < 0 || r >= (int)display_projects_.size()) continue;

        ProjectManager::Project &p = display_projects_[r];
        if (p.exists())
        {
            p.jobs = -1;
            p.size_bytes = -1;
            to_measure.push_back(p.path);
        }
        else
        {
            p.jobs = 0;
            p.size_bytes = 0;
        }
    }
    table->redraw();
    startSizeScan(to_measure);
}
