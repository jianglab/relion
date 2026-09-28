/***************************************************************************
 *
 * Author: "Jiang Lab"
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#include "gui_virtualize.h"
#include "src/virtualize_project.h"

#include <FL/Fl.H>
#include <FL/fl_draw.H>
#include <FL/fl_ask.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Window.H>

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

const char INTRO[] =
	"Replaces files that RELION can recompute by small descriptors, in place:\n"
	" - particle stacks of Extract jobs (relion_stacks_to_virtual)\n"
	" - micrographs of MotionCorr jobs made with RELION_VIRTUAL_MOVIE_AVERAGES=real\n"
	"   (relion_movie_averages_to_virtual; older micrographs cannot be regenerated\n"
	"   exactly and are kept)\n"
	"A file is replaced only after it is reproduced bit for bit. Run a dry run first:\n"
	"it reports what would be replaced and how much space it would save.";

struct VirtualizeWindow {
	Fl_Window        *win;
	Fl_Check_Button  *particles, *movies, *float16_fix;
	Fl_Int_Input     *threads;
	Fl_Output        *command;
	Fl_Button        *dry_btn, *convert_btn, *close_btn;
	Fl_Box           *status;
	Fl_Text_Buffer   *buffer;
	Fl_Text_Display  *log_view;

	std::string project, log_path, status_text, command_text, title_text, project_text;
	Fl_Box *project_box;
	pid_t pid;
	long  log_offset;

	explicit VirtualizeWindow(const std::string &dir)
		: project(dir), pid(-1), log_offset(0)
	{
		// The introduction's height depends on the display's font: measure it,
		// and lay everything else out below it
		const int W = 720, M = 15, inner = W - 2 * M;
		fl_font(FL_HELVETICA, 13);
		int tw = inner, th = 0;
		fl_measure(INTRO, tw, th, 0);
		int y = 10 + th + 8;
		const int log_h = 320;
		win = new Fl_Window(W, y + 26 + 26 * 2 + 10 + 22 + 26 + 10 + 30 + 10 + log_h + M,
		                    "Virtualize movie averages and particles");

		Fl_Box *intro = new Fl_Box(M, 10, inner, th, INTRO);
		intro->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_TOP);
		intro->labelfont(FL_HELVETICA);
		intro->labelsize(13);

		project_box = new Fl_Box(M, y, inner, 24, "");
		project_box->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_CLIP);
		project_box->labelfont(FL_HELVETICA_BOLD);
		project_box->labelsize(13);
		y += 26;

		particles = new Fl_Check_Button(M, y, 330, 24, "Particle stacks (Extract jobs)");
		particles->value(1);
		float16_fix = new Fl_Check_Button(360, y, 345, 24, "Accept the RELION 5.0-beta float16 fix");
		float16_fix->tooltip("Stacks written by RELION 5.0 betas hold a few pixels that the old float16 writer "
		                     "doubled. With this, such stacks are converted too, and those pixels come back "
		                     "corrected. Without it, they are kept.");
		y += 26;
		movies = new Fl_Check_Button(M, y, 330, 24, "Movie averages (MotionCorr jobs)");
		movies->value(1);
		threads = new Fl_Int_Input(460, y, 60, 24, "Threads:");
		threads->value("8");
		for (Fl_Widget *w : {(Fl_Widget *)particles, (Fl_Widget *)movies, (Fl_Widget *)float16_fix, (Fl_Widget *)threads})
		{
			w->callback(cb_options, this);
			w->when(FL_WHEN_CHANGED);
		}
		y += 26 + 10 + 22;   // room for the label above the command line

		command = new Fl_Output(M, y, inner, 26, "Command line, to run on a cluster node instead:");
		command->align(FL_ALIGN_TOP_LEFT);
		command->textsize(12);
		y += 26 + 10;

		dry_btn = new Fl_Button(M, y, 110, 30, "Dry run");
		dry_btn->callback(cb_dry, this);
		convert_btn = new Fl_Button(M + 120, y, 110, 30, "Convert...");
		convert_btn->callback(cb_convert, this);
		status = new Fl_Box(M + 240, y, inner - 240 - 80, 30, "");
		status->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_CLIP);
		close_btn = new Fl_Button(W - M - 70, y, 70, 30, "Close");
		close_btn->callback(cb_close, this);
		y += 30 + 10;

		buffer = new Fl_Text_Buffer();
		log_view = new Fl_Text_Display(M, y, inner, log_h);
		log_view->buffer(buffer);
		log_view->textfont(FL_COURIER);
		log_view->textsize(12);

		win->callback(cb_close, this);
		win->resizable(log_view);
		win->end();
		setProject(dir);
	}

	void showAs(bool modal)
	{
		// Modality can only change while hidden
		if (win->shown() && (win->modal() != 0) != modal) win->hide();
		if (modal) win->set_modal(); else win->set_non_modal();
		win->show();
	}

	bool running() const { return pid >= 0; }

	/// Point the window at a project (only while nothing is running)
	void setProject(const std::string &dir)
	{
		project = dir;
		project_text = "Project: " + dir;
		project_box->label(project_text.c_str());
		title_text = "Virtualize: " + dir;
		win->label(title_text.c_str());
		if (!log_path.empty() && log_path.compare(0, dir.size() + 1, dir + "/") != 0)
		{
			buffer->text("");   // the previous project's log
			log_path.clear();
			setStatus("");
		}
		updateCommand();
		win->redraw();
	}

	relion_virtualize::Options options(bool convert) const
	{
		relion_virtualize::Options o;
		o.particles = particles->value();
		o.movie_averages = movies->value();
		o.accept_float16_fix = float16_fix->value();
		o.convert = convert;
		o.threads = atoi(threads->value());
		return o;
	}

	void updateCommand()
	{
		const std::vector<std::string> c = relion_virtualize::commands(options(false), project);
		command_text.clear();
		for (size_t i = 0; i < c.size(); i++)
			command_text += (i ? " ; " : "") + c[i];
		command->value(command_text.empty() ? "(nothing selected)" : command_text.c_str());
		const bool can_run = !c.empty() && pid < 0;
		if (can_run) { dry_btn->activate(); convert_btn->activate(); }
		else { dry_btn->deactivate(); convert_btn->deactivate(); }
	}

	void setStatus(const std::string &s)
	{
		status_text = s;
		status->label(status_text.c_str());
		status->redraw();
	}

	void start(bool convert)
	{
		const relion_virtualize::Options o = options(convert);
		if (relion_virtualize::commands(o, project).empty()) return;
		log_path = relion_virtualize::logPath(project, convert);

		const int fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (fd < 0)
		{
			fl_alert("Cannot write the log %s.", log_path.c_str());
			return;
		}
		const std::string sh = relion_virtualize::script(o, project);
		const pid_t child = fork();
		if (child < 0)
		{
			close(fd);
			fl_alert("Cannot start the conversion.");
			return;
		}
		if (child == 0)
		{
			// Its own session: closing the window or the GUI does not stop it
			setsid();
			const int devnull = open("/dev/null", O_RDONLY);
			if (devnull >= 0) dup2(devnull, 0);
			dup2(fd, 1);
			dup2(fd, 2);
			execl("/bin/sh", "sh", "-c", sh.c_str(), (char *)NULL);
			_exit(127);
		}
		close(fd);
		pid = child;
		log_offset = 0;
		buffer->text("");
		updateCommand();
		setStatus(std::string(convert ? "Converting" : "Dry run") + " running; log: " + logName());
		Fl::add_timeout(1.0, cb_poll, this);
	}

	/// Append what the log gained since last time; notice the process ending.
	void poll()
	{
		FILE *f = fopen(log_path.c_str(), "r");
		if (f != NULL)
		{
			fseek(f, log_offset, SEEK_SET);
			char chunk[8192];
			size_t n;
			std::string add;
			while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) add.append(chunk, n);
			log_offset = ftell(f);
			fclose(f);
			// Progress bars rewrite their line with '\r': keep the last state only
			std::string clean;
			for (size_t i = 0; i < add.size(); i++)
			{
				if (add[i] == '\r')
				{
					const size_t nl = clean.find_last_of('\n');
					clean.erase(nl == std::string::npos ? 0 : nl + 1);
				}
				else clean += add[i];
			}
			if (!clean.empty())
			{
				buffer->append(clean.c_str());
				log_view->insert_position(buffer->length());
				log_view->show_insert_position();
			}
		}

		int st = 0;
		if (waitpid(pid, &st, WNOHANG) == pid)
		{
			pid = -1;
			// Skipped jobs keep their files; make that impossible to miss
			char *all = buffer->text();
			const std::string text = all ? all : "";
			free(all);
			const bool warned = text.find(" WARNING: ") != std::string::npos ||
			                    text.find("DIFFERENT") != std::string::npos ||
			                    text.find("[exit 0]") == std::string::npos;
			setStatus(std::string(warned ? "Finished WITH WARNINGS (some files kept; see the log below): "
			                             : "Finished; log: ") + logName());
			if (warned) status->labelcolor(FL_RED); else status->labelcolor(FL_FOREGROUND_COLOR);
			updateCommand();
			return;
		}
		Fl::repeat_timeout(1.0, cb_poll, this);
	}

	std::string logName() const { return log_path.substr(log_path.find_last_of('/') + 1); }

	static void cb_options(Fl_Widget *, void *v) { ((VirtualizeWindow *)v)->updateCommand(); }
	static void cb_poll(void *v) { ((VirtualizeWindow *)v)->poll(); }
	static void cb_dry(Fl_Widget *, void *v) { ((VirtualizeWindow *)v)->start(false); }

	static void cb_convert(Fl_Widget *, void *v)
	{
		VirtualizeWindow *w = (VirtualizeWindow *)v;
		const int ret = fl_choice(
			"Replace the verified files in this project by descriptors?\n\n"
			" - Only RELION builds with virtual data support can then read the project;\n"
			"   older builds fail with \"unsupported MRC mode\", and external programs\n"
			"   (cryoSPARC, crYOLO, ...) cannot read descriptors at all.\n"
			" - Virtual particles need their micrographs, and virtual micrographs need\n"
			"   their movies: keep those safe - they become the only copy of the data.\n"
			" - Files that are not reproduced bit for bit are kept.\n\n"
			"This reads every stack and regenerates every micrograph it replaces, which can\n"
			"take hours for a large project; it runs in the background.",
			"Cancel", "Convert", NULL);
		if (ret == 1) w->start(true);
	}

	static void cb_close(Fl_Widget *, void *v)
	{
		VirtualizeWindow *w = (VirtualizeWindow *)v;
		if (w->pid >= 0)
		{
			const int ret = fl_choice("The conversion is still running. It will carry on in the background\n"
			                          "after this window closes; its log is %s.",
			                          "Keep window", "Close window", NULL, w->log_path.c_str());
			if (ret != 1) return;
		}
		// Hidden, not destroyed: polling (and reaping the process) carries on,
		// and the menu brings the same window back
		w->win->hide();
	}
};

} // namespace

void runVirtualizeDialog(const std::string &project_dir, bool modal)
{
	// One window, and so one conversion at a time, for the life of the GUI
	static VirtualizeWindow *instance = NULL;
	if (instance == NULL)
	{
		instance = new VirtualizeWindow(project_dir);
		instance->showAs(modal);
		if (modal) while (instance->win->shown()) Fl::wait();
		return;
	}
	if (instance->running() && instance->project != project_dir)
		fl_message("A conversion of %s is still running.\n"
		           "Only one conversion runs at a time; this window shows its progress.",
		           instance->project.c_str());
	else if (!instance->running())
		instance->setProject(project_dir);
	instance->showAs(modal);
	// A modal window's caller waits for it, as fl_choice() does; the
	// conversion itself carries on after it closes
	if (modal) while (instance->win->shown()) Fl::wait();
}
