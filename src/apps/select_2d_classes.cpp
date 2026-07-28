/***************************************************************************
 *
 * Author: Jiang Lab
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 ***************************************************************************/

// Protect against FLTK's Complex symbol.
#define Complex tmpComplex
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Spinner.H>
#include <FL/fl_ask.H>
#include <FL/fl_draw.H>
#undef Complex

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "src/args.h"
#include "src/funcs.h"
#include "src/image.h"
#include "src/metadata_table.h"
#include "src/jaz/single_particle/obs_model.h"
#include "src/macros.h"
#include "src/select_2d_classes_utils.h"

namespace
{

const int THUMBNAIL_SIZE = 128;
const int TILE_WIDTH = THUMBNAIL_SIZE + 16;
const int TILE_HEIGHT = THUMBNAIL_SIZE + 46;
const int TOOLBAR_HEIGHT = 92;
const int STATUS_WIDTH = 400;

Fl_Color typeColor(int type_id)
{
	static const Fl_Color colors[] = {
		FL_RED, FL_GREEN, FL_BLUE, FL_CYAN, FL_MAGENTA, FL_YELLOW,
		fl_rgb_color(255, 128, 0), fl_rgb_color(128, 64, 255),
		fl_rgb_color(0, 180, 120), fl_rgb_color(255, 80, 160)
	};
	const int nr_colors = sizeof(colors) / sizeof(colors[0]);
	return type_id > 0 ? colors[(type_id - 1) % nr_colors] : FL_DARK3;
}

struct ClassInfo
{
	int class_number;
	int type_id;
	long int metadata_index;
	RFLOAT population;
	std::vector<unsigned char> pixels;
	std::vector<RFLOAT> feature;
};

class Select2DWindow;

class ClassTile : public Fl_Widget
{
public:
	ClassTile(int X, int Y, ClassInfo *info, Select2DWindow *owner)
		: Fl_Widget(X, Y, TILE_WIDTH, TILE_HEIGHT), info_(info), owner_(owner)
	{}

	void draw();
	int handle(int event);

private:
	ClassInfo *info_;
	Select2DWindow *owner_;
};

class Select2DProgram
{
public:
	IOParser parser;
	FileName fn_optimiser, output_dir, fn_model, fn_data;
	MetaDataTable MDclasses, MDdata;
	ObservationModel obs_model;
	std::vector<ClassInfo> classes;
	std::vector<int> original_order, similarity_order;
	int verb;

	Select2DProgram() : verb(1) {}

	void read(int argc, char **argv)
	{
		parser.setCommandLine(argc, argv);
		parser.addSection("General options");
		fn_optimiser = parser.getOption("--i", "Input optimiser STAR file from 2D classification");
		output_dir = parser.getOption("--o", "Output directory");
		verb = textToInteger(parser.getOption("--verb", "Verbosity", "1"));
		if (parser.checkForErrors(verb))
			REPORT_ERROR("Errors encountered on the command line.");
	}

	void initialise()
	{
		if (!fn_optimiser.contains("_optimiser.star"))
			REPORT_ERROR("Select 2D classes requires a *_optimiser.star file.");
		if (output_dir == "")
			REPORT_ERROR("Select 2D classes requires an output directory.");
		if (output_dir[output_dir.length() - 1] != '/')
			output_dir += "/";
		mktree(output_dir);

		MetaDataTable MDoptimiser;
		MDoptimiser.read(fn_optimiser, "optimiser_general");
		if (!MDoptimiser.containsLabel(EMDL_OPTIMISER_MODEL_STARFILE) ||
			!MDoptimiser.containsLabel(EMDL_OPTIMISER_DATA_STARFILE))
			REPORT_ERROR("The optimiser STAR file must contain rlnModelStarFile and rlnExperimentalDataStarFile.");
		if (!MDoptimiser.getValue(EMDL_OPTIMISER_MODEL_STARFILE, fn_model) ||
			!MDoptimiser.getValue(EMDL_OPTIMISER_DATA_STARFILE, fn_data) ||
			fn_model == "" || fn_data == "")
			REPORT_ERROR("The optimiser STAR file does not define valid model and data STAR filenames.");

		MetaDataTable MDgeneral;
		MDgeneral.read(fn_model, "model_general");
		if (!MDgeneral.containsLabel(EMDL_MLMODEL_DIMENSIONALITY))
			REPORT_ERROR("The model STAR file does not contain rlnReferenceDimensionality.");
		int dimensionality;
		MDgeneral.getValue(EMDL_MLMODEL_DIMENSIONALITY, dimensionality);
		if (dimensionality != 2)
			REPORT_ERROR("Select 2D classes only accepts a 2D classification optimiser STAR file.");

		MDclasses.read(fn_model, "model_classes");
		if (!MDclasses.containsLabel(EMDL_MLMODEL_REF_IMAGE))
			REPORT_ERROR("The model STAR file does not contain 2D class-average images.");

		ObservationModel::loadSafely(fn_data, obs_model, MDdata);
		if (!MDdata.containsLabel(EMDL_MICROGRAPH_NAME) ||
			!MDdata.containsLabel(EMDL_PARTICLE_HELICAL_TUBE_ID) ||
			!MDdata.containsLabel(EMDL_PARTICLE_CLASS))
			REPORT_ERROR("The data STAR file must contain rlnMicrographName, rlnHelicalTubeID and rlnClassNumber.");

		loadClasses();
		computeSimilarityOrder();
		loadSavedAssignments();
	}

	void loadClasses()
	{
		classes.clear();
		original_order.clear();
		const long int nr_classes = MDclasses.numberOfObjects();
		if (nr_classes < 1)
			REPORT_ERROR("The model STAR file does not contain any 2D classes.");
		classes.reserve(nr_classes);
		original_order.reserve(nr_classes);

		for (long int index = 0; index < nr_classes; index++)
		{
			FileName fn_image;
			MDclasses.getValue(EMDL_MLMODEL_REF_IMAGE, fn_image, index);
			Image<RFLOAT> image;
			image.read(fn_image);
			if (ZSIZE(image()) != 1)
				REPORT_ERROR("Select 2D classes encountered a non-2D class average.");

			ClassInfo info;
			info.class_number = index + 1;
			info.type_id = 0;
			info.metadata_index = index;
			info.population = 0.;
			if (MDclasses.containsLabel(EMDL_MLMODEL_PDF_CLASS))
				MDclasses.getValue(EMDL_MLMODEL_PDF_CLASS, info.population, index);

			makeFeature(image(), info.feature);
			makeThumbnail(image(), info.pixels);
			classes.push_back(info);
			original_order.push_back(index);
		}
	}

	void makeFeature(const MultidimArray<RFLOAT> &input, std::vector<RFLOAT> &feature)
	{
		MultidimArray<RFLOAT> image = input;
		selfScaleToSize(image, 48, 48);

		RFLOAT mean = 0.;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(image)
			mean += DIRECT_MULTIDIM_ELEM(image, n);
		mean /= MULTIDIM_SIZE(image);

		RFLOAT norm = 0.;
		feature.resize(MULTIDIM_SIZE(image));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(image)
		{
			const RFLOAT value = DIRECT_MULTIDIM_ELEM(image, n) - mean;
			feature[n] = value;
			norm += value * value;
		}
		norm = sqrt(norm);
		if (norm > 0.)
			for (long int n = 0; n < feature.size(); n++)
				feature[n] /= norm;
	}

	void makeThumbnail(const MultidimArray<RFLOAT> &input, std::vector<unsigned char> &pixels)
	{
		MultidimArray<RFLOAT> image = input;
		selfScaleToSize(image, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
		RFLOAT minval = 0., maxval = 0., sigma_contrast = 0.;
		getImageContrast(image, minval, maxval, sigma_contrast);
		const RFLOAT range = maxval > minval ? maxval - minval : 1.;
		pixels.resize(THUMBNAIL_SIZE * THUMBNAIL_SIZE);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(image)
		{
			RFLOAT value = (DIRECT_MULTIDIM_ELEM(image, n) - minval) / range;
			value = XMIPP_MAX(0., XMIPP_MIN(1., value));
			pixels[n] = static_cast<unsigned char>(ROUND(255. * value));
		}
	}

	void computeSimilarityOrder()
	{
		std::vector<std::vector<RFLOAT> > features;
		std::vector<RFLOAT> populations;
		features.reserve(classes.size());
		populations.reserve(classes.size());
		for (size_t index = 0; index < classes.size(); index++)
		{
			features.push_back(classes[index].feature);
			populations.push_back(classes[index].population);
		}
		similarity_order = Select2DClasses::similarityOrder(features, populations);
	}

	FileName assignmentsFilename() const
	{
		return output_dir + "class_type_assignments.star";
	}

	void loadSavedAssignments()
	{
		const FileName filename = assignmentsFilename();
		if (!exists(filename))
			return;
		MetaDataTable saved;
		saved.read(filename);
		if (!saved.containsLabel(EMDL_PARTICLE_CLASS) ||
			!saved.containsLabel(EMDL_PARTICLE_SELECTION_TYPE))
			return;

		FOR_ALL_OBJECTS_IN_METADATA_TABLE(saved)
		{
			int class_number, type_id;
			saved.getValue(EMDL_PARTICLE_CLASS, class_number);
			saved.getValue(EMDL_PARTICLE_SELECTION_TYPE, type_id);
			if (class_number >= 1 && class_number <= classes.size())
				classes[class_number - 1].type_id =
						Select2DClasses::validTypeOrJunk(
								type_id, Select2DClasses::MAX_NR_TYPES);
		}
	}

	void saveAssignments()
	{
		MetaDataTable output;
		output.setName("class_type_assignments");
		for (int index = 0; index < classes.size(); index++)
		{
			output.addObject(MDclasses.getObject(classes[index].metadata_index));
			output.setValue(EMDL_PARTICLE_CLASS, classes[index].class_number);
			output.setValue(EMDL_PARTICLE_SELECTION_TYPE, classes[index].type_id);
		}
		output.write(assignmentsFilename());
	}

	void voteAndWrite(int nr_types)
	{
		saveAssignments();

		std::map<int, int> class_to_type;
		for (int index = 0; index < classes.size(); index++)
			class_to_type[classes[index].class_number] =
					Select2DClasses::validTypeOrJunk(classes[index].type_id, nr_types);

		std::vector<std::string> micrographs;
		std::vector<int> tube_ids, assigned_types;
		micrographs.reserve(MDdata.numberOfObjects());
		tube_ids.reserve(MDdata.numberOfObjects());
		assigned_types.reserve(MDdata.numberOfObjects());
		for (long int particle = 0; particle < MDdata.numberOfObjects(); particle++)
		{
			FileName micrograph;
			int tube_id, class_number;
			MDdata.getValue(EMDL_MICROGRAPH_NAME, micrograph, particle);
			MDdata.getValue(EMDL_PARTICLE_HELICAL_TUBE_ID, tube_id, particle);
			MDdata.getValue(EMDL_PARTICLE_CLASS, class_number, particle);
			int type_id = 0;
			if (class_to_type.find(class_number) != class_to_type.end())
				type_id = class_to_type[class_number];
			micrographs.push_back(micrograph);
			tube_ids.push_back(tube_id);
			assigned_types.push_back(type_id);
		}

		const Select2DClasses::FilamentVoteResult votes =
				Select2DClasses::voteFilaments(
						micrographs, tube_ids, assigned_types, nr_types);

		std::vector<MetaDataTable> outputs(nr_types + 1);
		for (int type_id = 0; type_id <= nr_types; type_id++)
		{
			outputs[type_id].addMissingLabels(&MDdata);
			outputs[type_id].setName("particles");
		}

		for (long int particle = 0; particle < MDdata.numberOfObjects(); particle++)
		{
			const int winner = votes.particle_types[particle];
			outputs[winner].addObject(MDdata.getObject(particle));
			outputs[winner].setValue(EMDL_PARTICLE_SELECTION_TYPE, winner);
		}

		MetaDataTable output_nodes;
		output_nodes.setName("output_nodes");
		for (int type_id = 0; type_id <= nr_types; type_id++)
		{
			FileName filename;
			if (type_id == 0)
				filename = output_dir + "particles_junk.star";
			else
				filename = output_dir + "particles_type" + integerToString(type_id, 3) + ".star";

			if (obs_model.opticsMdt.numberOfObjects() > 0)
				obs_model.save(outputs[type_id], filename, "particles");
			else
				outputs[type_id].write(filename);

			output_nodes.addObject();
			output_nodes.setValue(EMDL_PIPELINE_NODE_NAME, filename);
			output_nodes.setValue(EMDL_PIPELINE_NODE_TYPE_LABEL,
					std::string("ParticleGroupMetadata.star.relion"));
			std::cout << "Saved " << outputs[type_id].numberOfObjects() << " particles from "
					  << votes.filament_counts[type_id] << " filaments to " << filename << std::endl;
		}

		output_nodes.addObject();
		output_nodes.setValue(EMDL_PIPELINE_NODE_NAME, assignmentsFilename());
		output_nodes.setValue(EMDL_PIPELINE_NODE_TYPE_LABEL,
				std::string("Image2DGroupMetadata.star.relion.classaverages"));
		output_nodes.write(output_dir + "RELION_OUTPUT_NODES.star");
	}
};

class Select2DWindow : public Fl_Double_Window
{
public:
	Select2DProgram *program;
	Fl_Spinner *nr_types_spinner;
	Fl_Choice *current_type_choice;
	Fl_Check_Button *similarity_button;
	Fl_Button *junk_button, *save_button, *finish_button;
	Fl_Box *status_box;
	Fl_Scroll *scroll;
	std::vector<ClassTile*> tiles;
	std::vector<int> display_order;
	int current_type;
	bool finished;

	Select2DWindow(int W, int H, Select2DProgram *program)
		: Fl_Double_Window(W, H, "RELION Select 2D classes"),
		  program(program), current_type(1), finished(false)
	{
		begin();

		Fl_Box *types_label = new Fl_Box(12, 10, 112, 25, "Number of types:");
		types_label->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
		nr_types_spinner = new Fl_Spinner(125, 10, 62, 25);
		nr_types_spinner->range(1, Select2DClasses::MAX_NR_TYPES);
		nr_types_spinner->step(1);
		nr_types_spinner->type(FL_INT_INPUT);

		Fl_Box *current_label = new Fl_Box(210, 10, 92, 25, "Current type:");
		current_label->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
		current_type_choice = new Fl_Choice(302, 10, 115, 25);

		similarity_button = new Fl_Check_Button(450, 10, 300, 25,
				"Group/sort by image similarity");

		junk_button = new Fl_Button(12, 48, 125, 28, "Set all junk");
		save_button = new Fl_Button(150, 48, 145, 28, "Save assignments");
		save_button->tooltip(
				"Save the current class-to-type assignments as a checkpoint. "
				"This does not run filament voting or finish the job.");
		finish_button = new Fl_Button(310, 48, 170, 28, "Finish and vote");
		finish_button->color(fl_rgb_color(100, 178, 178));
		status_box = new Fl_Box(
				XMIPP_MAX(500, W - STATUS_WIDTH - 12), 48,
				XMIPP_MIN(STATUS_WIDTH, W - 512), 28, "");
		status_box->align(FL_ALIGN_RIGHT | FL_ALIGN_INSIDE);

		scroll = new Fl_Scroll(0, TOOLBAR_HEIGHT, W, H - TOOLBAR_HEIGHT);
		scroll->type(Fl_Scroll::VERTICAL_ALWAYS);
		scroll->begin();
		for (int index = 0; index < program->classes.size(); index++)
			tiles.push_back(new ClassTile(0, 0, &program->classes[index], this));
		scroll->end();

		end();
		resizable(scroll);

		int initial_types = Select2DClasses::DEFAULT_NR_TYPES;
		for (int index = 0; index < program->classes.size(); index++)
			initial_types = XMIPP_MAX(initial_types, program->classes[index].type_id);
		initial_types = XMIPP_MIN(initial_types, Select2DClasses::MAX_NR_TYPES);
		nr_types_spinner->value(initial_types);
		rebuildTypeChoice(initial_types);

		display_order = program->original_order;
		reflow();
		updateStatus();

		nr_types_spinner->callback(cbNrTypes, this);
		nr_types_spinner->when(FL_WHEN_CHANGED);
		current_type_choice->callback(cbCurrentType, this);
		similarity_button->callback(cbSimilarity, this);
		junk_button->callback(cbJunk, this);
		save_button->callback(cbSave, this);
		finish_button->callback(cbFinish, this);
		callback(cbClose, this);
	}

	void assignClass(int index, bool junk)
	{
		ClassInfo &info = program->classes[index];
		if (junk)
			info.type_id = 0;
		else
			info.type_id = (info.type_id == current_type) ? 0 : current_type;
		tiles[index]->redraw();
		updateStatus();
	}

	int nrTypes() const
	{
		return ROUND(nr_types_spinner->value());
	}

	void changeNrTypes()
	{
		const int nr_types = nrTypes();
		for (int index = 0; index < program->classes.size(); index++)
		{
			program->classes[index].type_id =
					Select2DClasses::validTypeOrJunk(
							program->classes[index].type_id, nr_types);
			tiles[index]->redraw();
		}
		if (current_type > nr_types)
			current_type = nr_types;
		rebuildTypeChoice(nr_types);
		updateStatus();
	}

	void rebuildTypeChoice(int nr_types)
	{
		current_type_choice->clear();
		for (int type_id = 1; type_id <= nr_types; type_id++)
		{
			std::string label = "Type " + integerToString(type_id);
			current_type_choice->add(label.c_str());
		}
		current_type = XMIPP_MAX(1, XMIPP_MIN(current_type, nr_types));
		current_type_choice->value(current_type - 1);
		current_type_choice->redraw();
	}

	void changeCurrentType()
	{
		current_type = current_type_choice->value() + 1;
		updateStatus();
	}

	void changeSimilarity()
	{
		if (similarity_button->value())
		{
			display_order = program->similarity_order;
		}
		else
		{
			display_order = program->original_order;
		}
		reflow();
	}

	void setAllJunk()
	{
		for (int index = 0; index < program->classes.size(); index++)
		{
			program->classes[index].type_id = 0;
			tiles[index]->redraw();
		}
		updateStatus();
	}

	void saveAssignments()
	{
		try
		{
			program->saveAssignments();
			updateStatus("Assignments saved.");
		}
		catch (RelionError error)
		{
			fl_alert("%s", error.msg.c_str());
		}
	}

	void finishAndVote()
	{
		try
		{
			program->voteAndWrite(nrTypes());
			finished = true;
			hide();
		}
		catch (RelionError error)
		{
			fl_alert("%s", error.msg.c_str());
		}
	}

	void updateStatus(const std::string &prefix = "")
	{
		std::vector<int> counts(nrTypes() + 1, 0);
		for (int index = 0; index < program->classes.size(); index++)
		{
			const int type_id = program->classes[index].type_id;
			if (type_id >= 0 && type_id < counts.size())
				counts[type_id]++;
		}
		std::string text = prefix;
		if (text != "")
			text += "  ";
		text += "Current: type " + integerToString(current_type) +
				" | junk classes: " + integerToString(counts[0]);
		status_box->copy_label(text.c_str());
		status_box->redraw();
	}

	void reflow()
	{
		const int usable_width = XMIPP_MAX(TILE_WIDTH, scroll->w() - 20);
		const int columns = XMIPP_MAX(1, usable_width / TILE_WIDTH);
		for (int position = 0; position < display_order.size(); position++)
		{
			const int index = display_order[position];
			const int column = position % columns;
			const int row = position / columns;
			tiles[index]->resize(column * TILE_WIDTH, row * TILE_HEIGHT,
								TILE_WIDTH, TILE_HEIGHT);
		}
		scroll->redraw();
	}

	void resize(int X, int Y, int W, int H)
	{
		Fl_Double_Window::resize(X, Y, W, H);
		scroll->resize(0, TOOLBAR_HEIGHT, W, H - TOOLBAR_HEIGHT);
		const int status_x = XMIPP_MAX(500, W - STATUS_WIDTH - 12);
		status_box->resize(
				status_x, 48, XMIPP_MAX(50, W - status_x - 12), 28);
		reflow();
	}

	static void cbNrTypes(Fl_Widget*, void *data)
	{ static_cast<Select2DWindow*>(data)->changeNrTypes(); }
	static void cbCurrentType(Fl_Widget*, void *data)
	{ static_cast<Select2DWindow*>(data)->changeCurrentType(); }
	static void cbSimilarity(Fl_Widget*, void *data)
	{ static_cast<Select2DWindow*>(data)->changeSimilarity(); }
	static void cbJunk(Fl_Widget*, void *data)
	{ static_cast<Select2DWindow*>(data)->setAllJunk(); }
	static void cbSave(Fl_Widget*, void *data)
	{ static_cast<Select2DWindow*>(data)->saveAssignments(); }
	static void cbFinish(Fl_Widget*, void *data)
	{ static_cast<Select2DWindow*>(data)->finishAndVote(); }
	static void cbClose(Fl_Widget*, void *data)
	{
		Select2DWindow *window = static_cast<Select2DWindow*>(data);
		if (window->finished ||
			fl_choice("Close without running filament voting?", "Cancel", "Close", NULL) == 1)
			window->hide();
	}
};

void ClassTile::draw()
{
	fl_push_clip(x(), y(), w(), h());
	fl_color(FL_BLACK);
	fl_rectf(x(), y(), w(), h());

	fl_color(FL_WHITE);
	fl_font(FL_HELVETICA_BOLD, 13);
	const std::string class_label = "Class " + integerToString(info_->class_number);
	fl_draw(class_label.c_str(), x() + 5, y() + 16);

	fl_draw_image(info_->pixels.data(), x() + 8, y() + 22,
				  THUMBNAIL_SIZE, THUMBNAIL_SIZE, 1);

	const Fl_Color color = typeColor(info_->type_id);
	fl_color(color);
	fl_line_style(FL_SOLID, 3);
	fl_rect(x() + 6, y() + 20, THUMBNAIL_SIZE + 4, THUMBNAIL_SIZE + 4);
	fl_line_style(0);

	fl_font(FL_HELVETICA_BOLD, 13);
	const std::string type_label = info_->type_id == 0 ?
			"Junk" : "Type " + integerToString(info_->type_id);
	fl_draw(type_label.c_str(), x() + 7, y() + TILE_HEIGHT - 8);
	fl_pop_clip();
}

int ClassTile::handle(int event)
{
	if (event == FL_PUSH)
	{
		if (Fl::event_button() == FL_LEFT_MOUSE)
		{
			owner_->assignClass(info_->class_number - 1, false);
			return 1;
		}
		if (Fl::event_button() == FL_RIGHT_MOUSE)
		{
			owner_->assignClass(info_->class_number - 1, true);
			return 1;
		}
	}
	return Fl_Widget::handle(event);
}

} // namespace

int main(int argc, char **argv)
{
	Select2DProgram program;
	try
	{
		program.read(argc, argv);
		program.initialise();
		Fl::scheme("gtk+");
		Fl::visual(FL_RGB);
		Select2DWindow window(920, 700, &program);
		window.show(argc, argv);
		Fl::run();
		return window.finished ? RELION_EXIT_SUCCESS : RELION_EXIT_FAILURE;
	}
	catch (RelionError error)
	{
		std::cerr << error;
		return RELION_EXIT_FAILURE;
	}
}
