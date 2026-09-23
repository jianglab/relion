#ifndef SRC_CLASS2D_CONSENSUS_NMF_H_
#define SRC_CLASS2D_CONSENSUS_NMF_H_

#include <vector>

// Internal numerical primitives, also exercised against dense reference math.
namespace class2d_nmf_detail
{
void projectSimplex(std::vector<double> &values, std::vector<double> &scratch);
void gramMatrix(const std::vector<double> &h, int components, int features,
                std::vector<double> &gram);
// Returns half the squared residual; gradient is with respect to membership.
double patternObjectiveGradient(const std::vector<int> &labels, int source_classes,
    const std::vector<double> &h, const std::vector<double> &gram,
    const double *membership, int components, std::vector<double> &gradient);
double profileGradient(const std::vector<double> &h, const std::vector<double> &s,
    const std::vector<double> &t, int components, int features, int component, int feature);
}

#endif
