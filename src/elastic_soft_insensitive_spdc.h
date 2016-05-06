#ifndef FSSS_ELASTIC_SOFT_INSENSITIVE_SPDC_H_
#define FSSS_ELASTIC_SOFT_INSENSITIVE_SPDC_H_

#include "elastic_soft_insensitive.h"
#include <unordered_map>

namespace fsss {

using um_citr = std::unordered_map<int, double>::const_iterator;

class elastic_soft_insensitive_spdc : public elastic_soft_insensitive {
public:
  elastic_soft_insensitive_spdc(const std::string &libsvm_format,
                                const double &lam = 1.0,
                                const double &eps = 1.0,
                                const double &gamma = 0.5,
                                const double &stop_cri = 1e-3,
                                const int &max_iter = 1000, const int &fcg = 1);
  ~elastic_soft_insensitive_spdc();

  void set_regularized_parameter(const double &lam);
  void init_za_over_n(void);

  void calculate_dual_obj_value(const bool &flag_cal_v = false);
  void calculate_duality_gap(const bool &flag_cal_loss, const bool &flag_cal_v);
  void train(void);

  // for sfs3
  void train_sfs3(const bool dynamic = true);
  void train_fs(const bool dynamic = true);
  void train_ss(const bool dynamic = true);

  // // for plot screening rate
  void train_for_plot_inverse(void);

private:
  double tau_;
  double sigma_;
  double theta_;

  std::vector<std::unordered_map<int, double>> umx_;

  Eigen::VectorXd u_;
  Eigen::VectorXd v_;
  Eigen::VectorXd w_;
};
}

#endif
