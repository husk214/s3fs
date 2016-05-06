#ifndef FSSS_ELASTIC_SOFT_INSENSITIVE_PSDCA_H_
#define FSSS_ELASTIC_SOFT_INSENSITIVE_PSDCA_H_

#include "elastic_soft_insensitive.h"
#include <unordered_map>

namespace fsss {

template <typename _Tp> inline _Tp sign(_Tp val) {
  return 1.0 - (val <= 0.0) - (val < 0.0);
}
class elastic_soft_insensitive_psdca : public elastic_soft_insensitive {
public:
  elastic_soft_insensitive_psdca(const std::string &libsvm_format,
                                const double &lam = 1.0,
                                const double &eps = 1.0,
                                const double &gamma = 0.5,
                                const double &stop_cri = 1e-3,
                                const int &max_iter = 1000, const int &fcg = 1);
  ~elastic_soft_insensitive_psdca();

  void set_regularized_parameter(const double &lam);
  void init_w_za_over_n(void);

  void calculate_dual_obj_value(const bool &flag_cal_v = false);
  void calculate_duality_gap(const bool &flag_cal_loss, const bool &flag_cal_v);
  void train(void);

  // for sfs3
  void train_sfs3(const bool dynamic = true);
  void train_fs(const bool dynamic = true);
  void train_ss(const bool dynamic = true);


private:

  const double one_over_n_;

  Eigen::VectorXd v_;
  Eigen::ArrayXd zi_sq_;
};
}

#endif
