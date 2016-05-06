#ifndef FSSS_ELASTIC_SMOOTH_HINGE_H_
#define FSSS_ELASTIC_SMOOTH_HINGE_H_

#include "sdm/lib/utils.hpp"
#include "sdm/lib/supervised_data.hpp"

#include <numeric>
#include <chrono>
#include <unordered_set>
#include <random>
#include <queue>

namespace fsss {

using scm_iit = Eigen::SparseMatrix<double, 0, std::ptrdiff_t>::InnerIterator;
using srm_iit = Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator;
using di_pair = std::pair<double, int>;

using u_set = std::unordered_set<int>;
using u_set_pair = std::pair<std::unordered_set<int>, std::unordered_set<int>>;

using sec = std::chrono::seconds;
using mil_sec = std::chrono::milliseconds;
using sys_clk = std::chrono::system_clock;

class elastic_smooth_hinge : public sdm::supervised_data<double, 1> {
public:
  elastic_smooth_hinge(const std::string &libsvm_format,
                       const double &lam = 1e+3, const double &gamma = 0.5,
                       const double &stop_cri = 1e-3,
                       const int &max_iter = 1000, const int &fcg = 1,
                       const double &fsfs3 = 1e-1);
  ~elastic_smooth_hinge();

  int get_num_ins(void) const;
  int get_num_fea(void) const;

  void set_regularized_parameter(const double &lam);
  void set_stop_criterion(const double &eps);

  void set_ssfss_parameter(const int &msi = 2, const int &emsi = 1,
                           const double &fsg = 1e-1, const double &ssg = 1e-4,
                           const double &ser = 0.95);

  void calculate_primal_obj_value(const bool &flag_cal_loss = true);
  void calculate_dual_obj_value(const bool &flag_cal_za = false);
  void calculate_duality_gap(const bool &flag_cal_loss = true,
                             const bool &flag_cal_v = false);

  double get_regularized_parameter(void) const;
  double get_min_regularized_parameter(void) const;
  double get_primal_obj_value(void) const;
  double get_dual_obj_value(void) const;
  double get_duality_gap(void) const;
  double get_stop_criterion(void) const;

  Eigen::VectorXd get_primal_var(void) const;
  Eigen::VectorXd get_dual_var(void) const;

  // for Simultaneous Feature and Sample Safe Screening
  void claer_sfs3_index(void);

  void sfs3(const bool flag_init = true);

  void safe_feature_screen(const bool flag_only = false);
  void initial_safe_feature_screen(const bool flag_only = false);

  void safe_sample_screen(const bool flag_only = false);
  void initial_safe_sample_screen(const bool flag_only = false);

  void inverse_screen(void);
  void further_inverse_screen(void);

  // for plots of screening rate
  std::vector<int> screening_for_plot_inverse(void);

  double lambda_;
  double gamma_;
  double stop_criterion_;
  int itr_;
  int max_iter_;
  int frequency_cal_gap_;
  double frequency_sfs3_;

  double lambda_max_;
  double one_over_n_;
  double R_;

  double primal_obj_value_;
  double dual_obj_value_;
  double duality_gap_;
  double sum_loss_;
  Eigen::VectorXd primal_var_;
  Eigen::VectorXd dual_var_;
  Eigen::VectorXd b_;
  Eigen::VectorXd za_over_n_;
  Eigen::VectorXd spdc_w_;

  Eigen::SparseMatrix<double, 0, std::ptrdiff_t> cm_x_;
  Eigen::VectorXd zj_sq_;
  Eigen::ArrayXd zj_l2norm_;
  Eigen::ArrayXd zi_l2norm_;
  std::vector<int> all_fea_index_;
  std::vector<int> all_ins_index_;

  // for safe screeing
  u_set w_nnz_index_;
  u_set non_sam_screening_index_;
  u_set_pair sam_screening_index_;
  std::vector<int> nss_index_;
  std::vector<int> nfs_index_;

  u_set fea_screen_target_;
  u_set sam_screen_target_;

  double established_w_sq_norm_;
  Eigen::VectorXd unestablished_fea_flag_;

  double dif_alpha_sq_norm_;
  Eigen::VectorXd established_dual_var_;
  Eigen::VectorXd unestablished_dual_var_;
  Eigen::VectorXd unestablished_sam_flag_;

  int num_inv_fea_scr_;
  int num_inv_sam_scr_;
  int num_fea_sav_;
  int num_ins_sav_;
  int num_total_fea_sav_;
  int num_total_ins_sav_;

  int max_ssfss_itr_;
  int each_max_ssfss_itr_;
  double fs_start_gap_;
  double ss_start_gap_;
  double screening_end_rate_;
  bool flag_fs_end_;
  bool flag_ss_end_;

  double fea_dif_time_;
  double sam_dif_time_;
};
}

#endif
