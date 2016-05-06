#include "elastic_soft_insensitive.h"

namespace fsss {

elastic_soft_insensitive::elastic_soft_insensitive(
    const std::string &libsvm_format, const double &lam, const double &eps,
    const double &gamma, const double &stop_cri, const int &max_iter,
    const int &fcg, const double &fsfs3)
    : sdm::supervised_data_regression<double, 1>(libsvm_format), lambda_(lam),
      epsilon_(eps), gamma_(gamma), stop_criterion_(stop_cri), itr_(0),
      max_iter_(max_iter), frequency_cal_gap_(fcg), frequency_sfs3_(fsfs3),
      R_(0.0), primal_obj_value_(0.0), dual_obj_value_(0.0), cm_x_(x_) {
  primal_var_ = Eigen::VectorXd::Zero(num_fea_);
  spdc_w_ = primal_var_;
  dual_var_ = Eigen::VectorXd::Constant(num_ins_, 1.0);
  for (int i = 0; i < num_ins_; ++i)
    if (y_[i] < 0.0)
      dual_var_[i] = -1.0;

  b_ = dual_var_;
  one_over_n_ = 1.0 / static_cast<double>(num_ins_);

  int it_index = 0;
  double it_value = 0.0;
  double sum_j = 0.0;
  double max_j_abs_sum_xi = 0.0;
  double sum_j_sq = 0.0;
  zj_sq_.resize(num_fea_);
  zj_l2norm_.resize(num_fea_);
  for (int j = 0; j < num_fea_; ++j) {
    for (scm_iit it(cm_x_, j); it; ++it) {
      it_index = it.index();
      it_value = it.value();
      cm_x_.coeffRef(it_index, j) = it_value;
      sum_j += (it_value);
      sum_j_sq += it_value * it_value;
    }
    max_j_abs_sum_xi = std::max(max_j_abs_sum_xi, std::abs(sum_j));
    zj_sq_[j] = sum_j_sq;
    zj_l2norm_[j] = std::sqrt(sum_j_sq);
    sum_j = 0.0;
    sum_j_sq = 0.0;
  }

  zi_l2norm_.resize(num_ins_);
  for (int i = 0; i < num_ins_; ++i) {
    zi_l2norm_[i] = (x_.row(i)).norm();
    R_ = std::max(R_, zi_l2norm_[i]);
  }

  lambda_max_ = max_j_abs_sum_xi;
  if (lambda_ > lambda_max_) {
    lambda_ = lambda_max_;
  }

  all_fea_index_.resize(num_fea_);
  std::iota(std::begin(all_fea_index_), std::end(all_fea_index_), 0);

  std::copy(std::begin(all_fea_index_), std::end(all_fea_index_),
            std::back_inserter(nfs_index_));
  w_nnz_index_.clear();
  w_nnz_index_.insert(std::begin(all_fea_index_), std::end(all_fea_index_));

  fea_screen_target_.clear();
  fea_screen_target_.insert(std::begin(all_fea_index_),
                            std::end(all_fea_index_));

  all_ins_index_.resize(num_ins_);
  std::iota(std::begin(all_ins_index_), std::end(all_ins_index_), 0);

  std::copy(std::begin(all_ins_index_), std::end(all_ins_index_),
            std::back_inserter(nss_index_));

  non_sam_screening_index_.clear();
  non_sam_screening_index_.insert(std::begin(all_ins_index_),
                                  std::end(all_ins_index_));
  sam_screen_target_.clear();
  sam_screen_target_.insert(std::begin(all_ins_index_),
                            std::end(all_ins_index_));

  sam_screening_index_.first.clear();
  sam_screening_index_.second.clear();

  established_w_sq_norm_ = 0.0;
  unestablished_fea_flag_ = Eigen::VectorXd::Ones(num_fea_);

  dif_alpha_sq_norm_ = 0.0;
  established_dual_var_ = Eigen::VectorXd::Ones(num_ins_);
  unestablished_dual_var_ = Eigen::VectorXd::Zero(num_ins_);
  unestablished_sam_flag_ = Eigen::VectorXd::Zero(num_ins_);

  max_ssfss_itr_ = 10;
  each_max_ssfss_itr_ = 2;
  fs_start_gap_ = 1.0;
  ss_start_gap_ = 1e-3;
  screening_end_rate_ = 0.95;
  flag_fs_end_ = false;
  flag_ss_end_ = false;
}

elastic_soft_insensitive::~elastic_soft_insensitive() {}

int elastic_soft_insensitive::get_num_ins(void) const { return num_ins_; }

int elastic_soft_insensitive::get_num_fea(void) const { return num_fea_; }

double elastic_soft_insensitive::get_regularized_parameter(void) const {
  return lambda_;
}

double elastic_soft_insensitive::get_min_regularized_parameter(void) const {
  return lambda_max_;
}

double elastic_soft_insensitive::get_primal_obj_value(void) const {
  return primal_obj_value_;
}

double elastic_soft_insensitive::get_dual_obj_value(void) const {
  return dual_obj_value_;
}

double elastic_soft_insensitive::get_duality_gap(void) const {
  return duality_gap_;
}

double elastic_soft_insensitive::get_stop_criterion(void) const {
  return stop_criterion_;
}

void elastic_soft_insensitive::set_regularized_parameter(const double &lam) {
  lambda_ = lam;
  duality_gap_ = std::numeric_limits<double>::max();
  itr_ = 0;
}

void elastic_soft_insensitive::set_stop_criterion(const double &eps) {
  stop_criterion_ = eps;
}

void elastic_soft_insensitive::set_ssfss_parameter(const int &msi,
                                                   const int &emsi,
                                                   const double &fsg,
                                                   const double &ssg,
                                                   const double &ser) {
  max_ssfss_itr_ = msi;
  each_max_ssfss_itr_ = emsi;
  fs_start_gap_ = fsg;
  ss_start_gap_ = ssg;
  screening_end_rate_ = ser;
}

void elastic_soft_insensitive::calculate_primal_obj_value(
    const bool &flag_cal_loss) {
  sum_loss_ = 0.0;
  if (flag_cal_loss) {
    b_ = ((x_ * primal_var_).array() - y_).abs() - epsilon_;
    double bi;
    for (int i = 0; i < num_ins_; ++i) {
      bi = b_[i];
      if (bi > gamma_) {
        sum_loss_ += bi - 0.5 * gamma_;
      } else if (bi >= 0.0) {
        sum_loss_ += 0.5 * (bi) * (bi) / gamma_;
      }
    }
  }

  primal_obj_value_ =
      lambda_ * (primal_var_.lpNorm<1>() + 0.5 * primal_var_.squaredNorm()) +
      one_over_n_ * sum_loss_;
}

void elastic_soft_insensitive::calculate_dual_obj_value(
    const bool &flag_cal_v) {
  if (flag_cal_v) {
    za_over_n_ = Eigen::VectorXd::Zero(num_fea_);
    for (int i = 0; i < num_ins_; ++i)
      za_over_n_ += dual_var_[i] * x_.row(i);
    za_over_n_ *= 1.0 / (num_ins_);
  }

  Eigen::ArrayXd za = ((1.0 / lambda_) * za_over_n_.array()).abs() - 1.0;
  dual_obj_value_ =
      one_over_n_ *
          (dual_var_.dot(y_.matrix()) - 0.5 * gamma_ * dual_var_.squaredNorm() -
           epsilon_ * dual_var_.lpNorm<1>()) -
      lambda_ * 0.5 * ((za >= 0.0).select(za, 0.0).square().sum());
}

void elastic_soft_insensitive::calculate_duality_gap(const bool &flag_cal_loss,
                                                     const bool &flag_cal_v) {
  calculate_primal_obj_value(flag_cal_loss);
  calculate_dual_obj_value(flag_cal_v);
  duality_gap_ = primal_obj_value_ - dual_obj_value_;
}

Eigen::VectorXd elastic_soft_insensitive::get_primal_var(void) const {
  return primal_var_;
}

Eigen::VectorXd elastic_soft_insensitive::get_dual_var(void) const {
  return dual_var_;
}

void elastic_soft_insensitive::claer_sfs3_index(void) {
  w_nnz_index_.clear();
  w_nnz_index_.insert(std::begin(all_fea_index_), std::end(all_fea_index_));
  fea_screen_target_.clear();
  fea_screen_target_.insert(std::begin(all_fea_index_),
                            std::end(all_fea_index_));
  nfs_index_.clear();
  std::copy(std::begin(all_fea_index_), std::end(all_fea_index_),
            std::back_inserter(nfs_index_));

  sam_screening_index_.first.clear();
  sam_screening_index_.second.clear();
  non_sam_screening_index_.clear();
  non_sam_screening_index_.insert(std::begin(all_ins_index_),
                                  std::end(all_ins_index_));
  sam_screen_target_.clear();
  sam_screen_target_.insert(std::begin(all_ins_index_),
                            std::end(all_ins_index_));
  nss_index_.clear();
  std::copy(std::begin(all_ins_index_), std::end(all_ins_index_),
            std::back_inserter(nss_index_));

  established_w_sq_norm_ = 0.0;
  unestablished_fea_flag_.setOnes();

  dif_alpha_sq_norm_ = 0.0;
  established_dual_var_.setZero();
  unestablished_dual_var_ = dual_var_;
  unestablished_sam_flag_.setOnes();
  each_max_ssfss_itr_ = 0;
  num_inv_fea_scr_ = 0;
  num_inv_sam_scr_ = 0;
  num_fea_sav_ = 0;
  num_ins_sav_ = 0;
}

void elastic_soft_insensitive::sfs3(const bool flag_init) {
  int w_nnz_size_old = w_nnz_index_.size();
  int non_scr_size_old = non_sam_screening_index_.size();

  if (non_scr_size_old == 0 || w_nnz_size_old == 0 ||
      (duality_gap_ >= fs_start_gap_ && duality_gap_ >= ss_start_gap_) ||
      (flag_fs_end_ && flag_ss_end_))
    return;

  int further_screening_itr = 0;
  int w_nnz_size = 0, non_scr_size = num_ins_;
  while (w_nnz_size_old != w_nnz_size || non_scr_size != non_scr_size_old) {
    if (further_screening_itr > each_max_ssfss_itr_)
      break;
    ++further_screening_itr;
    w_nnz_size_old = w_nnz_index_.size();
    non_scr_size_old = non_sam_screening_index_.size();

    if (!flag_fs_end_) {
      if (duality_gap_ < fs_start_gap_ || non_scr_size_old < 0.1 * num_ins_) {
        if (w_nnz_size_old == num_fea_) {
          initial_safe_feature_screen();
        } else {
          safe_feature_screen();
        }
      }
    }

    if (!flag_ss_end_) {
      if (duality_gap_ < ss_start_gap_ ||
          w_nnz_index_.size() < 0.01 * num_fea_) {
        if (non_scr_size_old == num_ins_) {
          initial_safe_sample_screen();
        } else {
          safe_sample_screen();
        }
      }
    }
    w_nnz_size = w_nnz_index_.size();
    non_scr_size = non_sam_screening_index_.size();

    if ((num_fea_ - w_nnz_size + num_inv_fea_scr_) >
        screening_end_rate_ * num_fea_)
      flag_fs_end_ = true;

    if ((sam_screening_index_.first.size() +
         sam_screening_index_.second.size() + num_inv_sam_scr_) >
        screening_end_rate_ * num_ins_)
      flag_ss_end_ = true;

    if (non_scr_size == 0 || w_nnz_size == 0)
      break;
  }
  nss_index_.clear();
  nss_index_.reserve(non_sam_screening_index_.size());
  for (auto &&i : non_sam_screening_index_) {
    nss_index_.push_back(i);
  }
  nfs_index_.clear();
  nfs_index_.reserve(w_nnz_index_.size());
  for (auto &&j : w_nnz_index_) {
    nfs_index_.push_back(j);
  }
}

void elastic_soft_insensitive::safe_feature_screen(
    const bool flag_only) {
  auto start_time = sys_clk::now();
  std::vector<int> w_zero_new, new_sav;
  double dual_radius =
      std::sqrt(std::max(0.0, 2.0 * num_ins_ * duality_gap_ / gamma_));
  double primal_radius = std::sqrt(std::max(0.0, 2.0 * duality_gap_ / lambda_));

  double max_abs_zja;
  bool flag_ss_fill = true;
  if (non_sam_screening_index_.size() == num_ins_)
    flag_ss_fill = false;

  const double lambda_n = lambda_ * num_ins_;
  for (auto &&j : fea_screen_target_) {
    bool flag_saving = false;
    if (std::abs(primal_var_[j]) > primal_radius) {
      ++num_inv_fea_scr_;
      ++num_total_fea_sav_;
      flag_saving = true;
      new_sav.push_back(j);
    }
    double unestablished_term = (dual_var_.transpose() * cm_x_.col(j))(0);

    double zju_norm = zj_l2norm_[j];
    if (flag_ss_fill) {
      zju_norm = 0.0;
      for (scm_iit it(cm_x_, j); it; ++it)
        if (unestablished_sam_flag_[it.index()])
          zju_norm += it.value() * it.value();
      zju_norm = std::sqrt(zju_norm);
    }

    double first_half = unestablished_term;
    double second_half = dual_radius * zju_norm;
    max_abs_zja = std::max(std::abs(first_half - second_half),
                           std::abs(first_half + second_half));
    if (max_abs_zja <= lambda_n) {
      w_zero_new.push_back(j);
    } else if (first_half - second_half > lambda_n ||
               first_half + second_half < -lambda_n) {
      ++num_fea_sav_;
      if (!flag_saving) {
        ++num_total_fea_sav_;
        new_sav.push_back(j);
      }
    }
  }

  for (auto &&j : w_zero_new) {
    w_nnz_index_.erase(j);
    fea_screen_target_.erase(j);
    established_w_sq_norm_ += primal_var_[j] * primal_var_[j];
    primal_var_[j] = 0.0;
    spdc_w_[j] = 0.0;
    unestablished_fea_flag_[j] = 0.0;
  }

  for (auto &&j : new_sav) {
    fea_screen_target_.erase(j);
  }

  if (flag_only) {
    nfs_index_.clear();
    nfs_index_.reserve(w_nnz_index_.size());
    for (auto &&j : w_nnz_index_) {
      nfs_index_.push_back(j);
    }
  }
  auto end_time = sys_clk::now();
  fea_dif_time_ += static_cast<double>(
      std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void elastic_soft_insensitive::safe_sample_screen(
    const bool flag_only) {
  auto start_time = sys_clk::now();

  double primal_radius = std::sqrt(
      std::max(0.0, 2.0 * duality_gap_ / lambda_));
  double dual_radius =
      std::sqrt(std::max(0.0, (2.0 * num_ins_ * duality_gap_ / gamma_)));

  std::vector<int> new_nsv0, new_nsv1, new_nsvm1, new_sav;
  double unestablished_term, tmp, lb_zuwu, ub_zuwu;
  double yi = 0.0;
  for (auto &&i : sam_screen_target_) {
    bool flag_saving = false;
    if (std::abs(dual_var_[i]) > dual_radius &&
        dual_var_[i] > -1.0 + dual_radius && dual_var_[i] < 1.0 - dual_radius) {
      ++num_inv_sam_scr_;
      ++num_total_ins_sav_;
      flag_saving = true;
      new_sav.push_back(i);
    }
    yi = y_[i];
    unestablished_term = (x_.row(i) * primal_var_)(0);
    double xiu_sq_norm = 0.0;
    for (srm_iit it(x_, i); it; ++it)
      if (unestablished_fea_flag_[it.index()])
        xiu_sq_norm += it.value() * it.value();

    tmp = std::sqrt(xiu_sq_norm) * primal_radius;

    lb_zuwu = unestablished_term - tmp;
    ub_zuwu = unestablished_term + tmp;
    if (lb_zuwu >= yi - epsilon_ && ub_zuwu <= yi + epsilon_) {
      new_nsv0.push_back(i);
    } else if (ub_zuwu <= yi - gamma_ - epsilon_) {
      new_nsv1.push_back(i);
    } else if (lb_zuwu >= yi + gamma_ + epsilon_) {
      new_nsvm1.push_back(i);
    } else if ((-gamma_ + yi - epsilon_ < lb_zuwu && ub_zuwu < yi - epsilon_) ||
               (yi + epsilon_ < lb_zuwu && ub_zuwu < gamma_ + yi + epsilon_)) {
      ++num_ins_sav_;
      if (!flag_saving) {
        ++num_total_ins_sav_;
        new_sav.push_back(i);
      }
    }
  }

  double dvi;
  for (auto &&i : new_nsv0) {
    sam_screening_index_.first.insert(i);
    non_sam_screening_index_.erase(i);
    sam_screen_target_.erase(i);
    dvi = dual_var_[i];
    za_over_n_ -= one_over_n_ * dvi * x_.row(i);
    dif_alpha_sq_norm_ += dvi * dvi;
    dual_var_[i] = 0.0;
    unestablished_dual_var_[i] = 0.0;
    unestablished_sam_flag_[i] = 0.0;
  }
  for (auto &&i : new_nsv1) {
    sam_screening_index_.second.insert(i);
    non_sam_screening_index_.erase(i);
    sam_screen_target_.erase(i);
    dvi = dual_var_[i];
    if (dvi < 1.0) {
      dvi = 1.0 - dvi;
      dif_alpha_sq_norm_ += dvi * dvi;
      za_over_n_ += one_over_n_ * dvi * x_.row(i);
      dual_var_[i] = 1.0;
    }
    established_dual_var_[i] = 1.0;
    unestablished_dual_var_[i] = 0.0;
    unestablished_sam_flag_[i] = 0.0;
  }
  for (auto &&i : new_nsvm1) {
    sam_screening_index_.second.insert(i);
    non_sam_screening_index_.erase(i);
    sam_screen_target_.erase(i);
    dvi = dual_var_[i];
    if (dvi > -1.0) {
      dvi = -1.0 - dvi;
      dif_alpha_sq_norm_ += dvi * dvi;
      za_over_n_ += one_over_n_ * dvi * x_.row(i);
      dual_var_[i] = -1.0;
    }
    established_dual_var_[i] = 1.0;
    unestablished_dual_var_[i] = 0.0;
    unestablished_sam_flag_[i] = 0.0;
  }

  for (auto &&i : new_sav) {
    sam_screen_target_.erase(i);
  }

  if (flag_only) {
    nss_index_.clear();
    nss_index_.reserve(non_sam_screening_index_.size());
    for (auto &&i : non_sam_screening_index_) {
      nss_index_.push_back(i);
    }
  }

  auto end_time = sys_clk::now();
  sam_dif_time_ += static_cast<double>(
      std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void elastic_soft_insensitive::initial_safe_feature_screen(
    const bool flag_only) {
  auto start_time = sys_clk::now();
  double dual_radius =
      std::sqrt(std::max(0.0, 2.0 * num_ins_ * duality_gap_ / gamma_));
  double primal_radius = std::sqrt(2.0 * duality_gap_ / lambda_);

  num_inv_fea_scr_ = 0;
  num_fea_sav_ = 0;
  num_total_fea_sav_ = 0;
  established_w_sq_norm_ = 0.0;
  unestablished_fea_flag_.setOnes();
  Eigen::ArrayXd za = ((cm_x_.transpose() * dual_var_).array());
  const double lambda_n = lambda_ * num_ins_;
  double ub_abs_za = 0.0, ub_za = 0.0, lb_za = 0.0, second_half = 0.0;
  for (int j = 0; j < num_fea_; ++j) {
    bool flag_saving = false;
    if (std::abs(primal_var_[j]) > primal_radius) {
      ++num_inv_fea_scr_;
      ++num_total_fea_sav_;
      flag_saving = true;
      fea_screen_target_.erase(j);
    }
    second_half = dual_radius * zj_l2norm_[j];
    ub_abs_za = std::abs(za[j]) + second_half;
    ub_za = za[j] + second_half;
    lb_za = za[j] - second_half;
    if (ub_abs_za <= lambda_n) {
      w_nnz_index_.erase(j);
      fea_screen_target_.erase(j);
      established_w_sq_norm_ += primal_var_[j] * primal_var_[j];
      primal_var_[j] = 0.0;
      spdc_w_[j] = 0.0;
      unestablished_fea_flag_[j] = 0.0;
    } else if (lb_za > lambda_n || ub_za < -lambda_n) {
      ++num_fea_sav_;
      if (!flag_saving) {
        ++num_total_fea_sav_;
        fea_screen_target_.erase(j);
      }
    }
  }
  if (flag_only) {
    nfs_index_.clear();
    nfs_index_.reserve(w_nnz_index_.size());
    for (auto &&j : w_nnz_index_) {
      nfs_index_.push_back(j);
    }
  }
  auto end_time = sys_clk::now();
  fea_dif_time_ += static_cast<double>(
      std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void elastic_soft_insensitive::initial_safe_sample_screen(
    const bool flag_only) {
  auto start_time = sys_clk::now();
  double primal_radius = std::sqrt(2.0 * duality_gap_ / lambda_);
  double dual_radius =
      std::sqrt(std::max(0.0, 2.0 * num_ins_ * duality_gap_ / gamma_));
  num_inv_sam_scr_ = 0;
  num_ins_sav_ = 0;
  num_total_ins_sav_ = 0;
  dif_alpha_sq_norm_ = 0.0;
  established_dual_var_.setZero();
  unestablished_dual_var_ = dual_var_;
  unestablished_sam_flag_.setOnes();

  Eigen::ArrayXd lbzw = (x_ * primal_var_).array();
  Eigen::ArrayXd ubzw = lbzw + primal_radius * zi_l2norm_;
  lbzw -= primal_radius * zi_l2norm_;
  double dvi = 0.0;
  double yi;
  for (int i = 0; i < num_ins_; ++i) {
    dvi = dual_var_[i];
    yi = y_[i];
    bool flag_saving = false;
    if (std::abs(dvi) > dual_radius && dvi > -1.0 + dual_radius &&
        dvi < 1.0 - dual_radius) {
      ++num_inv_sam_scr_;
      ++num_total_ins_sav_;
      flag_saving = true;
      sam_screen_target_.erase(i);
    }
    if (lbzw[i] >= yi - epsilon_ && ubzw[i] <= yi + epsilon_) {
      sam_screening_index_.first.insert(i);
      non_sam_screening_index_.erase(i);
      sam_screen_target_.erase(i);
      za_over_n_ -= one_over_n_ * dvi * x_.row(i);
      dif_alpha_sq_norm_ += dvi * dvi;
      dual_var_[i] = 0.0;
      unestablished_dual_var_[i] = 0.0;
      unestablished_sam_flag_[i] = 0.0;
    } else if (ubzw[i] <= yi - gamma_ - epsilon_) {
      sam_screening_index_.second.insert(i);
      non_sam_screening_index_.erase(i);
      sam_screen_target_.erase(i);
      if (dvi < 1.0) {
        dvi = 1.0 - dvi;
        za_over_n_ += one_over_n_ * dvi * x_.row(i);
        dif_alpha_sq_norm_ += dvi * dvi;
        dual_var_[i] = 1.0;
      }
      established_dual_var_[i] = 1.0;
      unestablished_dual_var_[i] = 0.0;
      unestablished_sam_flag_[i] = 0.0;
    } else if (lbzw[i] >= yi + gamma_ + epsilon_) {
      sam_screening_index_.second.insert(i);
      non_sam_screening_index_.erase(i);
      sam_screen_target_.erase(i);
      if (dvi > -1.0) {
        dvi = -1.0 - dvi;
        za_over_n_ += one_over_n_ * dvi * x_.row(i);
        dif_alpha_sq_norm_ += dvi * dvi;
        dual_var_[i] = -1.0;
      }
      established_dual_var_[i] = 1.0;
      unestablished_dual_var_[i] = 0.0;
      unestablished_sam_flag_[i] = 0.0;
    } else if ((-gamma_ + yi - epsilon_ < lbzw[i] && ubzw[i] < yi - epsilon_) ||
               (yi + epsilon_ < lbzw[i] && ubzw[i] < gamma_ + yi + epsilon_)) {
      ++num_ins_sav_;
      if (!flag_saving) {
        ++num_total_ins_sav_;
        sam_screen_target_.erase(i);
      }
    }
  }
  if (flag_only) {
    nss_index_.clear();
    nss_index_.reserve(non_sam_screening_index_.size());
    for (auto &&i : non_sam_screening_index_) {
      nss_index_.push_back(i);
    }
  }
  auto end_time = sys_clk::now();
  sam_dif_time_ += static_cast<double>(
      std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void elastic_soft_insensitive::inverse_screen(void) {
  num_inv_fea_scr_ = 0;
  num_inv_sam_scr_ = 0;
  double primal_radius = std::sqrt(
      std::max(0.0, 2.0 * duality_gap_ / lambda_ - established_w_sq_norm_));
  double dual_radius = std::sqrt(std::max(
      0.0, 2.0 * num_ins_ * duality_gap_ / gamma_ - dif_alpha_sq_norm_));

  if (dual_radius < 1.0) {
    for (int i = 0; i < num_ins_; ++i) {
      if (std::abs(dual_var_[i]) > dual_radius &&
          dual_var_[i] > -1.0 + dual_radius &&
          dual_var_[i] < 1.0 - dual_radius) {
        ++num_inv_sam_scr_;
        sam_screen_target_.erase(i);
      }
    }
  }

  for (int j = 0; j < num_fea_; ++j) {
    if (std::abs(primal_var_[j]) > primal_radius) {
      ++num_inv_fea_scr_;
      fea_screen_target_.erase(j);
    }
  }
}

std::vector<int> elastic_soft_insensitive::screening_for_plot_inverse(void) {

  claer_sfs3_index();
  initial_safe_feature_screen();
  initial_safe_sample_screen();
  int w_nnz_size_old = w_nnz_index_.size();
  int non_scr_size_old = non_sam_screening_index_.size();
  int nsv0size = sam_screening_index_.first.size();
  int nsv1size = sam_screening_index_.second.size();
  std::vector<int> result_sfs3;

  result_sfs3.push_back(num_fea_ - w_nnz_size_old);
  result_sfs3.push_back(nsv0size);
  result_sfs3.push_back(nsv1size);

  nss_index_.clear();
  nss_index_.reserve(non_sam_screening_index_.size());
  for (auto &&i : non_sam_screening_index_) {
    nss_index_.push_back(i);
  }
  nfs_index_.clear();
  nfs_index_.reserve(w_nnz_index_.size());
  for (auto &&j : w_nnz_index_) {
    nfs_index_.push_back(j);
  }
  int num_inv_fea_scr_old = num_inv_fea_scr_;
  int num_inv_sam_scr_old = num_inv_sam_scr_;
  int num_fea_sav_old = num_fea_sav_;
  int num_ins_sav_old = num_ins_sav_;
  int num_total_fea_sav_old = num_total_fea_sav_;
  int num_total_ins_sav_old = num_total_ins_sav_;
  if (non_scr_size_old == 0 || w_nnz_size_old == 0) {
    result_sfs3.push_back(num_fea_);
    result_sfs3.push_back(0);
    result_sfs3.push_back(num_ins_);
    result_sfs3.push_back(2);
    // inverse_screen();
    result_sfs3.push_back(num_inv_fea_scr_old);
    result_sfs3.push_back(num_inv_sam_scr_old);
    result_sfs3.push_back(num_fea_sav_old);
    result_sfs3.push_back(num_ins_sav_old);
    result_sfs3.push_back(num_total_fea_sav_old);
    result_sfs3.push_back(num_total_ins_sav_old);
    result_sfs3.push_back(num_inv_fea_scr_);
    result_sfs3.push_back(num_inv_sam_scr_);
    result_sfs3.push_back(num_fea_sav_);
    result_sfs3.push_back(num_ins_sav_);
    result_sfs3.push_back(num_total_fea_sav_);
    result_sfs3.push_back(num_total_ins_sav_);
    return result_sfs3;
  }

  // further screening
  int further_screening_itr = 0;
  int w_nnz_size = 0, non_scr_size = num_ins_;
  while (w_nnz_size_old != w_nnz_size || non_scr_size != non_scr_size_old) {
    ++further_screening_itr;
    w_nnz_size_old = w_nnz_index_.size();
    non_scr_size_old = non_sam_screening_index_.size();
    safe_feature_screen();
    safe_sample_screen();
    w_nnz_size = w_nnz_index_.size();
    non_scr_size = non_sam_screening_index_.size();
  }

  nsv0size = sam_screening_index_.first.size();
  nsv1size = sam_screening_index_.second.size();

  result_sfs3.push_back(num_fea_ - w_nnz_index_.size());
  result_sfs3.push_back(nsv0size);
  result_sfs3.push_back(nsv1size);
  result_sfs3.push_back(further_screening_itr);

  // inverse_screen();
  result_sfs3.push_back(num_inv_fea_scr_old);
  result_sfs3.push_back(num_inv_sam_scr_old);
  result_sfs3.push_back(num_fea_sav_old);
  result_sfs3.push_back(num_ins_sav_old);
  result_sfs3.push_back(num_total_fea_sav_old);
  result_sfs3.push_back(num_total_ins_sav_old);
  result_sfs3.push_back(num_inv_fea_scr_);
  result_sfs3.push_back(num_inv_sam_scr_);
  result_sfs3.push_back(num_fea_sav_);
  result_sfs3.push_back(num_ins_sav_);
  result_sfs3.push_back(num_total_fea_sav_);
  result_sfs3.push_back(num_total_ins_sav_);
  return result_sfs3;
}
}
