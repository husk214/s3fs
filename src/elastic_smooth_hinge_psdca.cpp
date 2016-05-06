#include "elastic_smooth_hinge_psdca.h"

namespace fsss {

elastic_smooth_hinge_psdca::elastic_smooth_hinge_psdca(
    const std::string &libsvm_format, const double &lam, const double &gamma,
    const double &stop_cri, const int &max_iter, const int &fcg)
    : elastic_smooth_hinge(libsvm_format, lam, gamma, stop_cri, max_iter, fcg),
      one_over_n_(1.0 / static_cast<double>(num_ins_)) {
  zi_sq_ = zi_l2norm_.square();
}

elastic_smooth_hinge_psdca::~elastic_smooth_hinge_psdca() {}

void elastic_smooth_hinge_psdca::set_regularized_parameter(const double &lam) {
  lambda_ = lam;
  duality_gap_ = std::numeric_limits<double>::max();
  claer_sfs3_index();
  flag_fs_end_ = false;
  flag_ss_end_ = false;
  fea_dif_time_ = 0.0;
  sam_dif_time_ = 0.0;
}

void elastic_smooth_hinge_psdca::init_w_za_over_n(void) {
  za_over_n_.setZero();
  for (int i = 0; i < num_ins_; ++i) {
    if (dual_var_[i] > 0.0)
      za_over_n_ += (y_[i] * dual_var_[i]) * x_.row(i);
  }

  za_over_n_ *= one_over_n_;

  for (int j = 0; j < num_fea_; ++j)
    primal_var_[j] =
        fsss::sign((1.0 / lambda_) * za_over_n_[j]) *
        std::max(0.0, std::abs((1.0 / lambda_) * za_over_n_[j]) - 1.0);
}

void elastic_smooth_hinge_psdca::calculate_dual_obj_value(
    const bool &flag_cal_v) {
  if (flag_cal_v)
    init_w_za_over_n();

  Eigen::ArrayXd za = ((1.0 / lambda_) * za_over_n_.array()).abs() - 1.0;
  dual_obj_value_ =
      one_over_n_ * (dual_var_.sum() - 0.5 * gamma_ * dual_var_.squaredNorm()) -
      lambda_ * 0.5 * ((za >= 0.0).select(za, 0.0).square().sum());
}

void elastic_smooth_hinge_psdca::calculate_duality_gap(
    const bool &flag_cal_loss, const bool &flag_cal_v) {
  calculate_primal_obj_value(flag_cal_loss);
  calculate_dual_obj_value(flag_cal_v);
  duality_gap_ = std::max(0.0, primal_obj_value_ - dual_obj_value_);
}

void elastic_smooth_hinge_psdca::train(void) {
  const double one_over_nlam_ = 1.0 / (num_ins_ * lambda_);
  const double one_over_lam_ = 1.0 / lambda_;
  int i = 0;
  double delta_alpha_i = 0.0;
  double alpha_i_old = 0.0;

  std::default_random_engine g;
  std::uniform_int_distribution<> uni_dist(0, num_ins_ - 1);

  init_w_za_over_n();
  calculate_duality_gap(true, false);
  const auto ins_is_begin_it = std::begin(all_ins_index_);
  auto random_it = std::next(ins_is_begin_it, uni_dist(g));
  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_dist(g));
      i = *random_it;
      alpha_i_old = dual_var_[i];
      delta_alpha_i =
          (1.0 - gamma_ * alpha_i_old - y_[i] * (x_.row(i) * primal_var_)(0)) /
          (gamma_ + zi_sq_[i] * one_over_nlam_);
      delta_alpha_i =
          std::max(-alpha_i_old, std::min(1.0 - alpha_i_old, delta_alpha_i));
      dual_var_[i] += delta_alpha_i;
      za_over_n_ += (y_[i] * delta_alpha_i * one_over_n_) * x_.row(i);
      for (int j = 0; j < num_fea_; ++j) {
        primal_var_[j] =
            fsss::sign(one_over_lam_ * za_over_n_[j]) *
            std::max(0.0, std::abs(one_over_lam_ * za_over_n_[j]) - 1.0);
      }
    }
    if ((itr_ + 1) % frequency_cal_gap_ == 0) {
      calculate_duality_gap(true, false);
      // std::cout << itr_ << " optimization end gap : " << duality_gap_ << " "
      //           << primal_obj_value_ << ", " << dual_obj_value_ << std::endl;
    }
  }
}

void elastic_smooth_hinge_psdca::train_sfs3(const bool dynamic) {
  const double one_over_nlam_ = 1.0 / (num_ins_ * lambda_);
  const double one_over_lam_ = 1.0 / lambda_;
  int i = 0;
  double delta_alpha_i = 0.0;
  double alpha_i_old = 0.0;
  double yi = 0.0;

  init_w_za_over_n();
  calculate_duality_gap(true, false);
  if (duality_gap_ <= stop_criterion_)
    return;
  double gap_pre = duality_gap_;
  int pre_sfs_itr = 1;

  if (!dynamic) {
    sfs3(true);
  }

  int nss_size = nss_index_.size();

  std::vector<std::uniform_int_distribution<>> uni_distri;
  std::uniform_int_distribution<> uni_dist(0, nss_size - 1);
  uni_distri.push_back(uni_dist);
  auto ins_is_begin_it = std::begin(nss_index_);
  std::default_random_engine g;
  auto random_it = std::next(ins_is_begin_it, uni_distri[0](g));
  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      alpha_i_old = dual_var_[i];
      yi = y_[i];
      delta_alpha_i =
          (1.0 - gamma_ * alpha_i_old - yi * (x_.row(i) * primal_var_)(0)) /
          (gamma_ + zi_sq_[i] * one_over_nlam_);
      delta_alpha_i =
          std::max(-alpha_i_old, std::min(1.0 - alpha_i_old, delta_alpha_i));
      dual_var_[i] += delta_alpha_i;

      za_over_n_ += (yi * delta_alpha_i * one_over_n_) * x_.row(i);
      for (auto &&j : nfs_index_) {
        primal_var_[j] =
            fsss::sign(one_over_lam_ * za_over_n_[j]) *
            std::max(0.0, std::abs(one_over_lam_ * za_over_n_[j]) - 1.0);
      }
    }
    if (itr_ % frequency_cal_gap_ == 0) {
      calculate_duality_gap(true, false);
      if (duality_gap_ < stop_criterion_)
        return;
      if ((duality_gap_ / gap_pre < frequency_sfs3_ || itr_ <= 2) && dynamic) {
        gap_pre = duality_gap_;
        pre_sfs_itr = itr_;
        sfs3(false);
        if (nss_index_.size() == 0)
          return;
        ins_is_begin_it = std::begin(nss_index_);
        std::uniform_int_distribution<> u_d(0, nss_index_.size() - 1);
        uni_distri[0] = u_d;
      }
    }
  }
}

void elastic_smooth_hinge_psdca::train_fs(const bool dynamic) {
  const double one_over_nlam_ = 1.0 / (num_ins_ * lambda_);
  const double one_over_lam_ = 1.0 / lambda_;
  int i = 0;
  double delta_alpha_i = 0.0;
  double alpha_i_old = 0.0;
  double yi = 0.0;

  init_w_za_over_n();
  calculate_duality_gap(true, false);
  if (duality_gap_ <= stop_criterion_)
    return;

  double gap_pre = duality_gap_;
  double gap_old = duality_gap_;

  if (!dynamic)
    initial_safe_feature_screen(true);

  int nss_size = nss_index_.size();

  std::vector<std::uniform_int_distribution<>> uni_distri;
  std::uniform_int_distribution<> uni_dist(0, nss_size - 1);
  uni_distri.push_back(uni_dist);
  auto ins_is_begin_it = std::begin(nss_index_);
  std::default_random_engine g;
  auto random_it = std::next(ins_is_begin_it, uni_distri[0](g));
  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      alpha_i_old = dual_var_[i];
      yi = y_[i];
      delta_alpha_i =
          (1.0 - gamma_ * alpha_i_old - yi * (x_.row(i) * primal_var_)(0)) /
          (gamma_ + zi_sq_[i] * one_over_nlam_);
      delta_alpha_i =
          std::max(-alpha_i_old, std::min(1.0 - alpha_i_old, delta_alpha_i));
      dual_var_[i] += delta_alpha_i;

      za_over_n_ += (yi * delta_alpha_i * one_over_n_) * x_.row(i);
      for (auto &&j : nfs_index_) {
        primal_var_[j] =
            fsss::sign(one_over_lam_ * za_over_n_[j]) *
            std::max(0.0, std::abs(one_over_lam_ * za_over_n_[j]) - 1.0);
      }
    }
    if (itr_ % frequency_cal_gap_ == 0) {
      calculate_duality_gap(true, false);
      if ((duality_gap_ / gap_old < frequency_sfs3_ || itr_ < 2) && dynamic) {
        gap_old = duality_gap_;
        initial_safe_feature_screen(true);
      } else if (std::abs(gap_pre - duality_gap_) < 0.1 * stop_criterion_ &&
                 dynamic) {
        if ((num_fea_ - w_nnz_index_.size() + num_total_fea_sav_) <
            screening_end_rate_ * num_fea_) {
          safe_feature_screen(true);
        }
      }
      gap_pre = duality_gap_;
    }
  }
}

void elastic_smooth_hinge_psdca::train_ss(const bool dynamic) {
  const double one_over_nlam_ = 1.0 / (num_ins_ * lambda_);
  const double one_over_lam_ = 1.0 / lambda_;
  int i = 0;
  double delta_alpha_i = 0.0;
  double alpha_i_old = 0.0;
  double yi = 0.0;

  init_w_za_over_n();
  calculate_duality_gap(true, false);
  if (duality_gap_ <= stop_criterion_)
    return;

  double gap_pre = duality_gap_;
  double gap_old = duality_gap_;

  if (!dynamic)
    initial_safe_sample_screen(true);

  int nss_size = nss_index_.size();

  std::vector<std::uniform_int_distribution<>> uni_distri;
  std::uniform_int_distribution<> uni_dist(0, nss_size - 1);
  uni_distri.push_back(uni_dist);
  auto ins_is_begin_it = std::begin(nss_index_);
  std::default_random_engine g;
  auto random_it = std::next(ins_is_begin_it, uni_distri[0](g));
  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      alpha_i_old = dual_var_[i];
      yi = y_[i];
      delta_alpha_i =
          (1.0 - gamma_ * alpha_i_old - yi * (x_.row(i) * primal_var_)(0)) /
          (gamma_ + zi_sq_[i] * one_over_nlam_);
      delta_alpha_i =
          std::max(-alpha_i_old, std::min(1.0 - alpha_i_old, delta_alpha_i));
      dual_var_[i] += delta_alpha_i;

      za_over_n_ += (yi * delta_alpha_i * one_over_n_) * x_.row(i);
      for (auto &&j : nfs_index_) {
        primal_var_[j] =
            fsss::sign(one_over_lam_ * za_over_n_[j]) *
            std::max(0.0, std::abs(one_over_lam_ * za_over_n_[j]) - 1.0);
      }
    }
    if (itr_ % frequency_cal_gap_ == 0) {
      calculate_duality_gap(true, false);
      if ((duality_gap_ / gap_old < frequency_sfs3_ || itr_ < 2) && dynamic) {
        gap_old = duality_gap_;
        initial_safe_sample_screen(true);
        if (nss_index_.size() == 0)
          return;
        ins_is_begin_it = std::begin(nss_index_);
        std::uniform_int_distribution<> u_d(0, nss_index_.size() - 1);
        uni_distri[0] = u_d;
      } else if (std::abs(gap_pre - duality_gap_) < 0.1 * stop_criterion_ &&
                 dynamic) {
        if ((sam_screening_index_.first.size() +
             sam_screening_index_.second.size() + num_total_ins_sav_) <
            screening_end_rate_ * num_ins_) {
          safe_sample_screen(true);
          if (nss_index_.size() == 0)
            return;
          ins_is_begin_it = std::begin(nss_index_);
          std::uniform_int_distribution<> u_d(0, nss_index_.size() - 1);
          uni_distri[0] = u_d;
        }
      }
      gap_pre = duality_gap_;
    }
  }
}
}
