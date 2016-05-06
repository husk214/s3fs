#include "elastic_smooth_hinge_spdc.h"

namespace fsss {

elastic_smooth_hinge_spdc::elastic_smooth_hinge_spdc(
    const std::string &libsvm_format, const double &lam, const double &gamma,
    const double &stop_cri, const int &max_iter, const int &fcg)
    : elastic_smooth_hinge(libsvm_format, lam, gamma, stop_cri, max_iter, fcg) {
  za_over_n_ = Eigen::VectorXd::Zero(num_fea_);
  b_ = dual_var_;

  tau_ = std::sqrt(gamma_ / (lambda_ * num_ins_)) / R_;
  sigma_ = std::sqrt((num_ins_ * lambda_) / gamma_) / R_;
  theta_ =
      1.0 - 1.0 / (num_ins_ + R_ * std::sqrt(num_ins_ / (lambda_ * gamma_)));
}

elastic_smooth_hinge_spdc::~elastic_smooth_hinge_spdc() {}

void elastic_smooth_hinge_spdc::set_regularized_parameter(const double &lam) {
  lambda_ = lam;
  tau_ = std::sqrt(gamma_ / (lambda_ * num_ins_)) / R_;
  sigma_ = std::sqrt((num_ins_ * lambda_) / gamma_) / R_;
  theta_ =
      1.0 - 1.0 / (num_ins_ + R_ * std::sqrt(num_ins_ / (lambda_ * gamma_)));
  duality_gap_ = std::numeric_limits<double>::max();
  spdc_w_ = primal_var_;
  claer_sfs3_index();
  flag_fs_end_ = false;
  flag_ss_end_ = false;
  fea_dif_time_ = 0.0;
  sam_dif_time_ = 0.0;
}

void elastic_smooth_hinge_spdc::init_za_over_n(void) {
  za_over_n_ = Eigen::VectorXd::Zero(num_fea_);
  for (int i = 0; i < num_ins_; ++i) {
    if (dual_var_[i] > 0.0)
      za_over_n_ += y_[i] * dual_var_[i] * x_.row(i);
  }
  za_over_n_ *= (1.0) / static_cast<double>(num_ins_);
}

void elastic_smooth_hinge_spdc::calculate_dual_obj_value(
    const bool &flag_cal_v) {
  dual_obj_value_ =
      one_over_n_ * (dual_var_.sum() - 0.5 * gamma_ * dual_var_.squaredNorm());
  for (auto &&j : nfs_index_) {
    double tmp = 0.0;
    tmp = std::max(0.0, std::abs((1.0 / (num_ins_ * lambda_)) *
                                 (dual_var_.transpose() * cm_x_.col(j))(0)) -
                            1.0);
    dual_obj_value_ -= lambda_ * 0.5 * tmp * tmp;
  }
}

void elastic_smooth_hinge_spdc::calculate_duality_gap(const bool &flag_cal_loss,
                                                      const bool &flag_cal_v) {
  calculate_primal_obj_value(flag_cal_loss);
  calculate_dual_obj_value(flag_cal_v);
  duality_gap_ = std::max(0.0, primal_obj_value_ - dual_obj_value_);
}

void elastic_smooth_hinge_spdc::train(void) {
  int i = 0;
  double yi = 0.0;
  double beta_i = 0.0;
  double delta_alpha_i = 0.0;
  double alpha_i_new = 0.0;
  double tmp_w = 0.0;
  double eta = 0.0;
  const double tau_lam = tau_ * lambda_;
  const double sig_gam = -sigma_ * gamma_ - 1.0;
  const double taulam_one = tau_lam + 1.0;
  Eigen::VectorXd delta_v(num_fea_);

  std::default_random_engine g;
  std::uniform_int_distribution<> uni_dist(0, num_ins_ - 1);

  init_za_over_n();
  calculate_duality_gap(true, false);
  const auto ins_is_begin_it = std::begin(all_ins_index_);
  auto random_it = std::next(ins_is_begin_it, uni_dist(g));

  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_dist(g));
      i = *random_it;
      yi = y_[i];
      beta_i = (sigma_ * (yi * (x_.row(i) * spdc_w_)(0) - 1.0) - dual_var_[i]) /
               (sig_gam);
      alpha_i_new = std::min(1.0, std::max(0.0, beta_i));
      delta_alpha_i = alpha_i_new - dual_var_[i];
      dual_var_[i] = alpha_i_new;
      delta_v = yi * delta_alpha_i * x_.row(i);
      for (int j = 0; j < num_fea_; ++j) {
        tmp_w = primal_var_[j] + tau_ * (za_over_n_[j] + delta_v[j]);
        if (tmp_w > tau_lam) {
          eta = (tmp_w - tau_lam) / taulam_one;
        } else if (tmp_w < -tau_lam) {
          eta = (tmp_w + tau_lam) / taulam_one;
        } else {
          eta = 0.0;
        }
        spdc_w_[j] = eta + theta_ * (eta - primal_var_[j]);
        primal_var_[j] = eta;
      }
      for (srm_iit it(x_, i); it; ++it)
        za_over_n_[it.index()] += one_over_n_ * delta_v[it.index()];
    }
    if (itr_ % frequency_cal_gap_ == 0) {
      calculate_duality_gap(true, false);
      std::cout << itr_ << " optimization end gap : " << duality_gap_ << " "
                << primal_obj_value_ << " " << dual_obj_value_ << std::endl;
    }
  }
}

void elastic_smooth_hinge_spdc::train_sfs3(const bool dynamic) {
  int i = 0;
  double yi = 0.0;
  double beta_i = 0.0;
  double delta_alpha_i = 0.0;
  double alpha_i_new = 0.0;
  double tmp_w = 0.0;
  double eta = 0.0;
  const double tau_lam = tau_ * lambda_;
  const double sig_gam = -sigma_ * gamma_ - 1.0;
  const double taulam_one = tau_lam + 1.0;
  Eigen::VectorXd delta_v(num_fea_);
  init_za_over_n();
  calculate_duality_gap(true, false);
  if (duality_gap_ <= stop_criterion_)
    return;
  double gap_pre = duality_gap_;
  int pre_sfs_itr = 1;
  if (!dynamic)
    sfs3(true);

  int nss_size = nss_index_.size();

  std::vector<std::uniform_int_distribution<>> uni_distri;
  std::uniform_int_distribution<> uni_dist(0, nss_size - 1);
  uni_distri.push_back(uni_dist);
  auto ins_is_begin_it = std::begin(nss_index_);
  std::default_random_engine g;
  auto random_it = std::next(ins_is_begin_it, uni_distri[0](g));
  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    if (nss_size == 0 || nfs_index_.size() == 0)
      return;
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      yi = y_[i];
      beta_i = (sigma_ * (yi * (x_.row(i) * spdc_w_)(0) - 1.0) - dual_var_[i]) /
               (sig_gam);
      alpha_i_new = std::min(1.0, std::max(0.0, beta_i));
      delta_alpha_i = alpha_i_new - dual_var_[i];
      dual_var_[i] = alpha_i_new;
      delta_v = yi * delta_alpha_i * x_.row(i);
      for (auto &&j : nfs_index_) {
        tmp_w = primal_var_[j] + tau_ * (za_over_n_[j] + delta_v[j]);
        if (tmp_w > tau_lam) {
          eta = (tmp_w - tau_lam) / taulam_one;
        } else if (tmp_w < -tau_lam) {
          eta = (tmp_w + tau_lam) / taulam_one;
        } else {
          eta = 0.0;
        }
        spdc_w_[j] = eta + theta_ * (eta - primal_var_[j]);
        primal_var_[j] = eta;
      }
      for (srm_iit it(x_, i); it; ++it)
        za_over_n_[it.index()] += one_over_n_ * delta_v[it.index()];
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
    // std::cout << itr_ << " optimization end gap : " << duality_gap_
    //           << " , gap_old " << gap_old << " , pov " << primal_obj_value_
    //           << ", dov " << dual_obj_value_ << std::endl;
  }
}

void elastic_smooth_hinge_spdc::train_fs(const bool dynamic) {
  int i = 0;
  double yi = 0.0;
  double beta_i = 0.0;
  double delta_alpha_i = 0.0;
  double alpha_i_new = 0.0;
  double tmp_w = 0.0;
  double eta = 0.0;
  const double tau_lam = tau_ * lambda_;
  const double sig_gam = -sigma_ * gamma_ - 1.0;
  const double taulam_one = tau_lam + 1.0;
  Eigen::VectorXd delta_v(num_fea_);
  init_za_over_n();
  calculate_duality_gap(true, false);
  if (duality_gap_ <= stop_criterion_)
    return;
  double gap_old = duality_gap_;
  double gap_pre = duality_gap_;
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
    if (nss_size == 0 || nfs_index_.size() == 0)
      return;
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      yi = y_[i];
      beta_i = (sigma_ * (yi * (x_.row(i) * spdc_w_)(0) - 1.0) - dual_var_[i]) /
               (sig_gam);
      alpha_i_new = std::min(1.0, std::max(0.0, beta_i));
      delta_alpha_i = alpha_i_new - dual_var_[i];
      dual_var_[i] = alpha_i_new;
      delta_v = yi * delta_alpha_i * x_.row(i);
      for (auto &&j : nfs_index_) {
        tmp_w = primal_var_[j] + tau_ * (za_over_n_[j] + delta_v[j]);
        if (tmp_w > tau_lam) {
          eta = (tmp_w - tau_lam) / taulam_one;
        } else if (tmp_w < -tau_lam) {
          eta = (tmp_w + tau_lam) / taulam_one;
        } else {
          eta = 0.0;
        }
        spdc_w_[j] = eta + theta_ * (eta - primal_var_[j]);
        primal_var_[j] = eta;
      }
      for (srm_iit it(x_, i); it; ++it)
        za_over_n_[it.index()] += one_over_n_ * delta_v[it.index()];
    }
    if (itr_ % frequency_cal_gap_ == 0) {
      calculate_duality_gap(true, false);
      if ((duality_gap_ / gap_old < frequency_sfs3_ || itr_ < 2) && dynamic) {
        gap_old = duality_gap_;
        initial_safe_feature_screen(true);
        if (nss_index_.size() == 0)
          return;
        ins_is_begin_it = std::begin(nss_index_);
        std::uniform_int_distribution<> u_d(0, nss_index_.size() - 1);
        uni_distri[0] = u_d;
      } else if (std::abs(gap_pre - duality_gap_) < 0.1 * stop_criterion_ &&
                 dynamic) {
        inverse_screen();
        if ((num_fea_ - w_nnz_index_.size() + num_total_fea_sav_) <
            screening_end_rate_ * num_fea_) {
          safe_feature_screen(true);
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

void elastic_smooth_hinge_spdc::train_ss(const bool dynamic) {
  int i = 0;
  double yi = 0.0;
  double beta_i = 0.0;
  double delta_alpha_i = 0.0;
  double alpha_i_new = 0.0;
  double tmp_w = 0.0;
  double eta = 0.0;
  const double tau_lam = tau_ * lambda_;
  const double sig_gam = -sigma_ * gamma_ - 1.0;
  const double taulam_one = tau_lam + 1.0;
  Eigen::VectorXd delta_v(num_fea_);
  init_za_over_n();
  calculate_duality_gap(true, false);
  if (duality_gap_ <= stop_criterion_)
    return;
  double gap_old = duality_gap_;
  double gap_pre = duality_gap_;
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
    if (nss_size == 0 || nfs_index_.size() == 0)
      return;
    for (int ir = 0; ir < num_ins_; ++ir) {
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      yi = y_[i];
      beta_i = (sigma_ * (yi * (x_.row(i) * spdc_w_)(0) - 1.0) - dual_var_[i]) /
               (sig_gam);
      alpha_i_new = std::min(1.0, std::max(0.0, beta_i));
      delta_alpha_i = alpha_i_new - dual_var_[i];
      dual_var_[i] = alpha_i_new;
      delta_v = yi * delta_alpha_i * x_.row(i);
      for (auto &&j : nfs_index_) {
        tmp_w = primal_var_[j] + tau_ * (za_over_n_[j] + delta_v[j]);
        if (tmp_w > tau_lam) {
          eta = (tmp_w - tau_lam) / taulam_one;
        } else if (tmp_w < -tau_lam) {
          eta = (tmp_w + tau_lam) / taulam_one;
        } else {
          eta = 0.0;
        }
        spdc_w_[j] = eta + theta_ * (eta - primal_var_[j]);
        primal_var_[j] = eta;
      }
      for (srm_iit it(x_, i); it; ++it)
        za_over_n_[it.index()] += one_over_n_ * delta_v[it.index()];
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
    // std::cout << itr_ << " optimization end gap : " << duality_gap_
    //           << " , gap_old " << gap_old << " , pov " << primal_obj_value_
    //           << ", dov " << dual_obj_value_ << std::endl;
  }
}

void elastic_smooth_hinge_spdc::train_for_plot_inverse(void) {
  int total_iteration = 0;
  int i = 0;
  double yi = 0.0;
  double beta_i = 0.0;
  double delta_alpha_i = 0.0;
  double alpha_i_new = 0.0;
  double tmp_w = 0.0;
  double eta = 0.0;
  const double tau_lam = tau_ * lambda_;
  const double sig_gam = -sigma_ * gamma_ - 1.0;
  const double taulam_one = tau_lam + 1.0;
  const int fre_cal_gap = static_cast<int>(0.05 * num_ins_);
  auto init_var = [&] {
    for (int i = 0; i < num_ins_; ++i)
      dual_var_[i] = 0.0;
    for (int j = 0; j < num_fea_; ++j)
      primal_var_[j] = 1.0;
  };
  init_var();
  Eigen::VectorXd delta_v(num_fea_);
  init_za_over_n();
  calculate_duality_gap(true, false);

  int nss_size = nss_index_.size();

  std::vector<std::uniform_int_distribution<>> uni_distri;
  std::uniform_int_distribution<> uni_dist(0, nss_size - 1);
  uni_distri.push_back(uni_dist);
  auto ins_is_begin_it = std::begin(nss_index_);
  std::default_random_engine g;

  double threshold = 1e-1;
  double limit_thres = -std::log10(stop_criterion_);
  std::queue<double> thresholds;
  for (int i = 1; i <= limit_thres; ++i) {
    thresholds.push(threshold);
    threshold *= 0.1;
  }
  threshold = thresholds.front();
  thresholds.pop();

  auto random_it = std::next(ins_is_begin_it, uni_distri[0](g));
  for (itr_ = 1; itr_ < max_iter_ && duality_gap_ > stop_criterion_; ++itr_) {
    for (int ir = 0; ir < fre_cal_gap; ++ir) {
      ++total_iteration;
      random_it = std::next(ins_is_begin_it, uni_distri[0](g));
      i = *random_it;
      yi = y_[i];
      beta_i = (sigma_ * (yi * (x_.row(i) * spdc_w_)(0) - 1.0) - dual_var_[i]) /
               (sig_gam);
      alpha_i_new = std::min(1.0, std::max(0.0, beta_i));
      delta_alpha_i = alpha_i_new - dual_var_[i];
      dual_var_[i] = alpha_i_new;
      delta_v = yi * delta_alpha_i * x_.row(i);
      for (auto &&j : nfs_index_) {
        tmp_w = primal_var_[j] + tau_ * (za_over_n_[j] + delta_v[j]);
        if (tmp_w > tau_lam) {
          eta = (tmp_w - tau_lam) / taulam_one;
        } else if (tmp_w < -tau_lam) {
          eta = (tmp_w + tau_lam) / taulam_one;
        } else {
          eta = 0.0;
        }
        spdc_w_[j] = eta + theta_ * (eta - primal_var_[j]);
        primal_var_[j] = eta;
      }
      for (srm_iit it(x_, i); it; ++it)
        za_over_n_[it.index()] += one_over_n_ * delta_v[it.index()];
    }
    calculate_duality_gap(true, false);
    if (duality_gap_ < threshold) {
      std::vector<int> result_screening = screening_for_plot_inverse();
      std::cout << lambda_ << " " << threshold << " ";
      for (auto &&ele : result_screening)
        printf("%d ", ele);
      std::cout << duality_gap_ << " " << total_iteration << std::endl;
      if (thresholds.empty())
        return;
      ins_is_begin_it = std::begin(nss_index_);
      std::uniform_int_distribution<> u_d(0, nss_index_.size() - 1);
      uni_distri[0] = u_d;
      threshold = thresholds.front();
      thresholds.pop();

      if (result_screening[0] == num_fea_ ||
          (result_screening[1] + result_screening[2]) == num_ins_) {
        while (1) {
          std::cout << lambda_ << " " << threshold << " ";
          for (auto &&ele : result_screening)
            printf("%d ", ele);
          std::cout << duality_gap_ << " " << total_iteration << std::endl;
          if (thresholds.empty())
            return;
          threshold = thresholds.front();
          thresholds.pop();
        }
        return;
      }
    }
  }
}
}
