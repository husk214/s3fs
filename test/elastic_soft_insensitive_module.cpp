#include "elastic_soft_insensitive.h"
#include "elastic_soft_insensitive_spdc.h"
#include "elastic_soft_insensitive_psdca.h"

using namespace std;
using namespace Eigen;
using namespace fsss;
template <typename T> class TD;
int type;
double lam_l;
double lam_u;
double eps;
double epsilon_insensitive;
double gam;
int max_iter;
double rate;
int num_split;
double lam_min_rate;
int fre_fcg;
bool flag_sfs3_dynamic;

void exit_with_help(char *prob_name) {
  printf(
      "For Elastic Net Smoothed hinge loss\n"
      "Usage: %s [options] libsvm_format_dataset_name   \n"
      "                                        \n"
      " [ options ] : \n"
      // " -l  : lambda_min for model selection (default 1e-3) \n"
      " -u  : lambda \n"
      " -n  : epsilon of epsilon insensitive (default 1.0) \n"
      " -g  : gamma : smoothed hinge's parameter > 0      (default 0.5) \n"
      " -e  ; stop criterion for optimization        (default 1e-6)\n"
      " -i  : max outer iteration in optimization    (default 1000)\n"
      " -r  : rate (default 0.98)                                  \n"
      " -p  : the number of splits                   (default 100) \n"
      " -m  : lambda min rate (C_max = lam_min_rate * lam_l) (default 1e-3)\n"
      " -f  : frequency calculate duality gap                (default 1)   \n"
      " -d  : sfs3 static = 0 or dynamic = 1 flag            (default 1)\n",
      prob_name);
  std::cout << std::endl;
  exit(1);
}

void parse_command_line(int argc, char **argv, string &input_file_name,
                        string &input_file_name2) {
  int i;
  // default values
  type = 3;
  lam_l = 1e-3;
  lam_u = 1e+0;
  gam = 0.1;
  eps = 1e-4;
  epsilon_insensitive = 0.5;
  max_iter = 10000000;
  rate = 0.98;
  num_split = 100;
  lam_min_rate = 1e-4;
  fre_fcg = 1;
  flag_sfs3_dynamic = true;
  // parse options
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
      break;
    if (++i >= argc)
      exit_with_help(argv[0]);
    switch (argv[i - 1][1]) {
    case 's':
      type = atoi(argv[i]);
      break;
    case 'p':
      num_split = atoi(argv[i]);
      break;
    case 'l':
      lam_l = atof(argv[i]);
      break;
    case 'u':
      lam_u = atof(argv[i]);
      break;
    case 'i':
      max_iter = atoi(argv[i]);
      break;
    case 'n':
      epsilon_insensitive = atof(argv[i]);
      break;
    case 'g':
      gam = atof(argv[i]);
      break;
    case 'e':
      eps = atof(argv[i]);
      break;
    case 'r':
      rate = atof(argv[i]);
      break;
    case 'm':
      lam_min_rate = atof(argv[i]);
      break;
    case 'f':
      fre_fcg = atoi(argv[i]);
      break;
    case 'd':
      flag_sfs3_dynamic = static_cast<bool>(atoi(argv[i]));
      break;
    default:
      fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
      exit_with_help(argv[0]);
      break;
    }
  }

  // set_print_string_function(print_func);

  // determine filenames
  if (i >= argc)
    exit_with_help(argv[0]);

  input_file_name = argv[i];
  input_file_name2 = "none";
  if (i < argc - 1) {
    input_file_name2 = argv[i + 1];
  }
}

int main(int argc, char **argv) {
  string input_file_name;
  string input_file_name2;
  parse_command_line(argc, argv, input_file_name, input_file_name2);

  switch (type) {
  case 1: {
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    cout << "end construct" << endl;
    ensi.set_regularized_parameter(lam_u);
    ensi.train();
    // for (int i = 0; i < ensi.num_ins_; ++i) {
    //  if (std::abs(ensi.dual_var_[i]) < 1.0 - 1e-3)
    //    printf(" (%i:%f) ", i, ensi.dual_var_[i]);
    //}
    cout << ensi.duality_gap_ << endl;
    break;
  }
  case 11: {
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    // ensi.set_regularized_parameter(lam_u);
    // ensi.train();
    ensi.set_regularized_parameter(lam_u);
    ensi.train_sfs3(flag_sfs3_dynamic);
    cout << ensi.non_sam_screening_index_.size() << std::endl;
    for (int i = 0; i < ensi.num_fea_; ++i) {
      if (std::abs(ensi.primal_var_[i]) < 1000)
        printf(" (%i:%f) ", i, ensi.primal_var_[i]);
    }
    cout << ensi.epsilon_;
    break;
  }
  case 14: {
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    auto start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now;
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train();
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;
      pv = ensi.get_primal_var();
      dv = ensi.get_dual_var();
      int w0_size = sdm::compare(pv, 0.0, 1e-6);
      int nsv0_size = sdm::compare(dv, 0.0, 1e-6);
      int nsv1_size = sdm::compare(dv, 1.0, 1e-4);
      int nsvm1_size = sdm::compare(dv, -1.0, 1e-4);
      std::cout << " " << w0_size << " " << nsv0_size << " "
                << nsv1_size + nsvm1_size << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << std::endl;
    }
    break;
  }
  case 15: {
    // ssfss(freqency = 0.1 gap) for model selection
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    auto start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now << " ";
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train_sfs3(flag_sfs3_dynamic);
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;

      std::cout << 1e-3 * ensi.fea_dif_time_ << " " << 1e-3 * ensi.sam_dif_time_
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << " "
                << ensi.num_fea_ - ensi.w_nnz_index_.size() << " "
                << ensi.num_ins_ - ensi.non_sam_screening_index_.size() << " "
                << std::endl;
    }
    break;
  }
  case 16: {
    // featrue screeing(freqency = 0.1 gap) for model selection
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    std::chrono::time_point<sys_clk> start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now << " ";
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train_fs(flag_sfs3_dynamic);
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;

      std::cout << 1e-3 * ensi.fea_dif_time_ << " " << 1e-3 * ensi.sam_dif_time_
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << " "
                << ensi.num_fea_ - ensi.w_nnz_index_.size() << std::endl;
    }
    break;
  }
  case 17: {
    // sample screening (freqency = 0.1 gap) for model selection
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    auto start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now << " ";
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train_ss(flag_sfs3_dynamic);
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;

      std::cout << 1e-3 * ensi.fea_dif_time_ << " " << 1e-3 * ensi.sam_dif_time_
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << std::endl;
    }
    break;
  }
  case 19: {
    // for plot screening rate with inverse screening
    elastic_soft_insensitive_spdc ensi(input_file_name, lam_l,
                                       epsilon_insensitive, gam, eps, max_iter,
                                       fre_fcg);
    ensi.set_ssfss_parameter(100000, 1.0, 1.0, 1.0);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      ensi.set_regularized_parameter(lam_now);
      ensi.train_for_plot_inverse();
    }
    break;
  }
  case 2: {
    elastic_soft_insensitive_psdca ensi(
        input_file_name, lam_l, epsilon_insensitive, gam, eps, 1000, fre_fcg);
    cout << "end construct" << endl;
    ensi.set_regularized_parameter(lam_u);
    ensi.train();
    // for (int i = 0; i < ensi.num_ins_; ++i) {
    //  if (std::abs(ensi.dual_var_[i]) < 1.0 - 1e-3)
    //    printf(" (%i:%f) ", i, ensi.dual_var_[i]);
    //}
    cout << ensi.duality_gap_ << endl;
    break;
  }
  case 21: {
    elastic_soft_insensitive_psdca ensi(
        input_file_name, lam_l, epsilon_insensitive, gam, eps, 1000, fre_fcg);
    cout << "end construct" << endl;
    ensi.set_regularized_parameter(lam_u);
    ensi.train_sfs3();
    // for (int i = 0; i < ensi.num_ins_; ++i) {
    //  if (std::abs(ensi.dual_var_[i]) < 1.0 - 1e-3)
    //    printf(" (%i:%f) ", i, ensi.dual_var_[i]);
    //}
    cout << ensi.duality_gap_ << endl;
    break;
  }
  case 24: {
    elastic_soft_insensitive_psdca ensi(input_file_name, lam_l,
                                        epsilon_insensitive, gam, eps, max_iter,
                                        fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    auto start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now;
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train();
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;
      pv = ensi.get_primal_var();
      dv = ensi.get_dual_var();
      int w0_size = sdm::compare(pv, 0.0, 1e-6);
      int nsv0_size = sdm::compare(dv, 0.0, 1e-6);
      int nsv1_size = sdm::compare(dv, 1.0, 1e-4);
      int nsvm1_size = sdm::compare(dv, -1.0, 1e-4);
      std::cout << " " << w0_size << " " << nsv0_size << " "
                << nsv1_size + nsvm1_size << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << std::endl;
    }
    break;
  }
  case 25: {
    // ssfss(freqency = 0.1 gap) for model selection
    elastic_soft_insensitive_psdca ensi(input_file_name, lam_l,
                                        epsilon_insensitive, gam, eps, max_iter,
                                        fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    auto start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now << " ";
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train_sfs3(flag_sfs3_dynamic);
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;

      std::cout << 1e-3 * ensi.fea_dif_time_ << " " << 1e-3 * ensi.sam_dif_time_
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << " "
                << ensi.num_fea_ - ensi.w_nnz_index_.size() << " "
                << ensi.num_ins_ - ensi.non_sam_screening_index_.size() << " "
                << std::endl;
    }
    break;
  }
  case 26: {
    // featrue screeing(freqency = 0.1 gap) for model selection
    elastic_soft_insensitive_psdca ensi(input_file_name, lam_l,
                                        epsilon_insensitive, gam, eps, max_iter,
                                        fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    std::chrono::time_point<sys_clk> start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now << " ";
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train_fs(flag_sfs3_dynamic);
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;

      std::cout << 1e-3 * ensi.fea_dif_time_ << " " << 1e-3 * ensi.sam_dif_time_
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << " "
                << ensi.num_fea_ - ensi.w_nnz_index_.size() << std::endl;
    }
    break;
  }
  case 27: {
    // sample screening (freqency = 0.1 gap) for model selection
    elastic_soft_insensitive_psdca ensi(input_file_name, lam_l,
                                        epsilon_insensitive, gam, eps, max_iter,
                                        fre_fcg);
    double lam_start = ensi.lambda_max_ / ensi.num_ins_;
    double lam_end = lam_start * lam_min_rate;
    double log_interval = (log10(lam_start) - log10(lam_end)) / num_split;
    double lam_now;
    Eigen::VectorXd pv, dv;
    auto start_time = sys_clk::now();
    auto pre_start_time = start_time;
    auto end_time = start_time;
    auto dif_time = end_time - start_time;
    auto step_dif = end_time - pre_start_time;
    for (int i = num_split; i > 0; --i) {
      lam_now = std::pow(10.0, (log10(lam_end) + i * log_interval));
      std::cout << -i + num_split + 1 << " " << lam_now << " ";
      ensi.set_regularized_parameter(lam_now);
      pre_start_time = sys_clk::now();
      ensi.train_ss(flag_sfs3_dynamic);
      end_time = sys_clk::now();
      dif_time = end_time - start_time;
      step_dif = end_time - pre_start_time;

      std::cout << 1e-3 * ensi.fea_dif_time_ << " " << 1e-3 * ensi.sam_dif_time_
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(step_dif).count()
                << " "
                << 1e-3 * std::chrono::duration_cast<mil_sec>(dif_time).count()
                << " " << ensi.duality_gap_ << " " << ensi.itr_ << std::endl;
    }
    break;
  }
  default: { std::cout << "default" << std::endl; }
  }
  return 0;
}

void print_null(const char *s) {}
