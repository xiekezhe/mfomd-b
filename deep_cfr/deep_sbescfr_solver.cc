// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "deep_sbescfr_solver.h"

#include <memory>
#include <numeric>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {


double adapt(const DeepCFRConfig& config, double alpha,
             const std::vector<double>& values) {
  double sum_values = 0;
  double max_values = 0;
  static const double scale_up_coeff_ = config.cfr_rm_amp;
  static const double scale_down_coeff_ = config.cfr_rm_damp;
  static const double scale_ub_ = config.cfr_scale_ub;
  static const double scale_lb_ = config.cfr_scale_lb;
  static const double alpha_ub_ = config.cfr_rm_ub;
  static const double alpha_lb_ = config.cfr_rm_lb;
  for (auto& v : values) {
    if (abs(v) > max_values) {
      max_values = abs(v);
    }
    sum_values += v;
  }
  if (sum_values > scale_ub_ * max_values) {
    alpha *= scale_up_coeff_;
  } else if (sum_values < scale_lb_ * max_values) {
    alpha *= scale_down_coeff_;
  }
  alpha = std::min(alpha_ub_, std::max(alpha, alpha_lb_));
  return alpha;
}

DeepSbESCFRSolver::DeepSbESCFRSolver(
    const Game& game, const DeepCFRConfig& config,
    std::vector<std::shared_ptr<VPNetEvaluator>> value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> global_value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> policy_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_evals,
    bool anticipatory,
    std::mt19937* rng)
    : game_(game.Clone()),
      config_(config),
      rng_(rng),
      iterations_(0),
      step_(0),
      player_(-1),
      avg_type_(GetAverageType(config.average_type)),
      weight_type_(GetWeightType(config.weight_type)),
      dist_(0.0, 1.0),
      value_eval_(value_evals),
      global_value_eval_(global_value_evals),
      policy_eval_(policy_evals),
      current_policy_eval_(current_policy_evals),
      tree_(game_->NewInitialState()),
      root_node_(tree_.Root()),
      root_state_(root_node_->GetState()),
      use_regret_net(config.use_regret_net),
      use_policy_net(config.use_policy_net),
      use_tabular(config.use_tabular),
      anticipatory_(anticipatory),
      eta_(config.nfsp_eta),
      epsilon_(config.nfsp_epsilon) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}

std::vector<Trajectory> DeepSbESCFRSolver::RunIteration(Player player,
                                                        double alpha, int step,
                                                        int max_iterations) {
  return RunIteration(rng_, player, alpha, step, max_iterations);
}

std::vector<Trajectory> DeepSbESCFRSolver::RunIteration(std::mt19937* rng,
                                                        Player player,
                                                        double alpha, int step,
                                                        int max_iterations) {
  alpha_ = alpha;
  node_touch_ = 0;
  max_iterations_ = max_iterations;
  if (step_ != step || player_ != player) {
    iterations_ = 0;
  }
  step_ = step;
  player_ = player;
  ++iterations_;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  // Sample a chace seed at the start of an iteration.
  ChanceData chance_data = root_state_->SampleChance(rng);
  bool current_or_average[2] = {true, true};
  if (anticipatory_) {
    current_or_average[player] = dist_(*rng) < eta_;
    current_or_average[player== 0 ? 1 : 0] = false;
  }
  double value = UpdateRegrets(root_node_, player, 1, 1, 1, 1, value_trajectory,
                               policy_trajectory, step, rng, chance_data, current_or_average);
  value_trajectory.node_touched = node_touch_;
  value_trajectory.value = value;
  return {value_trajectory, policy_trajectory};
}

double DeepSbESCFRSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double ave_opponent_reach, double sampling_reach,
    Trajectory& value_trajectory, Trajectory& policy_trajectory, int step,
    std::mt19937* rng, const ChanceData& chance_data, bool current_or_average[2]) {
  State& state = *(node->GetState());
  universal_poker::UniversalPokerState* poker_state =
      static_cast<universal_poker::UniversalPokerState*>(node->GetState());
  state.SetChance(chance_data);
  // std::cout << state.ToString() << std::endl;
  if (state.IsTerminal()) {
    double value = opponent_reach*state.PlayerReturn(player);
    value_trajectory.states.push_back(ReplayNode{
          state.InformationStateString(player), state.InformationStateTensor(player), player, std::vector<Action>{}, std::vector<double>{},false,
          value,1.0, -1,std::vector<double>{},std::vector<double>{}, std::vector<Action>{},-1,0.0, 1.0, player_reach,
          opponent_reach,
          sampling_reach});
    return value;
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, ave_opponent_reach, sampling_reach,
                         value_trajectory, policy_trajectory, step, rng,
                         chance_data, current_or_average);
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  node_touch_ += 1;

  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();
  std::vector<double> information_tensor = state.InformationStateTensor();

  CFRInfoStateValues info_state_copy(legal_actions, kInitialTableValues);//此处代表读取平均策略
  CFRInfoStateValues current_info_state(legal_actions, kInitialTableValues);
  if (step != 1) {
    // get current policy
    CFRNetModel::InferenceInputs inference_input{is_key, legal_actions,
                                                 information_tensor};
    auto cfr_policy = current_policy_eval_[cur_player]
                          ->Inference(cur_player, inference_input)
                          .value;
    current_info_state.SetPolicy(cfr_policy);
    
  }
    if (step != 1 && cur_player != player) {
    CFRNetModel::InferenceInputs inference_input{is_key,legal_actions,
                                                 information_tensor};
    auto cfr_policy = policy_eval_[cur_player]->Inference(cur_player, inference_input).value;
    double eta = 1.0 / step;
    for (auto& p : cfr_policy) {
      p = eta * 1.0 / cfr_policy.size() + (1 - eta) * p;
    }
    current_info_state.SetCumulatePolicy(cfr_policy);
  }
  if (current_or_average[cur_player]) {
    info_state_copy = current_info_state;
  } else {
   // if (step != 1 &&  cur_player != player) {
     // info_state_copy.SetPolicy(current_info_state.cumulative_policy);
  //  } else {
      std::vector<double> uniform_strategy(legal_actions.size(),
                                           1.0 / legal_actions.size());
      info_state_copy.SetPolicy(uniform_strategy);
   // }
  }
  double value = 0.0;
  std::vector<double> child_values(legal_actions.size(), 0.0);
  int aidx = 0;

  if (cur_player == player) {
    //new
   
     aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    double new_sampling_reach =
        info_state_copy.current_policy[aidx] * sampling_reach;
    //double child_reach = info_state_copy.current_policy[aidx] * player_reach;
    // Walk over all actions at my nodes delete
    //for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
     double new_reach = current_info_state.current_policy[aidx] * player_reach;
      value= UpdateRegrets(
          node->GetChild(legal_actions[aidx]), player, new_reach,
          opponent_reach, ave_opponent_reach, new_sampling_reach, value_trajectory,
          policy_trajectory, step, rng, chance_data, current_or_average);
      //value += current_info_state.current_policy[aidx] * child_values[aidx];
   // }
  } else {
    // Sample at opponent nodes.
     aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    
    double new_reach = current_info_state.current_policy[aidx] * opponent_reach;
    double new_ave_reach = 1.0 / legal_actions.size() * ave_opponent_reach;
    double new_sampling_reach =
        info_state_copy.current_policy[aidx] * sampling_reach;
    value = UpdateRegrets(node->GetChild(legal_actions[aidx]), player,
                          player_reach, new_reach, new_ave_reach,
                          new_sampling_reach, value_trajectory,
                          policy_trajectory, step, rng, chance_data, current_or_average);
  }

  if (cur_player == player) {
    CFRNetModel::InferenceInputs inference_input{is_key, legal_actions,
                                                 information_tensor};
    auto cfrvalue = value_eval_[cur_player]->Inference(state).value;
    double delta = poker_state->MaxUtility() - poker_state->MinUtility();
    double Z_weight = 1.0;
    if (weight_type_ == WeightType::kLinear) {
      Z_weight = step;
    }
    double value_weight=0.0;
    State& state_next = *(node->GetChild(legal_actions[aidx])->GetState());
    if (state_next.IsTerminal()) { value_weight=1.0;}
    else{ value_weight=(double)(state_next.LegalActions().size());}

  /*  if (!use_regret_net || use_tabular) {
      CFRNetModel::TrainInputs value_train_input{
          is_key, legal_actions, legal_actions, information_tensor, std::vector<double>{},
          1.0};
      value_eval_[cur_player]->SetCFRTabular(value_train_input);
      CFRNetModel::TrainInputs train_input{is_key,        legal_actions,
                                           legal_actions, information_tensor,
                                           std::vector<double>{},   1.0};
      current_policy_eval_[cur_player]->SetCFRTabular(train_input);
    }
    if (use_regret_net) {*/
      value_trajectory.states.push_back(ReplayNode{
          is_key, information_tensor, cur_player, legal_actions, current_info_state.current_policy,current_or_average[player],
          0, value_weight,aidx,cfrvalue,std::vector<double>{}, std::vector<Action>{},-1,0.0, 1.0, player_reach,
          delta * delta * legal_actions.size() * Z_weight * ave_opponent_reach,
          sampling_reach});
  //  }
  }

  if ((avg_type_ == AverageType::kOpponent ||
       avg_type_ == AverageType::kLinearOpponent) &&
      cur_player == player && current_or_average[cur_player]) {
    double policy_weight = 1.0;
    if (avg_type_ == AverageType::kLinearOpponent) {
      policy_weight = step;
    }
    if (!use_policy_net || use_tabular) {
      std::vector<double> policy(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        policy[aidx] = current_info_state.current_policy[aidx] * policy_weight*
                        player_reach / sampling_reach;;
      }
      CFRNetModel::TrainInputs train_input{is_key,        legal_actions,
                                           legal_actions, information_tensor,
                                           policy,        1.0};
      policy_eval_[cur_player]->AccumulateCFRTabular(train_input);
    }
    if (use_policy_net) {
      policy_trajectory.states.push_back(
          ReplayNode{is_key,
                     information_tensor,
                     cur_player,
                     legal_actions,
                     current_info_state.current_policy,false,
                     1,1.0,
                     aidx,
                     std::vector<double>{},std::vector<double>{}, std::vector<Action>{},-1,0.0,
                     policy_weight,
                     player_reach,
                     opponent_reach,
                     sampling_reach});
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
