// Copyright 2024 The Google Research Authors.
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

#include "regularized_evolution.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <utility>

#include "algorithm.h"
#include "algorithm.pb.h"
#include "task_util.h"
#include "definitions.h"
#include "executor.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace automl_zero {

namespace {

using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::absl::Seconds;  // NOLINT
using ::std::abs;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_pair;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT

constexpr double kLn2 = 0.69314718056;
constexpr IntegerT kReductionFactor = 100;

}  // namespace

RegularizedEvolution::RegularizedEvolution(
    RandomGenerator* rand_gen, const IntegerT population_size,
    const IntegerT tournament_size, const IntegerT progress_every, 
    const bool hurdles, const double migrate_prob, const int evol_id, 
    vector<int> map_elites_grid_size, const bool qd, Generator* generator, 
    Evaluator* evaluator, Mutator* mutator, DB_Connection* db)
    : evaluator_(evaluator),
      rand_gen_(rand_gen),
      start_secs_(GetCurrentTimeNanos() / kNanosPerSecond),
      epoch_secs_(start_secs_),
      epoch_secs_last_progress_(epoch_secs_),
      num_individuals_last_progress_(std::numeric_limits<IntegerT>::min()),
      tournament_size_(tournament_size),
      progress_every_(progress_every),
      initialized_(false),
      generator_(generator),
      mutator_(mutator),
      db_(db),
      use_hurdles_(hurdles),
      hurdle_(0),
      migrate_prob_(migrate_prob),
      evol_id_(evol_id),
      population_size_(population_size),
      map_elites_grid_size_(map_elites_grid_size),
      algorithms_(population_size_, make_shared<Algorithm>()),
      fitnesses_(population_size_),
      early_fitnesses_(population_size_),
      qd_(qd),
      diversity_scores_(population_size_, 0),
      total_ops_(population_size_),
      total_vars_(population_size_),
      best_alg_(algorithms_[0]),
      best_fitness_(0.0),
      map_elites_grid_(std::accumulate(map_elites_grid_size_.begin(), map_elites_grid_size_.end(), 1.0, std::multiplies<double>()), make_shared<Algorithm>()),
      map_elites_grid_fitnesses_(map_elites_grid_.size()),
      num_individuals_(0) {}
      // can probably remove parallel_ and just check if a migrate_prob is given.
      // need to add this migrate prob or migrate_every to the proto

IntegerT RegularizedEvolution::Init() {
  // Otherwise, initialize the population from scratch.
  const IntegerT start_individuals = num_individuals_;
  std::vector<double>::iterator fitness_it = fitnesses_.begin();
  std::vector<double>::iterator early_fitness_it = early_fitnesses_.begin();
  for (shared_ptr<const Algorithm>& algorithm : algorithms_) {
    InitAlgorithm(&algorithm);
    *fitness_it = Execute(algorithm, false);
    *early_fitness_it = Execute(algorithm, true);
    ++fitness_it;
    ++early_fitness_it;
  }
  CHECK(fitness_it == fitnesses_.end());

  std::vector<double>::iterator map_elites_fitness_it = map_elites_grid_fitnesses_.begin();
  for (shared_ptr<const Algorithm>& algorithm : map_elites_grid_){
    InitAlgorithm(&algorithm);
    *map_elites_fitness_it = 0;
    ++map_elites_fitness_it;
  }
  CHECK(map_elites_fitness_it == map_elites_grid_fitnesses_.end());

  MaybeLogDiversity();
  MaybePrintProgress();
  initialized_ = true;
  return num_individuals_ - start_individuals;
}

IntegerT RegularizedEvolution::Run(const IntegerT max_train_steps,
                                   const IntegerT max_nanos) {
  CHECK(initialized_) << "RegularizedEvolution not initialized."
                      << std::endl;
  const IntegerT start_nanos = GetCurrentTimeNanos();
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  while (evaluator_->GetNumTrainStepsCompleted() - start_train_steps <
             max_train_steps &&
         GetCurrentTimeNanos() - start_nanos < max_nanos) {
    vector<double>::iterator next_fitness_it = fitnesses_.begin();
    vector<double>::iterator next_early_fitness_it = early_fitnesses_.begin();
    if (false == true)
    {
      MapElites();
    }
    else{
      bool first = true;
      for (shared_ptr<const Algorithm>& next_algorithm : algorithms_) {
        // keep the best algorithm in the population by force
        if (first == true){
          double pop_mean, pop_stdev, pop_best_fitness, pop_bestfit_diversity;
          PopulationStats(
            &pop_mean, &pop_stdev, &next_algorithm, &pop_best_fitness, &pop_bestfit_diversity);
          best_alg_ = next_algorithm;
          best_fitness_ = pop_best_fitness;
          first = false;
          continue;
        }
        // continue with selection and mutation as usual
        SingleParentSelect(&next_algorithm);
        mutator_->Mutate(1, &next_algorithm);
      
        if (hurdle_ != 0 && use_hurdles_ == true) {
          *next_early_fitness_it = Execute(next_algorithm, true);
          if (*next_early_fitness_it > hurdle_) {
            *next_fitness_it = Execute(next_algorithm, false);
          }
          else {
            *next_fitness_it = *next_early_fitness_it;
            }            
        }
        else {
        *next_fitness_it = Execute(next_algorithm, false);
        }
        
        ++next_fitness_it;
      }
      if (qd_ == true){
        vector<double>::iterator total_ops_it = total_ops_.begin();
        vector<double>::iterator total_vars_it = total_vars_.begin();
        for (shared_ptr<const Algorithm>& next_algorithm: algorithms_){
          *total_ops_it = GetTotalOps(next_algorithm);
          *total_vars_it = GetTotalVars(next_algorithm);
          ++total_ops_it;
          ++total_vars_it;
        }
        total_ops_it = total_ops_.begin();
        total_vars_it = total_vars_.begin();
        double min_div = std::numeric_limits<double>::max();
        double max_div = 0;
        for (double diversity_score : diversity_scores_){
          diversity_score = 0;
          for (double total_op : total_ops_){
            diversity_score += abs((*total_ops_it)-total_op);
          }
          for (double total_var : total_vars_){
            diversity_score += abs((*total_vars_it) - total_var);
          }
          if (diversity_score < min_div) {
            min_div = diversity_score;
          }
          if (diversity_score > max_div) {
            max_div = diversity_score;
          }
          diversity_score -= min_div;
          diversity_score /= (max_div-min_div);
          ++total_ops_it;
          ++total_vars_it;
        }
      }
    }

    
    if (use_hurdles_ == true) {
      // Sorting entire fitness vector and removing duplicate items
      std::set<double> early_fitnesses_set(early_fitnesses_.begin(), early_fitnesses_.end());
      vector<double> unique_fitnesses(early_fitnesses_set.begin(), early_fitnesses_set.end());
      
      hurdle_ = unique_fitnesses[int(unique_fitnesses.size()*.75)];
    }

    if (migrate_prob_ > 0){
      if (rand_gen_->UniformProbability() < migrate_prob_){
        cout << "inserting algs with evol id: " << evol_id_ << endl;
        db_->Delete(evol_id_);
        db_->Insert(evol_id_, algorithms_, fitnesses_);
        algorithms_ = db_->Migrate(evol_id_, algorithms_, fitnesses_);
      } 
    }

    // Sorting just the 75% percentile of items in the array, but not removing duplicate items
    // vector<double> unique_fitnesses;
    // for (IntegerT i=0; i < fitnesses_.size(); i++){
    //   unique_fitnesses.push_back(fitnesses_[i]);
    // }
    // std::nth_element(unique_fitnesses.begin(), unique_fitnesses.begin() + int(unique_fitnesses.size()*.75), unique_fitnesses.end());
    // hurdle_ = unique_fitnesses[int(unique_fitnesses.size()*.75)];
    MaybeLogDiversity();
    MaybePrintProgress();
  }
  return evaluator_->GetNumTrainStepsCompleted() - start_train_steps;
}

IntegerT RegularizedEvolution::NumIndividuals() const {
  return num_individuals_;
}

IntegerT RegularizedEvolution::PopulationSize() const {
  return population_size_;
}

IntegerT RegularizedEvolution::NumTrainSteps() const {
  return evaluator_->GetNumTrainStepsCompleted();
}

shared_ptr<const Algorithm> RegularizedEvolution::Get(
    double* fitness) {
  const IntegerT indiv_index =
      rand_gen_->UniformPopulationSize(population_size_);
  CHECK(fitness != nullptr);
  *fitness = fitnesses_[indiv_index];
  return algorithms_[indiv_index];
}

shared_ptr<const Algorithm> RegularizedEvolution::GetBest(
    double* fitness) {
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index] > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index];
    }
  }
  CHECK_NE(best_index, -1);
  *fitness = best_fitness;
  return algorithms_[best_index];
}

void RegularizedEvolution::PopulationStats(
    double* pop_mean, double* pop_stdev,
    shared_ptr<const Algorithm>* pop_best_algorithm,
    double* pop_best_fitness, double* pop_bestfit_diversity) const {
  double total = 0.0;
  double total_squares = 0.0;
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index] > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index];
    }
    const double fitness_double = static_cast<double>(fitnesses_[index]);
    total += fitness_double;
    total_squares += fitness_double * fitness_double;
  }
  CHECK_NE(best_index, -1);
  double size = static_cast<double>(population_size_);
  const double pop_mean_double = total / size;
  *pop_mean = static_cast<double>(pop_mean_double);
  double var = total_squares / size - pop_mean_double * pop_mean_double;
  if (var < 0.0) var = 0.0;
  *pop_stdev = static_cast<double>(sqrt(var));
  *pop_best_algorithm = algorithms_[best_index];
  *pop_best_fitness = best_fitness;
  *pop_bestfit_diversity = diversity_scores_[best_index];
}

void RegularizedEvolution::InitAlgorithm(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = make_shared<Algorithm>(generator_->TheInitModel());
  // TODO(ereal): remove next line. Affects random number generation.
  mutator_->Mutate(0, algorithm);
}

double RegularizedEvolution::Execute(shared_ptr<const Algorithm> algorithm, bool earlyEval=false) {
  ++num_individuals_;
  epoch_secs_ = GetCurrentTimeNanos() / kNanosPerSecond;
  if (earlyEval == true && true == true) {
    const double fitness_early = evaluator_->EarlyEvaluate(*algorithm);
    return fitness_early;
  }
  else {
    const double fitness = evaluator_->Evaluate(*algorithm);
    return fitness;
  }
  // return fitness;
}

void RegularizedEvolution::MapElites(){
  // Fill map elites predefined grid
  // For each algorithm in algorithms, check to see if it has a higher fitness
  // than the corresponding algorithm in the grid at its location, replace it if so
  // Otherwise, flag this algorithm. It needs to be set equal to a random algorithm
  // from the grid at the end of this function. It's fitness also needs to be set to
  // that of the algorithm that replaces it

  int row_length = map_elites_grid_size_[1];
  int fit_idx = 0;
  for (shared_ptr<const Algorithm> next_algorithm : algorithms_){
    int total_vars = GetTotalVars(next_algorithm);
    int total_ops = GetTotalOps(next_algorithm);
    int idx = (total_vars*row_length) + total_ops;
    if (fitnesses_[fit_idx] >= map_elites_grid_fitnesses_[idx]){
      map_elites_grid_[idx] = next_algorithm;
      map_elites_grid_fitnesses_[idx] = fitnesses_[fit_idx];
    }

    else {
      next_algorithm = nullptr;
      fitnesses_[fit_idx] = 0;
    }    
    fit_idx++;  
  }  
  vector<double>::iterator fitness_it = fitnesses_.begin(); 
    
  for (shared_ptr<const Algorithm>& next_algorithm : algorithms_){    if (next_algorithm == nullptr){
    int rand_idx = rand() % int(map_elites_grid_fitnesses_.size());
    while(map_elites_grid_[rand_idx] == nullptr){
        rand_idx = rand() % int(map_elites_grid_fitnesses_.size());
      }
    next_algorithm = map_elites_grid_[rand_idx];
    *fitness_it = map_elites_grid_fitnesses_[rand_idx];
  }
  ++fitness_it;
  }
}

//TODO (jdonovancs): Maybe make a class to keep these variables in. Or add to algorithm at some point?
int RegularizedEvolution::GetTotalOps(shared_ptr<const Algorithm> alg){
      std::vector<int> arith_op_key{0,1,2,3,4,5,6};
      std::vector<int> trig_op_key{7,8,9,10,11,12};
      std::vector<int> precalc_op_key{13,14,15,16,17};
      std::vector<int> linearalg_op_key{18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43};
      std::vector<int> probstat_op_key{44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64};
      
      int setup_ops = 0;
      int learn_ops = 0;
      int predict_ops = 0;
      int total_ops = 0;
      
      int arith_ops = 0;
      int trig_ops = 0;
      int precalc_ops = 0;
      int linearalg_ops = 0;
      int probstat_ops = 0;

      for (const shared_ptr<const Instruction>& instruction : alg->setup_) {
        if (std::find(arith_op_key.begin(), arith_op_key.end(), instruction->op_) != arith_op_key.end()){
            arith_ops++;
        }
        else if (std::find(trig_op_key.begin(), trig_op_key.end(), instruction->op_) != trig_op_key.end()){
            trig_ops++;
        }
        else if (std::find(precalc_op_key.begin(), precalc_op_key.end(), instruction->op_) != precalc_op_key.end()){
            precalc_ops++;
        }
        else if (std::find(linearalg_op_key.begin(), linearalg_op_key.end(), instruction->op_) != linearalg_op_key.end()){
            linearalg_ops++;
        }
        else if (std::find(probstat_op_key.begin(), probstat_op_key.end(), instruction->op_) != probstat_op_key.end()){
            probstat_ops++;
        }
        setup_ops++;
      }
      for (const shared_ptr<const Instruction>& instruction : alg->learn_) {
        if (std::find(arith_op_key.begin(), arith_op_key.end(), instruction->op_) != arith_op_key.end()){
            arith_ops++;
        }
        else if (std::find(trig_op_key.begin(), trig_op_key.end(), instruction->op_) != trig_op_key.end()){
            trig_ops++;
        }
        else if (std::find(precalc_op_key.begin(), precalc_op_key.end(), instruction->op_) != precalc_op_key.end()){
            precalc_ops++;
        }
        else if (std::find(linearalg_op_key.begin(), linearalg_op_key.end(), instruction->op_) != linearalg_op_key.end()){
            linearalg_ops++;
        }
        else if (std::find(probstat_op_key.begin(), probstat_op_key.end(), instruction->op_) != probstat_op_key.end()){
            probstat_ops++;
        }
        
        learn_ops++;
      }
      for (const shared_ptr<const Instruction>& instruction : alg->predict_) {
        if (std::find(arith_op_key.begin(), arith_op_key.end(), instruction->op_) != arith_op_key.end()){
            arith_ops++;
        }
        else if (std::find(trig_op_key.begin(), trig_op_key.end(), instruction->op_) != trig_op_key.end()){
            trig_ops++;
        }
        else if (std::find(precalc_op_key.begin(), precalc_op_key.end(), instruction->op_) != precalc_op_key.end()){
            precalc_ops++;
        }
        else if (std::find(linearalg_op_key.begin(), linearalg_op_key.end(), instruction->op_) != linearalg_op_key.end()){
            linearalg_ops++;
        }
        else if (std::find(probstat_op_key.begin(), probstat_op_key.end(), instruction->op_) != probstat_op_key.end()){
            probstat_ops++;
        }
        
        predict_ops++;
      }
      total_ops = setup_ops+learn_ops+predict_ops;
      return total_ops;
}

int RegularizedEvolution::GetTotalVars(shared_ptr<const Algorithm> alg){
  int scalar_vars = 0;
  int vector_vars = 0;
  int matrix_vars = 0;
  int total_vars = 0;
  std::string alg_str(alg->ToReadable());
  std::istringstream alg_stream(alg_str);
  std::string line;
  while (std::getline(alg_stream, line)){
    if (line.substr(2,1) == "s" && int(line[3]) > scalar_vars){
      scalar_vars = line[3] - '0';
    }
    else if (line.substr(2,1) == "s" && (scalar_vars < 10 && std::isdigit(line[4]))){
      scalar_vars = std::stoi(line.substr(2,2));
    }
    else if (line.substr(2,1) == "v" && int(line[3]) > vector_vars){
      vector_vars = line[3] - '0';
    }
    else if (line.substr(2,1) == "v" && (vector_vars < 10 && std::isdigit(line[4]))){
      vector_vars = std::stoi(line.substr(2,2));
    }
    else if (line.substr(2,1) == "m" && int(line[3]) > matrix_vars){
      matrix_vars = line[3] - '0';
    }
    else if (line.substr(2,1) == "s" && (matrix_vars < 10 && std::isdigit(line[4]))){
      matrix_vars = std::stoi(line.substr(2,2));
    }
  }
  total_vars = scalar_vars+vector_vars+matrix_vars;
  return total_vars;
}

shared_ptr<const Algorithm>
    RegularizedEvolution::BestFitnessTournament() {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1;
  for (IntegerT tour_idx = 0; tour_idx < tournament_size_; ++tour_idx) {
    const IntegerT algorithm_index =
        rand_gen_->UniformPopulationSize(population_size_);
    double curr_fitness = 0;
    if (qd_ == true){
      curr_fitness = fitnesses_[algorithm_index] + diversity_scores_[algorithm_index];
    }
    else
    {
      curr_fitness = fitnesses_[algorithm_index];
    }
    if (best_index == -1 || curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness;
      best_index = algorithm_index;
    }
  }
  return algorithms_[best_index];
}

void RegularizedEvolution::SingleParentSelect(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = BestFitnessTournament();
}

void RegularizedEvolution::MaybePrintProgress() {
  if (num_individuals_ < num_individuals_last_progress_ + progress_every_) {
    return;
  }
  num_individuals_last_progress_ = num_individuals_;
  double pop_mean, pop_stdev, pop_best_fitness, pop_bestfit_diversity;
  shared_ptr<const Algorithm> pop_best_algorithm;
  PopulationStats(
      &pop_mean, &pop_stdev, &pop_best_algorithm, &pop_best_fitness, &pop_bestfit_diversity);
  if (pop_best_fitness > best_fitness_){
    best_fitness_ = pop_best_fitness;
    best_alg_ = pop_best_algorithm;
  }
  double avg_diversity = std::accumulate(diversity_scores_.begin(), diversity_scores_.end(), 0.0) / int(diversity_scores_.size());
  std::cout << "indivs=" << num_individuals_ << ", " << setprecision(0) << fixed
            << "elapsed_secs=" << epoch_secs_ - start_secs_ << ", "
            << "mean=" << setprecision(6) << fixed << pop_mean << ", "
            << "stdev=" << setprecision(6) << fixed << pop_stdev << ", "
            << "best fit=" << setprecision(6) << fixed << pop_best_fitness << ", "
            << "best_overall=" << setprecision(6) << fixed << best_fitness_ << ", "
            << "mean_diversity=" << setprecision(6) << fixed << avg_diversity
            << "," << std::endl;
  std::cout.flush();
  db_->LogProgress(evol_id_, num_individuals_, epoch_secs_-start_secs_, pop_mean, pop_stdev, pop_best_fitness, pop_bestfit_diversity, pop_best_algorithm);
}

void RegularizedEvolution::MaybeLogDiversity() {
  if (num_individuals_ < num_individuals_last_progress_ + progress_every_) {
    return;
  }
  db_->LogDiversity(evol_id_, algorithms_, num_individuals_, diversity_scores_, fitnesses_);
}

void RegularizedEvolution::MaybePrintMapElites() {
  if (num_individuals_ < num_individuals_last_progress_ + progress_every_) {
    return;
  }
  for (double fitness : map_elites_grid_fitnesses_){
    cout << fitness << ", ";
  }
  for (shared_ptr<const Algorithm>& algorithm : map_elites_grid_){
    cout << algorithm->ToReadable() << endl;
  }
}

}  // namespace automl_zero
