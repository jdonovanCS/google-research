#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <unistd.h>


#include "algorithm.h"
#include "task_util.h"
#include "task.pb.h"
#include "definitions.h"
#include "instruction.pb.h"
#include "evaluator.h"
#include "experiment.pb.h"
#include "experiment_util.h"
#include "random_generator.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/time.h"

#include "db_connection.h"

typedef automl_zero::IntegerT IntegerT;
typedef automl_zero::RandomSeedT RandomSeedT;
typedef automl_zero::InstructionIndexT InstructionIndexT;

ABSL_FLAG(
    IntegerT, evol_id, 0,
    "The evol_id of the run in which we are attempting stacking"
    " If `0`, runs on across all runs in the database");
// ABSL_FLAG(
//     std::string, database_loc, "",
//     "The database to use for obtaining evol_id and algorithm.");
ABSL_FLAG(
    std::string, experiment_name, "",
    "The name of the experiment and database for this run. Required.");
ABSL_FLAG(
    std::string, final_tasks, "",
    "The tasks to use for the final evaluation. Must be a TaskCollection "
    "proto in text format. Required.");
ABSL_FLAG(
    RandomSeedT, random_seed, 0,
    "Seed for random generator. Use `0` to not specify a seed (creates a new "
    "seed each time). If running multiple experiments, this seed is set at the "
    "beginning of the first experiment. Does not affect tasks.");

namespace automl_zero {

namespace {
using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::strcpy;
using ::std::strcat;
}  // namespace

void run2() {  
  
  cout << "executing stacking here" << endl;
  char db_loc[100];
//   strcpy(db_loc, "/home/jordan/");
  strcpy(db_loc, "/p/work/rditljtd/automl_zero/");
  strcat(db_loc, GetFlag(FLAGS_experiment_name).c_str());
  strcat(db_loc, ".db3");
  cout << db_loc << endl;
  DB_Connection db(db_loc);
  
  const auto final_tasks =
      ParseTextFormat<TaskCollection>(GetFlag(FLAGS_final_tasks));
  RandomSeedT random_seed = GetFlag(FLAGS_random_seed);
  if (random_seed == 0) {
    random_seed = GenerateRandomSeed();
  }
  mt19937 bit_gen(random_seed);
  RandomGenerator rand_gen(&bit_gen);
  mt19937 final_bit_gen(rand_gen.UniformRandomSeed());
  RandomGenerator final_rand_gen(&final_bit_gen);
  std::unique_ptr<mt19937> bit_gen_o = make_unique<mt19937>(GenerateRandomSeed());
  mt19937* bit_gen2 = bit_gen_o.get();
  std::unique_ptr<RandomGenerator> rand_gen_o = make_unique<RandomGenerator>(bit_gen2);
  RandomGenerator* rand_gen2 = rand_gen_o.get();

  Evaluator final_evaluator(
      MEAN_FITNESS_COMBINATION,
      final_tasks,
      &final_rand_gen,
      nullptr,  // functional_cache
      nullptr,  // train_budget
      100.0);


  shared_ptr<const Algorithm> best_algorithm = make_shared<const Algorithm>();
  int evol_id=GetFlag(FLAGS_evol_id);
  best_algorithm = db.getBestAlgorithm(evol_id);
  // Stack the final solution for each algorithmic section (setup, predict, learn)
  // And do a final evaluation with it on the same unseen tasks.
  int stack_repeat = 3;
  shared_ptr<Algorithm> stacked_best_algorithm = make_shared<Algorithm>();
  auto stacked = make_unique<Algorithm>(*stacked_best_algorithm);
  for (int i=0; i<stack_repeat; i++){
    for (const shared_ptr<const Instruction> instruction : best_algorithm->setup_) {
        vector<shared_ptr<const Instruction>>* component_function = &stacked_best_algorithm->setup_;
        const InstructionIndexT position = stacked->setup_.size() + 1;
        const Op op = instruction->op_;
        component_function->insert(component_function->begin() + position, make_shared<const Instruction>(op, rand_gen2));
        // component_function->insert(component_function->begin() + position, make_shared<const Instruction>());
    }
    for (const shared_ptr<const Instruction>& instruction : best_algorithm->learn_) {
        vector<shared_ptr<const Instruction>>* component_function = &stacked_best_algorithm->learn_;
        const InstructionIndexT position = stacked->learn_.size() + 1;
        const Op op = instruction->op_;
        component_function->insert(component_function->begin() + position, make_shared<const Instruction>(op, rand_gen2));
        // component_function->insert(component_function->begin() + position, make_shared<const Instruction>());
    }
    for (const shared_ptr<const Instruction>& instruction : best_algorithm->predict_) {
        vector<shared_ptr<const Instruction>>* component_function = &stacked_best_algorithm->predict_;
        const InstructionIndexT position = stacked->predict_.size() + 1;
        const Op op = instruction->op_;
        component_function->insert(component_function->begin() + position, make_shared<const Instruction>(op, rand_gen2));
        // component_function->insert(component_function->begin() + position, make_shared<const Instruction>());
    }
  }

  cout << endl;
  cout << "Final evaluation of best algorithm stacked thrice "
       << "(on the same tasks above)..." << endl;
  const double final_stacked_fitness = 
      final_evaluator.Evaluate(*stacked_best_algorithm);
  
  cout << "Final evaluation fitness for stacked algorithm (on same data above) = "
       << final_stacked_fitness << endl;
  cout << "Stacked Algorithm: " << endl
       << stacked_best_algorithm->ToReadable() << endl;

}

}  // namespace automl_zero

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  automl_zero::run2();
  return 0;
}
