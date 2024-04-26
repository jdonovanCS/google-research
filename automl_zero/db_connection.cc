#include "db_connection.h"
#include "algorithm.h"
#include "instruction.h"
#include "definitions.h"

#include <stdlib.h>
#include <ctime>
#include <iostream>

namespace automl_zero {

namespace {

using ::std::abs;
using ::std::cout;
using ::std::endl;
using ::std::fixed;
using ::std::pair;
using ::std::vector;
using ::std::string;
using ::std::shared_ptr;
using ::std::ostringstream;
using ::std::ostream;
using ::std::make_shared;

} //namespace

DB_Connection::DB_Connection(const char* db_loc)
    : db_loc_(db_loc) {
        try {
            CppSQLite3DB db;
            cout << "SQLite Version: " << db.SQLiteVersion() << endl;
            cout << endl << "Creating database for run." << endl;
            // remove(db_loc);
            cout << db_loc << endl;
	    db.open(db_loc);

            db.execDML("Create table algs(id integer not null, evol_id integer not null, setup varchar(2000), learn varchar(2000), predict varchar(2000), blob_alg BLOB, fitness REAL, PRIMARY KEY (id))");
            db.execDML("Create table diversity(id integer not null, evol_id integer not null, total_vars integer, scalar_vars integer, vector_vars integer, matrix_vars integer, total_ops integer, setup_ops integer, learn_ops integer, predict_ops integer, arith_ops integer, trig_ops integer, precalc_ops integer, linearalg_ops integer, probstat_ops integer, diversity_score REAL, fitness REAL, num_indivs integer, alg_str varchar(2000), PRIMARY KEY (id))");
            db.execDML("Create table final(id integer not null, evol_id integer not null, alg_str varchar(2000), fitness REAL, PRIMARY KEY (id))");
            db.execDML("Create table progress(id integer not null, evol_id integer not null, num_indivs integer, elapsed_secs integer, mean REAL, stdev REAL, best_fit REAL, bestfit_diversity REAL, best_alg_str varchar(2000), PRIMARY KEY (id))");
        }
        catch (CppSQLite3Exception& e) {
            std::cerr << e.errorCode() << ":" << e.errorMessage() << endl;
        }
    }   

void DB_Connection::Delete(int evol_id){
    try{
        CppSQLite3DB db;

        db.open(db_loc_);
        ostringstream stmt;
        stmt << "delete from algs where evol_id = ";
        stmt << evol_id;
	db.execDML(stmt.str().c_str());
    }

    catch (CppSQLite3Exception& e) {
        std::cerr << "Error Deleting: " << e.errorCode() << ":" << e.errorMessage() << endl;
    }
}

void DB_Connection::Insert(int evol_id, std::vector<shared_ptr<const Algorithm>> algs, std::vector<double> fitnesses){
    
    try{
        CppSQLite3DB db;

        db.open(db_loc_);
        ostringstream stmt;
        stmt << "insert into algs (evol_id, setup, predict, learn, fitness, blob_alg) values ";
        int idx = 0;
        for (shared_ptr<const Algorithm>& next_algorithm : algs){
            ostringstream setup;
            ostringstream learn;
            ostringstream predict;
            
            for (const shared_ptr<const Instruction>& instruction : next_algorithm->setup_) {
                setup << instruction->ToString();
            }
            for (const shared_ptr<const Instruction>& instruction : next_algorithm->learn_) {
                learn << instruction->ToString();
            }
            for (const shared_ptr<const Instruction>& instruction : next_algorithm->predict_) {
                predict << instruction->ToString();
            }
            if (idx == 0){
                stmt << "(" << evol_id << ",\'"; 
            }
            else {
                stmt << ",(" << evol_id << ",\'";
            }
            stmt << setup.str(); 
            stmt << "\',\'"; 
            stmt << learn.str();
            stmt << "\',\'"; 
            stmt << predict.str();
            stmt << "\',";
            stmt << fitnesses[idx];
            stmt << ",\'";

            // Serialize algorithm so that it can be stored
            std::string alg_str;
            google::protobuf::TextFormat::PrintToString(next_algorithm->ToProto(), &alg_str);
            stmt << alg_str;
            stmt << "\')";
            
            idx++;
        }
        // cout << "query: " << stmt.str() << endl;
        int nRows = db.execScalar(stmt.str().c_str());
        cout << nRows << " rows inserted" << endl;
        db.close();
    }
    catch (CppSQLite3Exception& e) {
        std::cerr << "Error Inserting: " << e.errorCode() << ":" << e.errorMessage() << endl;
    }
}

std::vector<shared_ptr<const Algorithm>> DB_Connection::Migrate(int evol_id, std::vector<shared_ptr<const Algorithm>> algs, std::vector<double> fitnesses) {
    // this function should look at the evol_id and query the database for the entries that are not associated with it.
    try{
        CppSQLite3DB db;

        db.open(db_loc_);
        ostringstream stmt;
        stmt << "select * from algs where evol_id not in (" << evol_id << ") order by random() limit " << int(algs.size() / 2) << ";";
                
        CppSQLite3Query q = db.execQuery(stmt.str().c_str());
        
        int i = 0;

        // replace bottom half
        bool replace_bottom = false;

        // replace worst performers
        bool replace_worst = false;
        std::vector<size_t> idx(fitnesses.size());
        if (replace_worst == true){
            iota(idx.begin(), idx.end(), 0);
            stable_sort(idx.begin(), idx.end(), [&fitnesses](size_t i1, size_t i2) {return fitnesses[i1] < fitnesses[i2];});
        }

        while (!q.eof())
        {
            auto alg = ParseTextFormat<SerializedAlgorithm>(q.fieldValue(5));
            shared_ptr<const Algorithm> sh_alg = make_shared<const Algorithm>(alg);
            if (replace_worst == true){
                // replace worst performers
                algs[idx[i]] = sh_alg;
            }
            else if (replace_bottom == true){
                // replace bottom half
                int half = int(algs.size()/2);
                algs[i+half] = sh_alg;
            }
            else {
                // replace top half
                algs[i] = sh_alg;
            }
            q.nextRow();
            i++;
        }
        
        cout << i << " algorithms migrated" << endl;
        
        db.close();
    }
    catch (CppSQLite3Exception& e) {
        std::cerr << "Error Migrating: " << e.errorCode() << ":" << e.errorMessage() << endl;
    }


    return algs;
}

//TODO (jdonovancs): doing this functionality in regularized evolution also. Maybe just do it once there instead of twice.
void DB_Connection::LogDiversity(int evol_id, std::vector<shared_ptr<const Algorithm>> algs, int num_indivs, std::vector<double> diversity_scores, std::vector<double> fitnesses){
    // for diversity logging, may want to remove later
    ostringstream stmt;
    stmt << "insert into diversity (evol_id, total_ops, setup_ops, predict_ops, learn_ops, total_vars, scalar_vars, vector_vars, matrix_vars, arith_ops, trig_ops, precalc_ops, linearalg_ops, probstat_ops, diversity_score, fitness, num_indivs, alg_str) values ";
    int idx = 0;
    std::vector<int> arith_op_key{0,1,2,3,4,5,6};
    std::vector<int> trig_op_key{7,8,9,10,11,12};
    std::vector<int> precalc_op_key{13,14,15,16,17};
    std::vector<int> linearalg_op_key{18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43};
    std::vector<int> probstat_op_key{44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64};
    // vector<double> total_vars;
    // vector<double> scalar_vars;
    // vector<double> vector_vars;
    // vector<double> matrix_vars;
    // vector<double> total_ops;
    // vector<double> setup_ops;
    // vector<double> learn_ops;
    // vector<double> predict_ops;

    for (shared_ptr<const Algorithm>& next_algorithm : algs){
      int scalar_vars = 0;
      int vector_vars = 0;
      int matrix_vars = 0;
      int setup_ops = 0;
      int learn_ops = 0;
      int predict_ops = 0;
      int total_ops = 0;
      int total_vars = 0;
      int arith_ops = 0;
      int trig_ops = 0;
      int precalc_ops = 0;
      int linearalg_ops = 0;
      int probstat_ops = 0;
      for (const shared_ptr<const Instruction>& instruction : next_algorithm->setup_) {
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
                
                
                // if (line.substr(2,1) == "s" && int(line[3]) > scalar_vars){
                //   scalar_vars = int(line[3]);
                // }
                // else if (line.substr(2,1) == "s" && (scalar_vars < 10 && std::isdigit(line[4]))){
                //   scalar_vars = std::stoi(line.substr(2,2));
                // }
                // else if (line.substr(2,1) == "v" && int(line[3]) > vector_vars){
                //   vector_vars = int(line[3]);
                // }
                // else if (line.substr(2,1) == "v" && (vector_vars < 10 && std::isdigit(line[4]))){
                //   vector_vars = std::stoi(line.substr(2,2));
                // }
                // else if (line.substr(2,1) == "m" && int(line[3]) > matrix_vars){
                //   matrix_vars = int(line[3]);
                // }
                // else if (line.substr(2,1) == "s" && (matrix_vars < 10 && std::isdigit(line[4]))){
                //   matrix_vars = std::stoi(line.substr(2,2));
                // }
            }
      for (const shared_ptr<const Instruction>& instruction : next_algorithm->learn_) {
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
      for (const shared_ptr<const Instruction>& instruction : next_algorithm->predict_) {
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
      std::string alg_str(next_algorithm->ToReadable());
      std::istringstream alg_stream(alg_str);
      std::string line;
      while (std::getline(alg_stream, line)){
        if (line.substr(2,1) == "s" && int(line[3]) > scalar_vars){
          scalar_vars = line[3] - '0';
        //   cout << "hit 1 " << line[3] << endl;
        }
        else if (line.substr(2,1) == "s" && (scalar_vars < 10 && std::isdigit(line[4]))){
          scalar_vars = std::stoi(line.substr(2,2));
        //   cout << "hit 2 " << line.substr(2,2) << endl;
        }
        else if (line.substr(2,1) == "v" && int(line[3]) > vector_vars){
          vector_vars = line[3] - '0';
        //   cout << "hit 3 " << line[3] << endl;
        }
        else if (line.substr(2,1) == "v" && (vector_vars < 10 && std::isdigit(line[4]))){
          vector_vars = std::stoi(line.substr(2,2));
        //   cout << "hit 4 " << line.substr(2,2) << endl;
        }
        else if (line.substr(2,1) == "m" && int(line[3]) > matrix_vars){
          matrix_vars = line[3] - '0';
        //   cout << "hit 5 " << line[3] << endl;
        }
        else if (line.substr(2,1) == "s" && (matrix_vars < 10 && std::isdigit(line[4]))){
          matrix_vars = std::stoi(line.substr(2,2));
        //   cout << "hit 6 " << line.substr(2,2) << endl;
        }
      }
    //   cout << scalar_vars << " " << vector_vars << " " << matrix_vars << endl;

      total_ops = setup_ops+learn_ops+predict_ops;
      total_vars = scalar_vars+vector_vars+matrix_vars;
      if (idx == 0){
        stmt << "(" << evol_id << "," << total_ops << "," << setup_ops << "," << learn_ops << "," << predict_ops << "," << total_vars << "," << scalar_vars << "," << vector_vars << "," << matrix_vars << "," << arith_ops << "," << trig_ops << "," << precalc_ops << "," << linearalg_ops << "," << probstat_ops << "," << diversity_scores[idx] << "," << fitnesses[idx] << "," << num_indivs << ",'" << alg_str << "')";
      }
      else {
        stmt << ",(" << evol_id << "," << total_ops << "," << setup_ops << "," << learn_ops << "," << predict_ops << "," << total_vars << "," << scalar_vars << "," << vector_vars << "," << matrix_vars << "," << arith_ops << "," << trig_ops << "," << precalc_ops << "," << linearalg_ops << "," << probstat_ops << "," << diversity_scores[idx] << "," << fitnesses[idx] << "," << num_indivs << ",'" << alg_str << "')";
      }
      idx++;
    }
    try{
        CppSQLite3DB db;

        db.open(db_loc_);
        int nRows = db.execScalar(stmt.str().c_str());
        cout << nRows << " rows inserted" << endl;
        db.close();
    }
    catch (CppSQLite3Exception& e) {
        std::cerr << "Error Logging Diversity: "<< e.errorCode() << ":" << e.errorMessage() << endl;
    }

}

void DB_Connection::LogProgress(int evol_id, int num_indivs, int elapsed_secs, double mean, double stdev, double best_fit, double bestfit_diversity, shared_ptr<const Algorithm> best_alg){
    ostringstream stmt;
    stmt << "insert into progress (evol_id, num_indivs, elapsed_secs, mean, stdev, best_fit, bestfit_diversity, best_alg_str) values ";
    std::string best_alg_str = best_alg->ToReadable();
    stmt << "(" << evol_id << "," << num_indivs << "," << elapsed_secs << "," << mean << "," << stdev << "," << best_fit << "," << bestfit_diversity << ",'" << best_alg_str << "')";
    try{
        CppSQLite3DB db;
        db.open(db_loc_);
        int nRows = db.execScalar(stmt.str().c_str());
        cout << nRows << " rows inserted" << endl;
        db.close();
        }
    catch (CppSQLite3Exception& e){
        std::cerr << "Error Logging Progress: " << e.errorCode() << ":" << e.errorMessage() << endl;
    }
}

void DB_Connection::LogFinal(int evol_id, std::string alg_str, double fitness){
    ostringstream stmt;
    stmt << "insert into final (fitness, alg_str) values ";
    stmt << "(" << fitness << ",'" << alg_str << "')";
    try{
        CppSQLite3DB db;
        db.open(db_loc_);
        int nRows = db.execScalar(stmt.str().c_str());
        cout << nRows << " rows inserted" << endl;
        db.close();
    }
    catch (CppSQLite3Exception& e){
        std::cerr << "Error Logging Final: " << e.errorCode() << ":" << e.errorMessage() << endl;
    }
}

}


