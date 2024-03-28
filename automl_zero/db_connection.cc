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
        std::cerr << e.errorCode() << ":" << e.errorMessage() << endl;
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
        std::cerr << e.errorCode() << ":" << e.errorMessage() << endl;
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
        std::cerr << e.errorCode() << ":" << e.errorMessage() << endl;
    }


    return algs;
}

}


