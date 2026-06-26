/*
per runnare senza CMake usare:
g++ -std=c++17 Entropy_NN/main.cpp -Iinclude -Iexternal -O2 -fopenmp -o Entropy_NN/main

g++ -std=c++20 Entropy_NN/main.cpp -Iinclude -Iexternal -O2 -fopenmp -o Entropy_NN/main



./Entropy_NN/main

Con Cmake entropy_nn
cambio
per runnare il test di panna:
./tests

Spiegazione libreria:

    -distance.hpp ha le funzioni di distanze. per fare il calcolo si usa compute(). le classi invece sono in base al tipo di distanza
        ::EuclideanDistance:: per la distanza euclidea, ...

        EuclideanDistance::compute(points[0], points[0]);



    -data.hpp contiene i tipi di rrappresentazione dei punti nello spazio: ::EuclideanPoints:: per i punti in uno spazio euclideo
        ::UnitNormPoints:: per punti normalizzati, ::NormedPoints:: oer cosine/angular search



    -rand.hpp contiene le funzioni per fare sample, creare random vectors ...




Prossime cose:

    - test per vedere con che probabilita dice una cosa giusta?
    - capire bene che valore mettere di k
    - fare un analisi migliore per le variabili dell'lsh?

    Per ora se considero tutto vuoto, piu aumento R piu l'algoritmo ci mette, il che è strano

    é giusto che se faccio il sample intorno alla sfera,  comunque alla fine sto anche dicendo quanto sia "lontano il punto"


    provare con 2 dimensioni e plot
*/
#include <iostream>
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/values.hpp"
#include <cmath>
#include <unordered_map>
#include <sstream>


using namespace panna;
using namespace std;

size_t N = 10000;
size_t d = 50;
float R = 1000.0f;
float c = 2.0f;
float delta = 0.05;

constexpr uint8_t K = 4;

using Dataset  = EuclideanPoints;
using Distance = EuclideanDistance;


using Builder  = E2LSHBuilder<K, Dataset, Distance>;
using Hasher   = E2LSH<K, Dataset, Distance>;
using Value = Hasher::Value;




float estimate_entropy(const unordered_map<string, size_t>& bucket_counter, size_t total_samples)
{
    float entropy = 0.0f;

    for (const auto& [bucket, count] : bucket_counter) {

        float p = static_cast<float>(count) / total_samples;

        entropy -= p * log(p);

    }

    return entropy;
}


void displayVisitedBuckets(unordered_map<string, size_t> bucket_counter) {
    cout << "\nBuckets visited so far:\n";

    int number_of_iterations = 0;

    for (const auto& [bucket, count] : bucket_counter) {

        cout << bucket
            << " -> "
            << count
            << endl;

        number_of_iterations++;
    }
    cout << "\nDifferent number of buckets seen:" << number_of_iterations <<"\n";
}

//function to create hash tables
void build_lsh_table(Dataset& points, size_t repetitions, Hasher& hasher, vector<PrefixMap<Value>>& tables)
{
    tables.resize(repetitions);

    PrefixMap<Value>::populate_from(tables, points, hasher);
}


void build_hasher(Dataset& points, size_t repetitions, Hasher& hasher)
{
    //mettere w a mano (deviazione standard gaussiana campionata)
    Builder builder(points.get_dimensions());

    builder.fit(points);

    hasher = builder.build(repetitions);
}

//gives a bucket a unique string value
string bucket_to_string(const Value& h)
{
    stringstream ss;
    ss << h;
    return ss.str();
}

//Sample around the point q in a sphere of range radius
EuclideanPoints sample_around(const EuclideanPointHandle& q, float radius)
{
    EuclideanPoints u(q.dimensions);

    // Direzione casuale
    vector<float> direction(q.dimensions);

    float norm = 0.0f;

    // Genera un vettore gaussiano
    for (size_t i = 0; i < q.dimensions; i++) {

        direction[i] = sample_random_normal();
        norm += direction[i] * direction[i];

    }

    norm = sqrt(norm);


    // Normalizza -> punto uniforme sulla sfera
    for (size_t i = 0; i < q.dimensions; i++) {
        direction[i] /= norm;
    }


    // Estrai una distanza uniforme nella palla
    float r = radius * sample_random_01();

    // Costruisci il punto campionato
    vector<float> sampled_point(q.dimensions);


    for (size_t i = 0; i < q.dimensions; i++) {
        sampled_point[i] = q.vector[i] + r * direction[i];
    }

    u.push_back(sampled_point.begin(), sampled_point.end());


    return u;
}



//function to display points
void displayPoint(int number_of_dimentions_displayed, EuclideanPointHandle p)
{
    for(size_t i = 0; i < 20; i++) {
        cout << p.vector[i] << " ";
    }
    cout << endl;
    cout << endl;
}



vector<float> make_query(size_t d)
{
    vector<float> q(d);

    for (size_t i = 0; i < d; i++) {
        q[i] = sample_random_normal();
    }

    cout<< "\n";

    return q;
}



int main() {

    //seed per questioni randomiche in panna. fissalo per avere sempre lo stesso caso
    seed_global_rng(chrono::high_resolution_clock::now().time_since_epoch().count());



    //generates the dataset
    EuclideanPoints points(d);

    for(size_t i = 0; i < N; i++) {
        points.push_back_random();
    }

    cout << "Dataset size: " << points.size() << endl;
    cout << "Dimension: " << points.get_dimensions() << endl;
    cout << "\nFirst point:\n";
    displayPoint(20, points[0]);



    //generates the hash table
    Hasher hasher;
    build_hasher(points, 1, hasher);

    //project all point in the dataset
    vector<PrefixMap<Value>> tables;
    build_lsh_table(points, 1, hasher, tables);



    //create query
    auto q = make_query(d);

    EuclideanPoints query_set(d);
    query_set.push_back(q.begin(), q.end());

    auto q_handle = query_set[0];
    auto p_handle = points[0];

    auto dist = EuclideanDistance::compute(p_handle, q_handle);
    cout << "distance query from first point = " << dist << endl;


    //Hash the query
    vector<Value> q_hashes;
    hasher.hash(q_handle, q_hashes);

    cout << "\nQuery bucket: "<< q_hashes[0]<< " "<< endl;

    /*
    //retrieve point in the query

    //cursor has the values with same hash of the query point
    auto cursor = tables[0].create_cursor(q_hashes[0]);

    //ranges is a vecto with all the indices of the hash with hash of the cursor
    auto ranges = cursor.get_indices();

    //since hash is ordered, we know thah all value with same hash are between first and last
    auto begin = ranges[0].first;
    auto end   = ranges[0].second;

    cout << endl << "number of points in the same bucket " << end - begin << endl;
    cout << "starting from " << *begin << " to " << *(end) << endl;
    cout << "Points in same bucket:\n";

    for (auto it = begin; it != end; ++it) {
        uint32_t idx = *it;
        cout << idx << " ";
    }
    cout << endl;
    */


    //parte di sample intorno alla query

    cout << "\n=========================\n";
    cout << "\nSTART SEARCHING\n";
    cout << "\n=========================\n";
    const size_t min_iterations = 10;
    size_t t = 1;
    unordered_map<string, size_t> bucket_counter;
    bool found = false;
    float entropy = 0;


    while (!found && (t < entropy * pow(2.0f, entropy) * log(1/delta)|| t < min_iterations)) {

        cout << "\n=========================\n";
        cout << "Iteration " << int(t) << endl;
        cout << "threshold " << entropy * pow(2.0f, entropy)* log(1/delta) <<endl;

        // Campiona un punto nella palla B(q,cR)
        auto random_sample = sample_around(q_handle, c*R);
        float sample_dist = EuclideanDistance::compute(q_handle, random_sample[0]);

        cout << "Sample distance from query = " << sample_dist << endl;


        // Hash del punto campionato
        vector<Value> sample_hashes;
        hasher.hash(random_sample[0], sample_hashes);

        //count of number of times ive seen it
        string bucket_id = bucket_to_string(sample_hashes[0]);
        bucket_counter[bucket_id]++;

        cout << "Sample bucket: " << sample_hashes[0] << endl;
        cout << "Visited " << bucket_counter[bucket_id] << " times" << endl;


        //estimate entropy
        entropy = estimate_entropy(bucket_counter, t + 1);

        cout << "Estimated entropy = " << entropy << endl;

        if (bucket_counter[bucket_id] == 1){
            // Recupera il bucket
            auto cursor = tables[0].create_cursor(sample_hashes[0]);

            auto ranges = cursor.get_indices();

            auto begin = ranges[0].first;
            auto end   = ranges[0].second;

            size_t bucket_size = distance(begin, end);

            cout << "Bucket size = " << bucket_size << endl;


            // Esplora tutti i punti nel bucket
            for (auto it = begin; it != end; ++it) {

                uint32_t idx = *it;

                auto candidate = points[idx];

                float dist = EuclideanDistance::compute(q_handle,candidate);

                cout << "Point " << idx << " distance = " << dist << endl;

                // Controllo ANN
                if (dist <= c * R) {

                    cout << "\nFOUND!\n";
                    cout << "Point index: " << idx << endl;

                    cout << "Distance = " << dist << endl;

                    found = true;
                    break;
                }
            }
        }
        t++;
    }

    if (!found) {
        cout << "\nNo approximate neighbour found.\n";
    }

    //displayVisitedBuckets(bucket_counter);

    cout << "\number of iterations t "<< t - 1<< "\n";
    cout << "\nEntropy: " << pow(2.0f, entropy)* log(1/delta) * entropy << entropy << "\n";






    cout << endl;
    return 0;
}
